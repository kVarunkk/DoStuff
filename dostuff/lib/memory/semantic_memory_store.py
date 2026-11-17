import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any, Literal

from dostuff.helpers.memory.resolve_memory_operation import resolve_memory_operation
from dostuff.lib.memory.chroma import get_chroma_client
from dostuff.lib.memory.safe_query import safe_query

# Facts less similar than this to the prompt aren't recalled: unrelated prompts like "hi" score
# ~0.0-0.2 against stored facts, relevant ones ~0.3-0.5 (Chroma's default MiniLM embeddings)
MIN_FACT_SIMILARITY = 0.25

# "user": true of the person in every project; "project": a rule about one codebase (memory recall spec D4)
FactScope = Literal["user", "project"]


class MemoryStore(ABC):
    @abstractmethod
    async def add(self, user_id: str, facts: list[dict]) -> None: ...

    @abstractmethod
    async def query(self, user_id: str, query_text: str, top_k: int = 5) -> list[dict]: ...


class SemanticMemoryStore(MemoryStore):
    def __init__(self, persist_path: str = ".dostuff/data/chroma"):
        self._client = get_chroma_client(persist_path)
        self._collection = self._client.get_or_create_collection(name="semantic_memory", metadata={"hnsw:space": "cosine"})

    async def add(self, user_id: str, facts: list[dict]) -> None:
        for fact in facts or []:
            await resolve_memory_operation(user_id, fact, self)

    async def upsert_fact(self, user_id: str, fact: dict, scope: FactScope = "user", working_dir: str | None = None) -> None:
        """Low-level write — used by resolve_memory_operation and by consolidate (after the user confirms a merge).

        The document is the value alone (what a query is compared with); `created_at_epoch` survives an update.
        """
        key = str(fact.get("key", "")).strip()
        value = str(fact.get("value", "")).strip()
        if not key or not value:
            return
        row_id = fact_id(user_id, key, scope, working_dir)
        meta: dict[str, Any] = {"user_id": user_id, "key": key, "value": value, "scope": scope,
                                "source": str(fact.get("source") or "user_message")}
        if scope == "project":
            meta["working_dir"] = _require_dir(working_dir)

        def _write() -> None:
            now = int(time.time())
            old = self._collection.get(ids=[row_id], include=["metadatas"])
            old_metas = old.get("metadatas") or []
            old_meta = old_metas[0] if old_metas and old_metas[0] else {}
            meta["created_at_epoch"] = int(old_meta.get("created_at_epoch") or now)
            meta["updated_at_epoch"] = now
            self._collection.upsert(ids=[row_id], documents=[value], metadatas=[meta])

        await asyncio.to_thread(_write)

    async def query(self, user_id: str, query_text: str, top_k: int = 3, min_similarity: float = 0.0,
                    scope: FactScope | None = None, working_dir: str | None = None) -> list[dict]:
        """Memories at least `min_similarity` similar to `query_text`, as fact dicts (see `_as_fact`) with their `similarity`.

        `scope` and `working_dir` select facts as in `list_all`.
        """
        # safe_query caps n_results at the collection size and survives a broken HNSW index,
        # so a save or a recall degrades instead of failing (chroma-core/chroma#7032)
        where = _where(user_id, scope, working_dir)
        results = await asyncio.to_thread(safe_query, self._collection, query_text, top_k, where)
        raw_metas = results.get("metadatas") or [[]]
        raw_dists = results.get("distances") or [[]]
        metadatas = raw_metas[0] if raw_metas and raw_metas[0] is not None else []
        distances = raw_dists[0] if raw_dists and raw_dists[0] is not None else []
        return [{**_as_fact(meta), "similarity": 1.0 - dist} for meta, dist in zip(metadatas, distances)
                if meta and 1.0 - dist >= min_similarity]

    async def list_all(self, user_id: str, scope: FactScope | None = None, working_dir: str | None = None) -> list[dict]:
        """Saved facts, newest first (rows written before timestamps existed last).

        `scope="user"`: user facts; `scope="project"`: `working_dir`'s facts; `scope=None`: user facts plus
        `working_dir`'s facts, or every fact of the user when `working_dir` is None too.
        """
        results = await asyncio.to_thread(self._collection.get, where=_where(user_id, scope, working_dir))
        facts = [_as_fact(meta) for meta in (results.get("metadatas") or []) if meta]
        return sorted(facts, key=lambda f: (-f["updated_at_epoch"], f["scope"] != "project", f["key"]))

    async def delete(self, user_id: str, key: str, scope: FactScope = "user", working_dir: str | None = None) -> bool:
        """Forget one fact by key (a user fact, or `working_dir`'s project fact); False if there was none."""
        row_id = fact_id(user_id, key, scope, working_dir)
        found = await asyncio.to_thread(self._collection.get, ids=[row_id])
        if not found.get("ids"):
            return False
        await asyncio.to_thread(self._collection.delete, ids=[row_id])
        return True

    async def delete_ids(self, row_ids: list[str]) -> None:
        """Drop these rows by id (`fact_id`); ids not present are ignored."""
        if row_ids:
            await asyncio.to_thread(self._collection.delete, ids=list(row_ids))


def _require_dir(working_dir: str | None) -> str:
    if not working_dir:
        raise ValueError("A project fact needs the project's working_dir.")
    return working_dir


def fact_id(user_id: str, key: str, scope: FactScope = "user", working_dir: str | None = None) -> str:
    """Row id: `{user}:{key}` for a user fact, `{user}:p{sha1(working_dir)[:12]}:{key}` for a project fact."""
    if scope == "project":
        digest = hashlib.sha1(_require_dir(working_dir).encode("utf-8")).hexdigest()[:12]
        return f"{user_id}:p{digest}:{key}"
    return f"{user_id}:{key}"


def _where(user_id: str, scope: FactScope | None, working_dir: str | None) -> dict:
    """Chroma filter for one scope. A row with no `scope` (written before scopes existed) is a user fact,
    so "user" is `scope != project`, which Chroma also matches when the key is missing."""
    if scope == "user":
        return {"$and": [{"user_id": user_id}, {"scope": {"$ne": "project"}}]}
    if scope == "project":
        return {"$and": [{"user_id": user_id}, {"scope": "project"}, {"working_dir": _require_dir(working_dir)}]}
    if working_dir is not None:  # user facts plus this project's
        return {"$and": [{"user_id": user_id},
                         {"$or": [{"scope": {"$ne": "project"}}, {"working_dir": working_dir}]}]}
    return {"user_id": user_id}


def _as_fact(meta: Any) -> dict:
    scope = "project" if meta.get("scope") == "project" else "user"
    fact = {
        "key": str(meta.get("key") or ""),
        "value": str(meta.get("value") or ""),
        "source": str(meta.get("source") or "unknown"),
        "scope": scope,
        "created_at_epoch": int(meta.get("created_at_epoch") or 0),
        "updated_at_epoch": int(meta.get("updated_at_epoch") or 0),
    }
    if scope == "project":
        fact["working_dir"] = str(meta.get("working_dir") or "")
    return fact
