import asyncio
from abc import ABC, abstractmethod
from typing import Any

from dostuff.helpers.memory.resolve_memory_operation import resolve_memory_operation
from dostuff.lib.memory.chroma import get_chroma_client


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

    async def upsert_fact(self, user_id: str, fact: dict) -> None:
        """Low-level write — used by resolve_memory_operation, not called directly elsewhere."""
        key = str(fact.get("key", "")).strip()
        value = str(fact.get("value", "")).strip()
        if not key or not value:
            return
        await asyncio.to_thread(
            self._collection.upsert,
            ids=[f"{user_id}:{key}"],
            documents=[f"User's {key.replace('_', ' ')} is {value}"],
            metadatas=[{"user_id": user_id, "key": key, "value": value, "source": str(fact.get("source") or "user_message")}],
        )

    async def query(self, user_id: str, query_text: str, top_k: int = 3) -> list[dict]:
        """Similar memories as structured {key, value, source} dicts."""
        def _query() -> Any:
            # n_results larger than the collection makes Chroma warn; cap it
            n = min(top_k, self._collection.count())
            if n <= 0:
                return {}
            return self._collection.query(query_texts=[query_text], n_results=n, where={"user_id": user_id})

        results = await asyncio.to_thread(_query)
        raw_metas = results.get("metadatas") or [[]]
        metadatas = raw_metas[0] if raw_metas and raw_metas[0] is not None else []
        return [_as_fact(meta) for meta in metadatas if meta]

    async def list_all(self, user_id: str) -> list[dict]:
        """Every saved fact for this user (for /memories)."""
        results = await asyncio.to_thread(self._collection.get, where={"user_id": user_id})
        facts = [_as_fact(meta) for meta in (results.get("metadatas") or []) if meta]
        return sorted(facts, key=lambda f: f["key"])


def _as_fact(meta: Any) -> dict:
    return {
        "key": str(meta.get("key") or ""),
        "value": str(meta.get("value") or ""),
        "source": str(meta.get("source") or "unknown"),
    }
