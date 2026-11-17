import asyncio
from abc import ABC, abstractmethod
import json
import math
import time
from collections.abc import Collection
from typing import Any

from dostuff.lib.memory.chroma import get_chroma_client
from dostuff.lib.memory.safe_query import safe_query

# Episodes less similar than this to the prompt are left out: without a floor, the top_k nearest
# are recalled even for an unrelated prompt like "hi" (unrelated prompts score ~0.0-0.2)
MIN_EPISODE_SIMILARITY = 0.3

# An episode at least this similar to one already saved in the same directory is the same moment
# saved twice. Rewordings of one moment score ~0.75-0.9 (even under another event type); a related
# but different moment ~0.6
DUPLICATE_EPISODE_SIMILARITY = 0.85
# Between this and DUPLICATE_EPISODE_SIMILARITY a new episode may be the same moment reworded or a related one; the
# model is asked which (memory recall D10)
SIMILAR_EPISODE_SIMILARITY = 0.6
# Most episodes kept per project; past it the lowest decayed score goes, never a DECISION (memory recall D10)
EPISODE_CAP_PER_PROJECT = 200

# Recency decay per day by event type (memory recall D9): a decision, preference or correction stays true until it is
# replaced, a lesson stays useful for months (half-life ~70 days), an event is about now (half-life ~14 days). With
# one flat 0.05 a 150-day-old lesson kept 0.06% of its similarity and lost to anything newer.
RECENCY_DECAY = {"DECISION": 0.0, "PREFERENCE": 0.0, "CORRECTION": 0.0, "LEARNING": 0.01, "EVENT": 0.05}
UNKNOWN_TYPE_DECAY = 0.05
# Scores this close are a tie, and the newer episode goes first
TIE_SCORE = 0.02


def decay_rate(event_type: Any) -> float:
    return RECENCY_DECAY.get(str(event_type or "").upper(), UNKNOWN_TYPE_DECAY)


def rank_episodes(episodes: list[dict]) -> list[dict]:
    """Best score first; a run of episodes within TIE_SCORE of the run's best is ordered newest first."""
    by_score = sorted(episodes, key=lambda e: float(e["score"]), reverse=True)
    ranked: list[dict] = []
    while by_score:
        top = float(by_score[0]["score"])
        tied = [e for e in by_score if top - float(e["score"]) <= TIE_SCORE]
        by_score = by_score[len(tied):]
        ranked += sorted(tied, key=lambda e: int(e.get("created_at_epoch") or 0), reverse=True)
    return ranked


class EpisodicStore(ABC):
    @abstractmethod
    async def add(self, episodes: list[dict]) -> None:
        """Persist a list of formatted atomic episode objects."""
        ...

    @abstractmethod
    async def query(
        self, user_id: str, query_text: str, top_k: int = 3, working_dir: str | None = None, *,
        exclude_ids: Collection[str] = (),
    ) -> list[dict]:
        """Retrieve relevant and recent episodic memories, each with its `id`, none of `exclude_ids`."""
        ...


class EpisodicMemoryStore(EpisodicStore):
    def __init__(self, persist_path: str = ".dostuff/data/chroma"):
        self._client = get_chroma_client(persist_path)
        self._collection = self._client.get_or_create_collection(
            name="episodic_memory",
            metadata={"hnsw:space": "cosine"}
        )

    async def add(self, episodes: list[dict]) -> None:
        """Batch inserts atomic episode objects extracted from session history."""
        if not episodes:
            return
        await asyncio.to_thread(
            self._collection.upsert,
            ids=[ep["id"] for ep in episodes],
            documents=[ep["vector_text"] for ep in episodes],
            metadatas=[ep["metadata"] for ep in episodes],
        )

    async def query(
        self,
        user_id: str,
        query_text: str,
        top_k: int = 3,
        working_dir: str | None = None,
        recency_decay_rate: float | None = None,
        min_similarity: float = MIN_EPISODE_SIMILARITY,
        *,
        exclude_ids: Collection[str] = (),
        stats: dict[str, Any] | None = None,
        event_type: str | None = None,
    ) -> list[dict]:
        """Episodes of this user from any past session (only from `working_dir` when given),
        at least `min_similarity` similar to `query_text`, re-ranked with an exponential recency decay
        whose rate depends on the event type (`RECENCY_DECAY`; `recency_decay_rate` forces one rate for all).

        `exclude_ids` (the episodes this session already shows the model, memory recall D8) are dropped before the
        top_k cut, and the candidate pool grows by as many, so excluding never leaves fewer results than exist.
        `event_type`, when given, keeps only episodes of that type.
        `stats`, when given, gets `matching`: how many candidates passed the floor before the cut, as text, with a
        "+" when the whole pool matched (only a pool is fetched, so there may be more).
        """
        excluded = set(exclude_ids)
        candidate_k = max(top_k * 3, 10) + len(excluded)
        conditions: list[dict[str, Any]] = [{"user_id": user_id}]
        if working_dir:
            conditions.append({"working_dir": working_dir})
        if event_type:
            conditions.append({"event_type": event_type})
        where: dict[str, Any] = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        def _query() -> Any:
            return safe_query(self._collection, query_text, candidate_k, where)

        results = await asyncio.to_thread(_query)

        raw_ids = results.get("ids") or [[]]
        raw_docs = results.get("documents") or [[]]
        raw_metas = results.get("metadatas") or [[]]
        raw_dists = results.get("distances") or [[]]

        ids = raw_ids[0] if raw_ids and raw_ids[0] is not None else []
        documents = raw_docs[0] if raw_docs and raw_docs[0] is not None else []
        metadatas = raw_metas[0] if raw_metas and raw_metas[0] is not None else []
        distances = raw_dists[0] if raw_dists and raw_dists[0] is not None else []

        now_epoch = int(time.time())
        scored_episodes: list[dict[str, Any]] = []

        for episode_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            if episode_id in excluded:
                continue
            # 1. Convert cosine distance to vector similarity score [0, 1]
            similarity = 1.0 - dist
            if similarity < min_similarity:
                continue

            # 2. Compute age in days based on absolute timestamp
            raw_epoch = meta.get("created_at_epoch")
            created_at_epoch = int(raw_epoch) if isinstance(raw_epoch, (int, float)) else 0
            days_old = max(0.0, (now_epoch - created_at_epoch) / 86400.0)

            # 3. Apply exponential time decay: Score = Similarity * e^(-decay * days)
            rate = decay_rate(meta.get("event_type")) if recency_decay_rate is None else recency_decay_rate
            time_factor = math.exp(-rate * days_old)
            hybrid_score = similarity * time_factor

            raw_tags = meta.get("tags")
            tags = json.loads(raw_tags) if isinstance(raw_tags, str) else []

            scored_episodes.append({
                "id": str(episode_id),
                "score": hybrid_score,
                "event_type": meta.get("event_type"),
                "anchor_event": meta.get("anchor_event"),
                "summary": meta.get("summary"),
                "tags": tags,
                "created_at_iso": meta.get("created_at_iso"),
                "created_at_epoch": created_at_epoch,
                "session_id": meta.get("session_id"),
            })

        if stats is not None:
            # A full pool whose farthest candidate still passed the floor: more beyond it may match too
            pool_full = len(ids) >= candidate_k and bool(distances) and 1.0 - distances[-1] >= min_similarity
            stats["matching"] = f"{len(scored_episodes)}{'+' if pool_full else ''}"

        # Best decayed score first, ties newest first (D9), then top_k
        return rank_episodes(scored_episodes)[:top_k]

    async def nearest(
        self, user_id: str, vector_text: str, working_dir: str, min_similarity: float = SIMILAR_EPISODE_SIMILARITY
    ) -> dict | None:
        """The episode from `working_dir` most similar to `vector_text`, if at least `min_similarity`: its `id`,
        `similarity`, `event_type`, `anchor_event` and `summary` (what save_episode checks a new episode against)."""
        where = {"$and": [{"user_id": user_id}, {"working_dir": working_dir}]}

        def _query() -> Any:
            return safe_query(self._collection, vector_text, 1, where)

        results = await asyncio.to_thread(_query)
        ids = (results.get("ids") or [[]])[0] or []
        dists = (results.get("distances") or [[]])[0] or []
        metas = (results.get("metadatas") or [[]])[0] or []
        if not ids or not dists or 1.0 - dists[0] < min_similarity:
            return None
        meta = (metas[0] if metas else None) or {}
        return {
            "id": str(ids[0]),
            "similarity": 1.0 - dists[0],
            "event_type": str(meta.get("event_type") or ""),
            "anchor_event": str(meta.get("anchor_event") or ""),
            "summary": str(meta.get("summary") or ""),
        }

    async def prune(self, user_id: str, working_dir: str, cap: int = EPISODE_CAP_PER_PROJECT) -> list[str]:
        """Delete this user's episodes from `working_dir` beyond `cap`, lowest decayed score first (with no prompt to
        compare against, the score is the D9 recency factor alone; equal scores go oldest first). DECISIONs are never
        deleted, so a project with more than `cap` decisions keeps them all. Returns the deleted ids."""
        where = {"$and": [{"user_id": user_id}, {"working_dir": working_dir}]}
        found = await asyncio.to_thread(self._collection.get, where=where, include=["metadatas"])
        ids, metas = found.get("ids") or [], found.get("metadatas") or []
        excess = len(ids) - cap
        if excess <= 0:
            return []
        now_epoch = int(time.time())

        def created(meta: dict) -> int:
            raw = meta.get("created_at_epoch")
            return int(raw) if isinstance(raw, (int, float)) else 0

        candidates = [
            (math.exp(-decay_rate(meta.get("event_type")) * max(0.0, (now_epoch - created(meta)) / 86400.0)),
             created(meta), str(episode_id))
            for episode_id, meta in zip(ids, metas)
            if str((meta or {}).get("event_type") or "").upper() != "DECISION"
        ]
        doomed = [episode_id for _, _, episode_id in sorted(candidates)[:excess]]
        if doomed:
            await asyncio.to_thread(self._collection.delete, ids=doomed)
        return doomed

    async def count(self, user_id: str, working_dir: str | None = None) -> int:
        """How many episodes this user has saved (only from `working_dir` when given): `saved_in_project` (D8)."""
        where: dict[str, Any] = {"user_id": user_id}
        if working_dir:
            where = {"$and": [{"user_id": user_id}, {"working_dir": working_dir}]}
        found = await asyncio.to_thread(self._collection.get, where=where, include=[])
        return len(found.get("ids") or [])

    async def list_all(self, user_id: str, working_dir: str | None = None) -> list[dict]:
        """Every saved episode for this user (only from `working_dir` when given), newest first (for /episodes)."""
        where: dict[str, Any] = {"user_id": user_id}
        if working_dir:
            where = {"$and": [{"user_id": user_id}, {"working_dir": working_dir}]}
        results = await asyncio.to_thread(self._collection.get, where=where)
        episodes = [
            {
                "id": episode_id,
                "event_type": str(meta.get("event_type") or ""),
                "anchor_event": str(meta.get("anchor_event") or ""),
                "created_at_iso": str(meta.get("created_at_iso") or ""),
                "created_at_epoch": meta.get("created_at_epoch") if isinstance(meta.get("created_at_epoch"), (int, float)) else 0,
            }
            for episode_id, meta in zip(results.get("ids") or [], results.get("metadatas") or [])
            if meta
        ]
        return sorted(episodes, key=lambda ep: ep["created_at_epoch"], reverse=True)

    async def delete(self, user_id: str, episode_id: str) -> bool:
        """Forget one episode of this user by id; False if there was none."""
        found = await asyncio.to_thread(self._collection.get, ids=[episode_id], where={"user_id": user_id})
        if not found.get("ids"):
            return False
        await asyncio.to_thread(self._collection.delete, ids=[episode_id])
        return True

    async def delete_ids(self, episode_ids: list[str]) -> None:
        """Drop these episodes by id; ids not present are ignored."""
        if episode_ids:
            await asyncio.to_thread(self._collection.delete, ids=list(episode_ids))
