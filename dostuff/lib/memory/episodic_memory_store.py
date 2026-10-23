import asyncio
from abc import ABC, abstractmethod
import json
import math
import time
from typing import Any

from dostuff.lib.memory.chroma import get_chroma_client

# Episodes less similar than this to the prompt are left out: without a floor, the top_k nearest
# are recalled even for an unrelated prompt like "hi" (unrelated prompts score ~0.0-0.2)
MIN_EPISODE_SIMILARITY = 0.3


class EpisodicStore(ABC):
    @abstractmethod
    async def add(self, episodes: list[dict]) -> None:
        """Persist a list of formatted atomic episode objects."""
        ...

    @abstractmethod
    async def query(
        self, user_id: str, query_text: str, top_k: int = 3, working_dir: str | None = None
    ) -> list[dict]:
        """Retrieve relevant and recent episodic memories."""
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
        recency_decay_rate: float = 0.05,
        min_similarity: float = MIN_EPISODE_SIMILARITY,
    ) -> list[dict]:
        """Episodes of this user from any past session (only from `working_dir` when given),
        at least `min_similarity` similar to `query_text`, re-ranked with an exponential recency decay.
        """
        candidate_k = max(top_k * 3, 10)
        where: dict[str, Any] = {"user_id": user_id}
        if working_dir:
            where = {"$and": [{"user_id": user_id}, {"working_dir": working_dir}]}

        def _query() -> Any:
            n = min(candidate_k, self._collection.count())
            if n <= 0:
                return {}
            return self._collection.query(query_texts=[query_text], n_results=n, where=where)

        results = await asyncio.to_thread(_query)

        raw_docs = results.get("documents") or [[]]
        raw_metas = results.get("metadatas") or [[]]
        raw_dists = results.get("distances") or [[]]

        documents = raw_docs[0] if raw_docs and raw_docs[0] is not None else []
        metadatas = raw_metas[0] if raw_metas and raw_metas[0] is not None else []
        distances = raw_dists[0] if raw_dists and raw_dists[0] is not None else []

        now_epoch = int(time.time())
        scored_episodes: list[dict[str, Any]] = []

        for doc, meta, dist in zip(documents, metadatas, distances):
            # 1. Convert cosine distance to vector similarity score [0, 1]
            similarity = 1.0 - dist
            if similarity < min_similarity:
                continue

            # 2. Compute age in days based on absolute timestamp
            raw_epoch = meta.get("created_at_epoch")
            created_at_epoch = int(raw_epoch) if isinstance(raw_epoch, (int, float)) else 0
            days_old = max(0.0, (now_epoch - created_at_epoch) / 86400.0)

            # 3. Apply exponential time decay: Score = Similarity * e^(-decay * days)
            time_factor = math.exp(-recency_decay_rate * days_old)
            hybrid_score = similarity * time_factor

            raw_tags = meta.get("tags")
            tags = json.loads(raw_tags) if isinstance(raw_tags, str) else []

            raw_context = meta.get("context_window")
            context_window = json.loads(raw_context) if isinstance(raw_context, str) else []

            scored_episodes.append({
                "score": hybrid_score,
                "event_type": meta.get("event_type"),
                "anchor_event": meta.get("anchor_event"),
                "summary": meta.get("summary"),
                "tags": tags,
                "context_window": context_window,
                "created_at_iso": meta.get("created_at_iso"),
                "session_id": meta.get("session_id"),
            })

        # Sort by hybrid score descending and pick top_k
        scored_episodes.sort(key=lambda x: float(x["score"]), reverse=True)
        return scored_episodes[:top_k]
