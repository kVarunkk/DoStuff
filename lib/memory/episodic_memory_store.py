from abc import ABC, abstractmethod
import json
import math
import time
import chromadb

class EpisodicStore(ABC):
    @abstractmethod
    async def add(self, episodes: list[dict]) -> None:
        """Persist a list of formatted atomic episode objects."""
        ...

    @abstractmethod
    async def query(
        self, user_id: str, session_id: str, query_text: str, top_k: int = 3
    ) -> list[dict]:
        """Retrieve relevant and recent episodic memories."""
        ...


class ChromaEpisodicStore(EpisodicStore):
    def __init__(self, persist_path: str = "./chroma_data"):
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name="episodic_memory", 
            metadata={"hnsw:space": "cosine"}
        )

    async def add(self, episodes: list[dict]) -> None:
        """Batch inserts atomic episode objects extracted from session history."""
        if not episodes:
            return

        ids = [ep["id"] for ep in episodes]
        documents = [ep["vector_text"] for ep in episodes]
        metadatas = [ep["metadata"] for ep in episodes]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    async def query(
        self,
        user_id: str,
        session_id: str,
        query_text: str,
        top_k: int = 3,
        recency_decay_rate: float = 0.05,
    ) -> list[dict]:
        """Queries episodic memories scoped by user_id and session_id,

        re-ranking candidates using an exponential recency decay function.
        """
        candidate_k = max(top_k * 3, 10)

        results = self._collection.query(
            query_texts=[query_text],
            n_results=candidate_k,
            where={
                "$and": [
                    {"user_id": user_id},
                    {"session_id": session_id},
                ]
            },
        )

        raw_docs = results.get("documents") or [[]]
        raw_metas = results.get("metadatas") or [[]]
        raw_dists = results.get("distances") or [[]]

        documents = raw_docs[0] if raw_docs and raw_docs[0] is not None else []
        metadatas = raw_metas[0] if raw_metas and raw_metas[0] is not None else []
        distances = raw_dists[0] if raw_dists and raw_dists[0] is not None else []

        now_epoch = int(time.time())
        scored_episodes = []

        for doc, meta, dist in zip(documents, metadatas, distances):
            # 1. Convert cosine distance to vector similarity score [0, 1]
            similarity = 1.0 - dist

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
        scored_episodes.sort(key=lambda x: x["score"], reverse=True)
        return scored_episodes[:top_k]