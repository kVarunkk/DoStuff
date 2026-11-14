"""Long-term memory searched two ways at once: the `memory.backend: hybrid` experiment.

SQLite is the store of record (listing, counting, pruning, deleting go there; FTS5 keyword search, see
sqlite_memory_store). A Chroma collection mirrors every write as an embedding index, so a prompt that shares no word
with a memory can still find it (the paraphrase gap the live eval showed on sqlite's automatic recall).

A search asks both and keeps the union: each side filters by its own floor for the kind of search asked (recall,
search, dedupe), and a row's rank is its best score relative to that floor (1.0 = exactly at a floor). The two scores
are on different scales, so neither is compared with the other directly. A failing embedding index never fails a
search or a save: the keyword side's answer is used alone.
"""
import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection
from typing import Any, TypeVar

from dostuff.helpers.memory.resolve_memory_operation import DEDUP_MIN_SIMILARITY, resolve_memory_operation
from dostuff.lib.memory.episodic_memory_store import (
    DUPLICATE_EPISODE_SIMILARITY,
    EPISODE_CAP_PER_PROJECT,
    MIN_EPISODE_SIMILARITY,
    SIMILAR_EPISODE_SIMILARITY,
    EpisodicMemoryStore,
    rank_episodes,
)
from dostuff.lib.memory.semantic_memory_store import MIN_FACT_SIMILARITY, FactScope, SemanticMemoryStore
from dostuff.lib.memory.sqlite_memory_store import SQLiteEpisodeStore, SQLiteFactStore

log = logging.getLogger(__name__)
T = TypeVar("T")

# Search floor of the embedding store (search_memory's SEARCH_MIN_SIMILARITY; not imported: the tool module is heavier)
_EMBED_SEARCH_FLOOR = 0.2


async def _embedded(call: Callable[[], Awaitable[T]], fallback: T) -> T:
    """The embedding index's answer, or `fallback` if it fails (the keyword side still answers)."""
    try:
        return await call()
    except Exception as e:  # noqa: BLE001 - an index error must not cost the user a recall or a save
        log.warning("memory embedding index: %s", e)
        return fallback


def _floors(kinds: dict[float, tuple[float, float]], asked: float, recall: tuple[float, float]) -> tuple[float, float]:
    """(keyword floor, embedding floor) for the floor a caller asked for: one of this store's named floors, else the
    same fraction of each side's recall floor."""
    if asked in kinds:
        return kinds[asked]
    if asked <= 0:
        return 0.0, 0.0
    return recall[0] * asked, recall[1] * asked


def _merge(sides: list[tuple[list[dict], float, str]], key: Callable[[dict], Any]) -> list[tuple[float, dict]]:
    """Union by id of each side's (rows, floor, score field); a row ranks by its best score / floor."""
    best: dict[str, tuple[float, dict]] = {}
    for rows, floor, field in sides:
        for row in rows:
            rel = float(row[field]) / floor if floor > 0 else float(row[field])
            k = key(row)
            if k not in best or rel > best[k][0]:
                best[k] = (rel, row)
    return sorted(best.values(), key=lambda b: -b[0])


def _fact_key(fact: dict) -> tuple:
    return fact.get("scope") or "user", fact.get("working_dir") or "", fact["key"]


class HybridFactStore:
    """Facts (`SemanticMemoryStore`'s API): SQLite of record plus an embedding index."""

    # Named floors, one per kind of search; callers read them by name (getattr), so each maps to both sides' floor
    RECALL_FLOOR = 0.3
    SEARCH_FLOOR = 0.2
    DEDUPE_FLOOR = 0.1

    def __init__(self, db_path: str, index_path: str):
        self.keyword = SQLiteFactStore(db_path)
        self.embedding = SemanticMemoryStore(persist_path=index_path)
        self.db_path = db_path
        kw = SQLiteFactStore
        self._kinds = {self.RECALL_FLOOR: (kw.RECALL_FLOOR, MIN_FACT_SIMILARITY),
                       self.SEARCH_FLOOR: (kw.SEARCH_FLOOR, _EMBED_SEARCH_FLOOR),
                       self.DEDUPE_FLOOR: (kw.DEDUPE_FLOOR, DEDUP_MIN_SIMILARITY)}

    async def add(self, user_id: str, facts: list[dict]) -> None:
        for fact in facts or []:
            await resolve_memory_operation(user_id, fact, self)

    async def upsert_fact(self, user_id: str, fact: dict, scope: FactScope = "user", working_dir: str | None = None) -> None:
        await self.keyword.upsert_fact(user_id, fact, scope=scope, working_dir=working_dir)
        await _embedded(lambda: self.embedding.upsert_fact(user_id, fact, scope=scope, working_dir=working_dir), None)

    async def query(self, user_id: str, query_text: str, top_k: int = 3, min_similarity: float = 0.0,
                    scope: FactScope | None = None, working_dir: str | None = None) -> list[dict]:
        kw_floor, em_floor = _floors(self._kinds, min_similarity, self._kinds[self.RECALL_FLOOR])
        no_rows: list[dict] = []
        kw, em = await asyncio.gather(
            self.keyword.query(user_id, query_text, top_k=top_k, min_similarity=kw_floor, scope=scope,
                               working_dir=working_dir),
            _embedded(lambda: self.embedding.query(user_id, query_text, top_k=top_k, min_similarity=em_floor,
                                                   scope=scope, working_dir=working_dir), no_rows))
        merged = _merge([(kw, kw_floor, "similarity"), (em, em_floor, "similarity")], _fact_key)
        return [row for _, row in merged][:top_k]

    async def list_all(self, user_id: str, scope: FactScope | None = None, working_dir: str | None = None) -> list[dict]:
        return await self.keyword.list_all(user_id, scope=scope, working_dir=working_dir)

    async def delete(self, user_id: str, key: str, scope: FactScope = "user", working_dir: str | None = None) -> bool:
        deleted = await self.keyword.delete(user_id, key, scope=scope, working_dir=working_dir)
        await _embedded(lambda: self.embedding.delete(user_id, key, scope=scope, working_dir=working_dir), False)
        return deleted


class HybridEpisodeStore:
    """Episodes (`EpisodicMemoryStore`'s API): SQLite of record plus an embedding index."""

    RECALL_FLOOR = 0.3
    SEARCH_FLOOR = 0.2
    # save_episode compares nearest()'s similarity with these; nearest() reports on this scale (see there)
    SIMILAR = SIMILAR_EPISODE_SIMILARITY
    DUPLICATE = DUPLICATE_EPISODE_SIMILARITY

    def __init__(self, db_path: str, index_path: str):
        self.keyword = SQLiteEpisodeStore(db_path)
        self.embedding = EpisodicMemoryStore(persist_path=index_path)
        self.db_path = db_path
        kw = SQLiteEpisodeStore
        self._kinds = {self.RECALL_FLOOR: (kw.RECALL_FLOOR, MIN_EPISODE_SIMILARITY),
                       self.SEARCH_FLOOR: (kw.SEARCH_FLOOR, _EMBED_SEARCH_FLOOR)}

    async def add(self, episodes: list[dict]) -> None:
        await self.keyword.add(episodes)
        await _embedded(lambda: self.embedding.add(episodes), None)

    async def query(self, user_id: str, query_text: str, top_k: int = 3, working_dir: str | None = None,
                    recency_decay_rate: float | None = None, min_similarity: float | None = None, *,
                    exclude_ids: Collection[str] = (), stats: dict[str, Any] | None = None,
                    event_type: str | None = None) -> list[dict]:
        asked = self.RECALL_FLOOR if min_similarity is None else min_similarity
        kw_floor, em_floor = _floors(self._kinds, asked, self._kinds[self.RECALL_FLOOR])
        common: dict[str, Any] = {"top_k": top_k, "working_dir": working_dir, "recency_decay_rate": recency_decay_rate,
                                  "exclude_ids": exclude_ids, "event_type": event_type}
        kw_stats: dict[str, Any] = {}
        em_stats: dict[str, Any] = {}
        no_rows: list[dict] = []
        kw, em = await asyncio.gather(
            self.keyword.query(user_id, query_text, min_similarity=kw_floor, stats=kw_stats, **common),
            _embedded(lambda: self.embedding.query(user_id, query_text, min_similarity=em_floor, stats=em_stats,
                                                   **common), no_rows))
        merged = _merge([(kw, kw_floor, "score"), (em, em_floor, "score")], lambda e: e["id"])
        if stats is not None:
            # Matches past the top_k on each side are counted, not fetched: the larger side's count is a lower bound
            counts = [str(s.get("matching", "0")) for s in (kw_stats, em_stats)]
            more = any(c.endswith("+") for c in counts)
            stats["matching"] = f"{max(len(merged), *(int(c.rstrip('+')) for c in counts))}{'+' if more else ''}"
        # rank_episodes breaks near-ties by recency on `score`: give it the relative score both sides share
        return rank_episodes([{**row, "score": rel} for rel, row in merged])[:top_k]

    async def nearest(self, user_id: str, vector_text: str, working_dir: str,
                      min_similarity: float | None = None) -> dict | None:
        """The closest episode by either side, its similarity put on the embedding store's scale: each side's own
        duplicate band (SIMILAR..DUPLICATE) is mapped onto 0.6..0.85, so save_episode's thresholds mean the same."""
        kw_side = (SQLiteEpisodeStore.SIMILAR, SQLiteEpisodeStore.DUPLICATE)
        em_side = (SIMILAR_EPISODE_SIMILARITY, DUPLICATE_EPISODE_SIMILARITY)
        kw, em = await asyncio.gather(
            self.keyword.nearest(user_id, vector_text, working_dir, min_similarity=kw_side[0]),
            _embedded(lambda: self.embedding.nearest(user_id, vector_text, working_dir, min_similarity=em_side[0]), None))
        best: dict | None = None
        for found, (lo, hi) in ((kw, kw_side), (em, em_side)):
            if not found:
                continue
            scaled = self.SIMILAR + (float(found["similarity"]) - lo) * (self.DUPLICATE - self.SIMILAR) / (hi - lo)
            if best is None or scaled > best["similarity"]:
                best = {**found, "similarity": min(1.0, scaled)}
        floor = self.SIMILAR if min_similarity is None else min_similarity
        return best if best and best["similarity"] >= floor else None

    async def prune(self, user_id: str, working_dir: str, cap: int = EPISODE_CAP_PER_PROJECT) -> list[str]:
        doomed = await self.keyword.prune(user_id, working_dir, cap=cap)
        async def drop(episode_id: str) -> bool:
            return await self.embedding.delete(user_id, episode_id)

        for episode_id in doomed:
            await _embedded(lambda: drop(episode_id), False)
        return doomed

    async def count(self, user_id: str, working_dir: str | None = None) -> int:
        return await self.keyword.count(user_id, working_dir=working_dir)

    async def list_all(self, user_id: str, working_dir: str | None = None) -> list[dict]:
        return await self.keyword.list_all(user_id, working_dir=working_dir)

    async def delete(self, user_id: str, episode_id: str) -> bool:
        deleted = await self.keyword.delete(user_id, episode_id)
        await _embedded(lambda: self.embedding.delete(user_id, episode_id), False)
        return deleted
