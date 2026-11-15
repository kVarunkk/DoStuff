"""Long-term memory searched two ways at once: the `memory.backend: hybrid` experiment.

SQLite is the store of record (listing, counting, pruning, deleting go there; FTS5 keyword search, see
sqlite_memory_store). A Chroma collection mirrors every write as an embedding index, so a prompt that shares no word
with a memory can still find it (the paraphrase gap the live eval showed on sqlite's automatic recall).

A search asks both and keeps the union: each side filters by its own floor for the kind of search asked (recall,
search, dedupe), and a row's rank is its best score relative to that floor (1.0 = exactly at a floor). The two scores
are on different scales, so neither is compared with the other directly. A failing embedding index never fails a
search or a save: the keyword side's answer is used alone.

The two stores can't share a transaction, so the index is kept as a cache SQLite can always rebuild:
- every write to SQLite adds an `index_outbox` entry in the same transaction; `_Outbox.drain` then copies each named
  row's current state to the index (gone = delete) and drops the entry only once the index took it. A crash or an index
  error leaves the entry for the next drain (after each write, and before a search once `_RETRY_AFTER` has passed).
- a search keeps an index hit only if its row is still in SQLite, with SQLite's values, so an index that is behind
  never brings back a forgotten memory or an old value.
"""
import asyncio
import logging
import sqlite3
import time
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
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
from dostuff.lib.memory.semantic_memory_store import MIN_FACT_SIMILARITY, FactScope, SemanticMemoryStore, fact_id
from dostuff.lib.memory.sqlite_memory_store import SQLiteEpisodeStore, SQLiteFactStore, episode_fields, fact_fields

log = logging.getLogger(__name__)
T = TypeVar("T")

# Search floor of the embedding store (search_memory's SEARCH_MIN_SIMILARITY; not imported: the tool module is heavier)
_EMBED_SEARCH_FLOOR = 0.2
# After a failed drain, searches don't retry it for this long (a write still does): a broken index would otherwise
# cost every turn a failing call
_RETRY_AFTER = 60.0
_DRAIN_BATCH = 50


async def _embedded(call: Callable[[], Awaitable[T]], fallback: T) -> T:
    """The embedding index's answer, or `fallback` if it fails (the keyword side still answers)."""
    try:
        return await call()
    except Exception as e:  # noqa: BLE001 - an index error must not cost the user a recall or a save
        log.warning("memory embedding index: %s", e)
        return fallback


class _Outbox:
    """Copies the rows `kind`'s outbox entries name from SQLite to the embedding index (module docstring)."""

    def __init__(self, keyword: SQLiteFactStore | SQLiteEpisodeStore, table: str,
                 apply: Callable[[list[sqlite3.Row], list[str]], Awaitable[None]]):
        self._db = keyword._db
        self._kind = keyword.OUTBOX_KIND
        self._table = table
        self._apply = apply  # (rows to write, ids to delete)
        self._failed_at: float | None = None

    async def drain(self, *, retry: bool = True) -> bool:
        """Apply every pending entry; False if the index failed (the rest stay queued). `retry=False` skips the
        attempt within `_RETRY_AFTER` of a failure."""
        if not retry and self._failed_at is not None and time.monotonic() - self._failed_at < _RETRY_AFTER:
            return False
        try:
            while pending := await self._db.outbox_pending(self._kind, _DRAIN_BATCH):
                rows = await self._db.rows_by_id(self._table, [row_id for row_id, _ in pending])
                await self._apply([rows[i] for i, _ in pending if i in rows], [i for i, _ in pending if i not in rows])
                await self._db.outbox_done(self._kind, pending)
        except Exception as e:  # noqa: BLE001 - the write is in SQLite already; the index catches up later
            log.warning("memory embedding index: %s (pending changes kept for the next try)", e)
            self._failed_at = time.monotonic()
            return False
        self._failed_at = None
        return True


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


@dataclass
class IndexReport:
    """How one embedding index compares with memory.db (`dostuff memory check`)."""
    kind: str  # "facts" or "episodes"
    rows: int  # in memory.db
    indexed: int  # in the index
    missing: int = 0  # in memory.db, not in the index (e.g. saved under the sqlite backend, or imported)
    stale: int = 0  # in both, but the index holds an older text
    extra: int = 0  # in the index only (forgotten in memory.db)
    pending: int = 0  # queued writes not yet in the index
    broken: str | None = None  # the index can't answer queries (Chroma "Error finding id")

    @property
    def ok(self) -> bool:
        return not (self.missing or self.stale or self.extra or self.pending or self.broken)

    def describe(self) -> str:
        if self.broken:
            return f"{self.kind}: index BROKEN ({self.broken})"
        if self.ok:
            return f"{self.kind}: ok ({self.rows} saved, all indexed)"
        parts = [f"{n} {label}" for n, label in ((self.missing, "not indexed"), (self.stale, "out of date"),
                                                   (self.extra, "forgotten but still indexed"),
                                                   (self.pending, "queued")) if n]
        return f"{self.kind}: {self.rows} saved, {', '.join(parts)}"


class IndexSync:
    """Check and repair of one hybrid store's embedding index against memory.db, which is the data: the index can
    always be rebuilt from it. Subclasses name their table and what of a row the index must hold."""

    keyword: SQLiteFactStore | SQLiteEpisodeStore
    embedding: SemanticMemoryStore | EpisodicMemoryStore
    outbox: _Outbox
    _table: str
    _collection_name: str

    @staticmethod
    def _row_state(row: sqlite3.Row) -> tuple:
        raise NotImplementedError

    @staticmethod
    def _index_state(meta: dict) -> tuple:
        raise NotImplementedError

    def _diff(self, rows: dict[str, sqlite3.Row]) -> tuple[list[str], list[str], list[str], int]:
        """(missing, stale, extra) ids and how many rows the index holds; reads metadata only, so a broken vector
        graph doesn't stop it."""
        found = self.embedding._collection.get(include=["metadatas"])
        index = {str(i): (m or {}) for i, m in zip(found.get("ids") or [], found.get("metadatas") or [])}
        missing = [i for i in rows if i not in index]
        stale = [i for i in rows if i in index and self._row_state(rows[i]) != self._index_state(index[i])]
        extra = [i for i in index if i not in rows]
        return missing, stale, extra, len(index)

    async def check_index(self) -> IndexReport:
        from dostuff.helpers.memory.repair_index import check_collection
        rows = await self.keyword._db.all_rows(self._table)
        missing, stale, extra, indexed = await asyncio.to_thread(self._diff, rows)
        return IndexReport(self._table, len(rows), indexed, len(missing), len(stale), len(extra),
                           await self.keyword._db.outbox_count(self.keyword.OUTBOX_KIND),
                           await asyncio.to_thread(check_collection, self.embedding._collection))

    async def repair_index(self) -> IndexReport:
        """Bring the index in line with memory.db: a broken one is dropped and rebuilt, otherwise only the rows that
        differ are queued and written. Returns the state after. Raises if the index can't be written."""
        from dostuff.helpers.memory.repair_index import check_collection
        if await asyncio.to_thread(check_collection, self.embedding._collection):
            await asyncio.to_thread(self._recreate_collection)
        rows = await self.keyword._db.all_rows(self._table)
        missing, stale, extra, _ = await asyncio.to_thread(self._diff, rows)
        await self.keyword._db.enqueue(self.keyword.OUTBOX_KIND, [*missing, *stale, *extra])
        if not await self.outbox.drain():
            raise RuntimeError(f"the {self._table} embedding index could not be written (see the log)")
        return await self.check_index()

    def _recreate_collection(self) -> None:
        client = self.embedding._client
        client.delete_collection(self._collection_name)
        self.embedding._collection = client.get_or_create_collection(name=self._collection_name,
                                                                     metadata={"hnsw:space": "cosine"})


class HybridFactStore(IndexSync):
    """Facts (`SemanticMemoryStore`'s API): SQLite of record plus an embedding index."""

    keyword: SQLiteFactStore
    embedding: SemanticMemoryStore
    _table = "facts"
    _collection_name = "semantic_memory"

    @staticmethod
    def _row_state(row: sqlite3.Row) -> tuple:
        return (row["value"],)

    @staticmethod
    def _index_state(meta: dict) -> tuple:
        return (meta.get("value"),)

    # Named floors, one per kind of search; callers read them by name (getattr), so each maps to both sides' floor
    RECALL_FLOOR = 0.3
    SEARCH_FLOOR = 0.2
    DEDUPE_FLOOR = 0.1

    def __init__(self, db_path: str, index_path: str):
        self.keyword = SQLiteFactStore(db_path, outbox=True)
        self.embedding = SemanticMemoryStore(persist_path=index_path)
        self.db_path = db_path
        self.outbox = _Outbox(self.keyword, "facts", self._apply)
        kw = SQLiteFactStore
        self._kinds = {self.RECALL_FLOOR: (kw.RECALL_FLOOR, MIN_FACT_SIMILARITY),
                       self.SEARCH_FLOOR: (kw.SEARCH_FLOOR, _EMBED_SEARCH_FLOOR),
                       self.DEDUPE_FLOOR: (kw.DEDUPE_FLOOR, DEDUP_MIN_SIMILARITY)}

    async def _apply(self, rows: list[sqlite3.Row], gone: list[str]) -> None:
        for row in rows:
            await self.embedding.upsert_fact(row["user_id"], fact_fields(row), scope=row["scope"],
                                             working_dir=row["working_dir"])
        await self.embedding.delete_ids(gone)

    async def add(self, user_id: str, facts: list[dict]) -> None:
        for fact in facts or []:
            await resolve_memory_operation(user_id, fact, self)

    async def upsert_fact(self, user_id: str, fact: dict, scope: FactScope = "user", working_dir: str | None = None) -> None:
        await self.keyword.upsert_fact(user_id, fact, scope=scope, working_dir=working_dir)
        await self.outbox.drain()

    async def query(self, user_id: str, query_text: str, top_k: int = 3, min_similarity: float = 0.0,
                    scope: FactScope | None = None, working_dir: str | None = None) -> list[dict]:
        await self.outbox.drain(retry=False)
        kw_floor, em_floor = _floors(self._kinds, min_similarity, self._kinds[self.RECALL_FLOOR])
        no_rows: list[dict] = []
        kw, em = await asyncio.gather(
            self.keyword.query(user_id, query_text, top_k=top_k, min_similarity=kw_floor, scope=scope,
                               working_dir=working_dir),
            _embedded(lambda: self.embedding.query(user_id, query_text, top_k=top_k, min_similarity=em_floor,
                                                   scope=scope, working_dir=working_dir), no_rows))

        def row_id(fact: dict) -> str:
            if fact.get("scope") != "project":
                return fact_id(user_id, fact["key"], "user")
            if not fact.get("working_dir"):
                return ""  # a malformed index row: matches nothing in SQLite, so it is dropped
            return fact_id(user_id, fact["key"], "project", fact["working_dir"])

        merged = _merge([(kw, kw_floor, "similarity"), (em, em_floor, "similarity")], row_id)
        # SQLite decides what exists and what it says: an index hit for a deleted or since-changed fact is dropped
        # or given the current value
        current = await self.keyword._db.rows_by_id("facts", [row_id(row) for _, row in merged])
        found = [{**fact_fields(current[i]), "similarity": row["similarity"]}
                 for _, row in merged if (i := row_id(row)) in current]
        return found[:top_k]

    async def list_all(self, user_id: str, scope: FactScope | None = None, working_dir: str | None = None) -> list[dict]:
        return await self.keyword.list_all(user_id, scope=scope, working_dir=working_dir)

    async def delete(self, user_id: str, key: str, scope: FactScope = "user", working_dir: str | None = None) -> bool:
        deleted = await self.keyword.delete(user_id, key, scope=scope, working_dir=working_dir)
        await self.outbox.drain()
        return deleted


def _episode_doc(row: sqlite3.Row) -> dict:
    """An episode row as the embedding index stores it (save_episode's vector text; no None in the metadata)."""
    vector_text = (f"- EVENT TYPE: {row['event_type']}\n  - ANCHOR EVENT: {row['anchor_event']}\n"
                   f"  - SUMMARY: {row['summary']}")
    meta = {k: row[k] for k in ("user_id", "session_id", "working_dir", "event_type", "anchor_event", "summary",
                                "tags", "created_at_epoch", "created_at_iso") if row[k] is not None}
    return {"id": row["id"], "vector_text": vector_text, "metadata": meta}


_EPISODE_TEXT = ("event_type", "anchor_event", "summary", "tags")


class HybridEpisodeStore(IndexSync):
    """Episodes (`EpisodicMemoryStore`'s API): SQLite of record plus an embedding index."""

    keyword: SQLiteEpisodeStore
    embedding: EpisodicMemoryStore
    _table = "episodes"
    _collection_name = "episodic_memory"

    @staticmethod
    def _row_state(row: sqlite3.Row) -> tuple:
        return tuple(row[k] for k in _EPISODE_TEXT)

    @staticmethod
    def _index_state(meta: dict) -> tuple:
        return tuple(meta.get(k) for k in _EPISODE_TEXT)

    RECALL_FLOOR = 0.3
    SEARCH_FLOOR = 0.2
    # save_episode compares nearest()'s similarity with these; nearest() reports on this scale (see there)
    SIMILAR = SIMILAR_EPISODE_SIMILARITY
    DUPLICATE = DUPLICATE_EPISODE_SIMILARITY

    def __init__(self, db_path: str, index_path: str):
        self.keyword = SQLiteEpisodeStore(db_path, outbox=True)
        self.embedding = EpisodicMemoryStore(persist_path=index_path)
        self.db_path = db_path
        self.outbox = _Outbox(self.keyword, "episodes", self._apply)
        kw = SQLiteEpisodeStore
        self._kinds = {self.RECALL_FLOOR: (kw.RECALL_FLOOR, MIN_EPISODE_SIMILARITY),
                       self.SEARCH_FLOOR: (kw.SEARCH_FLOOR, _EMBED_SEARCH_FLOOR)}

    async def _apply(self, rows: list[sqlite3.Row], gone: list[str]) -> None:
        await self.embedding.add([_episode_doc(r) for r in rows])
        await self.embedding.delete_ids(gone)

    async def add(self, episodes: list[dict]) -> None:
        await self.keyword.add(episodes)
        await self.outbox.drain()

    async def query(self, user_id: str, query_text: str, top_k: int = 3, working_dir: str | None = None,
                    recency_decay_rate: float | None = None, min_similarity: float | None = None, *,
                    exclude_ids: Collection[str] = (), stats: dict[str, Any] | None = None,
                    event_type: str | None = None) -> list[dict]:
        await self.outbox.drain(retry=False)
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
        # SQLite decides what exists and what it says (see HybridFactStore.query)
        current = await self.keyword._db.rows_by_id("episodes", [row["id"] for _, row in merged])
        merged = [(rel, episode_fields(current[row["id"]])) for rel, row in merged if row["id"] in current]
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
        duplicate band (SIMILAR..DUPLICATE) is mapped onto 0.6..0.85, so save_episode's thresholds mean the same.
        Only an episode still in SQLite counts: a deleted one in the index must not make save_episode skip a save."""
        await self.outbox.drain(retry=False)
        kw_side = (SQLiteEpisodeStore.SIMILAR, SQLiteEpisodeStore.DUPLICATE)
        em_side = (SIMILAR_EPISODE_SIMILARITY, DUPLICATE_EPISODE_SIMILARITY)
        kw, em = await asyncio.gather(
            self.keyword.nearest(user_id, vector_text, working_dir, min_similarity=kw_side[0]),
            _embedded(lambda: self.embedding.nearest(user_id, vector_text, working_dir, min_similarity=em_side[0]), None))
        current = await self.keyword._db.rows_by_id("episodes", [f["id"] for f in (kw, em) if f])
        best: dict | None = None
        for found, (lo, hi) in ((kw, kw_side), (em, em_side)):
            if not found or found["id"] not in current:
                continue
            scaled = self.SIMILAR + (float(found["similarity"]) - lo) * (self.DUPLICATE - self.SIMILAR) / (hi - lo)
            if best is None or scaled > best["similarity"]:
                row = current[found["id"]]
                best = {"id": row["id"], "similarity": min(1.0, scaled), "event_type": row["event_type"] or "",
                        "anchor_event": row["anchor_event"] or "", "summary": row["summary"] or ""}
        floor = self.SIMILAR if min_similarity is None else min_similarity
        return best if best and best["similarity"] >= floor else None

    async def prune(self, user_id: str, working_dir: str, cap: int = EPISODE_CAP_PER_PROJECT) -> list[str]:
        doomed = await self.keyword.prune(user_id, working_dir, cap=cap)
        if doomed:
            await self.outbox.drain()
        return doomed

    async def count(self, user_id: str, working_dir: str | None = None) -> int:
        return await self.keyword.count(user_id, working_dir=working_dir)

    async def list_all(self, user_id: str, working_dir: str | None = None) -> list[dict]:
        return await self.keyword.list_all(user_id, working_dir=working_dir)

    async def delete(self, user_id: str, episode_id: str) -> bool:
        deleted = await self.keyword.delete(user_id, episode_id)
        await self.outbox.drain()
        return deleted
