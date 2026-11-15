"""Long-term memory in SQLite with FTS5 keyword search: the `memory.backend: sqlite` alternative to Chroma.

Same API as SemanticMemoryStore / EpisodicMemoryStore, so every caller works with either. No embedding model and no
optional extra: Python's sqlite3 with FTS5 is enough. Facts and episodes live in `<data dir>/memory.db`.

Ranking: FTS5 finds the rows sharing a word with the query (porter-stemmed, so "tests" finds "test"), BM25 keeps the
best `_POOL`, and each is scored by the cosine of idf-weighted term sets: 1.0 for the same words, 0 for none in
common. Like the embedding store's cosine, the score is bounded, so floors and the duplicate check keep a meaning; its
numbers differ (a reworded sentence scores lower), so the store carries its own floors (`RECALL_FLOOR` etc.), set
from the offline recall set (tests/test_memory_recall_eval.py). What it can't do is match a paraphrase that shares
no word with the query.
"""
import asyncio
import json
import math
import re
import sqlite3
import time
from collections.abc import Collection, Iterable
from typing import Any

from dostuff.helpers.memory.resolve_memory_operation import resolve_memory_operation
from dostuff.lib.memory.episodic_memory_store import EPISODE_CAP_PER_PROJECT, decay_rate, rank_episodes
from dostuff.lib.memory.semantic_memory_store import FactScope, fact_id

# Candidates BM25 keeps before the cosine re-rank; a store holds hundreds of rows, so this is nearly all matches
_POOL = 200
_TOKENIZE = "porter unicode61"
_WORD = re.compile(r"[^\W_]+")
# Words that say nothing about which memory is meant. Their absence is what makes "hi" match nothing
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing don done down during each few for from further had has have having he her here hers him
his how i if in into is it its itself just let me more most my myself no nor not now of off on once only or other our
ours out over own same she should so some such than that the their theirs them then there these they this those
through to too under until up us very was we were what when where which while who whom why will with would you your
yours yourself hi hello hey thanks thank please ok okay yes yeah sure tell know get got
""".split())

# The words people use for each episode type, indexed with it: "what did we decide" stems to "decid", which the type
# name DECISION ("decis") doesn't match. The type's own vocabulary, nothing about any one project.
TYPE_WORDS = {
    "DECISION": "decision decided decide chose choice settled agreed",
    "PREFERENCE": "preference prefers prefer wants likes",
    "CORRECTION": "correction corrected mistake wrong fixed",
    "LEARNING": "learning learned lesson discovered found",
    "EVENT": "event happened occurred",
}
_TYPE_WORDS_SQL = "CASE upper({col}) " + " ".join(
    f"WHEN '{t}' THEN '{t.lower()} {w}'" for t, w in TYPE_WORDS.items()) + " ELSE {col} END"
_NEW_TYPE, _OLD_TYPE = _TYPE_WORDS_SQL.format(col="new.event_type"), _TYPE_WORDS_SQL.format(col="old.event_type")

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, scope TEXT NOT NULL, working_dir TEXT,
    key TEXT NOT NULL, value TEXT NOT NULL, source TEXT, created_at_epoch INTEGER, updated_at_epoch INTEGER);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(key, value, content='facts', tokenize='{_TOKENIZE}');
CREATE VIRTUAL TABLE IF NOT EXISTS facts_vocab USING fts5vocab(facts_fts, 'row');
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, key, value) VALUES (new.rowid, new.key, new.value); END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, key, value) VALUES ('delete', old.rowid, old.key, old.value); END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, key, value) VALUES ('delete', old.rowid, old.key, old.value);
    INSERT INTO facts_fts(rowid, key, value) VALUES (new.rowid, new.key, new.value); END;

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT, working_dir TEXT, event_type TEXT,
    anchor_event TEXT, summary TEXT, tags TEXT, created_at_epoch INTEGER, created_at_iso TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    event_type, anchor_event, summary, tags, content='episodes', tokenize='{_TOKENIZE}');
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_vocab USING fts5vocab(episodes_fts, 'row');
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, event_type, anchor_event, summary, tags)
    VALUES (new.rowid, {_NEW_TYPE}, new.anchor_event, new.summary, new.tags); END;
CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, event_type, anchor_event, summary, tags)
    VALUES ('delete', old.rowid, {_OLD_TYPE}, old.anchor_event, old.summary, old.tags); END;
CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, event_type, anchor_event, summary, tags)
    VALUES ('delete', old.rowid, {_OLD_TYPE}, old.anchor_event, old.summary, old.tags);
    INSERT INTO episodes_fts(rowid, event_type, anchor_event, summary, tags)
    VALUES (new.rowid, {_NEW_TYPE}, new.anchor_event, new.summary, new.tags); END;

CREATE TABLE IF NOT EXISTS index_outbox (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, row_id TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS index_outbox_row ON index_outbox(kind, row_id);
"""
# `index_outbox`: rows whose copy in a secondary index (the hybrid backend's embeddings) is out of date. A store opened
# with `outbox=True` adds an entry in the same transaction as the write, so a crash can't leave a change the index
# never hears about. An entry names the row only: whoever drains it reads the row's current state (gone = delete).


def _enqueue(conn: sqlite3.Connection, kind: str, row_ids: Iterable[str]) -> None:
    conn.executemany("INSERT INTO index_outbox (kind, row_id) VALUES (?, ?)", [(kind, i) for i in row_ids])


def fts5_available() -> bool:
    """True if this Python's sqlite3 was built with FTS5 (almost every build; the store needs it)."""
    try:
        sqlite3.connect(":memory:").execute(f"CREATE VIRTUAL TABLE t USING fts5(a, tokenize='{_TOKENIZE}')")
        return True
    except sqlite3.Error:
        return False


def content_words(text: str) -> list[str]:
    """The words of `text` that can tell memories apart: lowercased, split at punctuation and underscores
    (`py_all_exports` is three words), stopwords and single characters dropped."""
    return [w for w in _WORD.findall((text or "").lower()) if len(w) > 1 and w not in STOPWORDS]


class _Db:
    """One SQLite file, a connection per operation (operations run in worker threads)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    async def outbox_pending(self, kind: str, limit: int) -> list[tuple[str, int]]:
        """(row_id, newest seq) of up to `limit` rows with outbox entries, oldest first."""
        def _pending() -> list[tuple[str, int]]:
            with self.connect() as conn:
                return [(str(r[0]), int(r[1])) for r in conn.execute(
                    "SELECT row_id, max(seq) FROM index_outbox WHERE kind = ? GROUP BY row_id ORDER BY min(seq) LIMIT ?",
                    (kind, limit))]

        return await asyncio.to_thread(_pending)

    async def outbox_done(self, kind: str, done: list[tuple[str, int]]) -> None:
        """Drop the entries a drain applied: up to the seq it read, so a write made meanwhile is still pending."""
        def _done() -> None:
            with self.connect() as conn:
                conn.executemany("DELETE FROM index_outbox WHERE kind = ? AND row_id = ? AND seq <= ?",
                                 [(kind, row_id, seq) for row_id, seq in done])

        await asyncio.to_thread(_done)

    async def rows_by_id(self, table: str, row_ids: Collection[str]) -> dict[str, sqlite3.Row]:
        """The current rows of `table` ("facts" or "episodes") among `row_ids`; a missing id is not in the dict."""
        ids = list(dict.fromkeys(row_ids))

        def _rows() -> dict[str, sqlite3.Row]:
            out: dict[str, sqlite3.Row] = {}
            with self.connect() as conn:
                for i in range(0, len(ids), 500):
                    chunk = ids[i:i + 500]
                    marks = ",".join("?" * len(chunk))
                    out.update({r["id"]: r for r in conn.execute(f"SELECT * FROM {table} WHERE id IN ({marks})", chunk)})
            return out

        return await asyncio.to_thread(_rows) if ids else {}


def _stems(conn: sqlite3.Connection, texts: list[str]) -> list[set[str]]:
    """Each text's content words as the FTS index stores them (porter stems), through a scratch FTS5 table, so query
    and documents are stemmed exactly alike."""
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS temp.qtok USING fts5(t, tokenize='{_TOKENIZE}')")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.qtok_v USING fts5vocab(qtok, 'instance')")
    conn.execute("DELETE FROM temp.qtok")
    conn.executemany("INSERT INTO temp.qtok(rowid, t) VALUES (?, ?)",
                     [(i + 1, " ".join(content_words(t))) for i, t in enumerate(texts)])
    out: list[set[str]] = [set() for _ in texts]
    for doc, term in conn.execute("SELECT doc, term FROM temp.qtok_v"):
        out[int(doc) - 1].add(str(term))
    conn.execute("DELETE FROM temp.qtok")
    return out


def _idf(conn: sqlite3.Connection, vocab: str, base: str, terms: Iterable[str]) -> dict[str, float]:
    terms = sorted(set(terms))
    total = conn.execute(f"SELECT count(*) FROM {base}").fetchone()[0]
    df: dict[str, int] = {}
    for i in range(0, len(terms), 500):
        chunk = terms[i:i + 500]
        marks = ",".join("?" * len(chunk))
        df.update({str(t): int(d) for t, d in conn.execute(
            f"SELECT term, doc FROM {vocab} WHERE term IN ({marks})", chunk)})
    return {t: math.log(1.0 + (total + 1) / (df.get(t, 0) + 0.5)) for t in terms}


def _cosine(query: set[str], doc: set[str], idf: dict[str, float]) -> float:
    shared = sum(idf[t] ** 2 for t in query & doc)
    norm = math.sqrt(sum(idf[t] ** 2 for t in query) * sum(idf[t] ** 2 for t in doc))
    return shared / norm if norm else 0.0


def _ranked(conn: sqlite3.Connection, table: str, text_of: Any, query_text: str, where: str, params: list,
            pool: int = _POOL) -> tuple[list[tuple[float, sqlite3.Row]], bool]:
    """(score, row) for the rows of `table` matching `where` that share a word with `query_text`, best first, and
    whether the BM25 pool was full (more rows may match)."""
    words = content_words(query_text)
    if not words:
        return [], False
    match = " OR ".join(f'"{w}"' for w in dict.fromkeys(words))
    rows = conn.execute(
        f"SELECT b.* FROM {table}_fts f JOIN {table} b ON b.rowid = f.rowid "
        f"WHERE {table}_fts MATCH ? AND {where} ORDER BY bm25({table}_fts) LIMIT ?",
        [match, *params, pool]).fetchall()
    if not rows:
        return [], False
    query, *docs = _stems(conn, [query_text, *(text_of(r) for r in rows)])
    idf = _idf(conn, f"{table}_vocab", table, query.union(*docs))
    scored = sorted(((_cosine(query, d, idf), r) for d, r in zip(docs, rows)), key=lambda s: -s[0])
    return scored, len(rows) >= pool


def _others(conn: sqlite3.Connection, table: str, where: str, params: list, skip: set[str], limit: int,
            order: str) -> list[sqlite3.Row]:
    """Rows sharing no word with the query, for a caller that asked for no floor (an embedding store returns its
    nearest rows however far they are)."""
    rows = conn.execute(f"SELECT * FROM {table} WHERE {where} ORDER BY {order} LIMIT ?",
                        [*params, limit + len(skip)]).fetchall()
    return [r for r in rows if r["id"] not in skip][:limit]


# ---- facts -----------------------------------------------------------------------------------------------------------

def _fact_where(user_id: str, scope: FactScope | None, working_dir: str | None) -> tuple[str, list]:
    if scope == "user":
        return "b.user_id = ? AND b.scope != 'project'", [user_id]
    if scope == "project":
        if not working_dir:
            raise ValueError("A project fact needs the project's working_dir.")
        return "b.user_id = ? AND b.scope = 'project' AND b.working_dir = ?", [user_id, working_dir]
    if working_dir is not None:
        return "b.user_id = ? AND (b.scope != 'project' OR b.working_dir = ?)", [user_id, working_dir]
    return "b.user_id = ?", [user_id]


def fact_fields(row: sqlite3.Row) -> dict:
    fact = {"key": row["key"], "value": row["value"], "source": row["source"] or "unknown", "scope": row["scope"],
            "created_at_epoch": int(row["created_at_epoch"] or 0), "updated_at_epoch": int(row["updated_at_epoch"] or 0)}
    if row["scope"] == "project":
        fact["working_dir"] = row["working_dir"] or ""
    return fact


class SQLiteFactStore:
    """Saved facts (`SemanticMemoryStore`'s API) in SQLite, searched with FTS5."""

    # Floors for this store's score (idf-weighted term cosine), from the offline recall set's sweep (2026-11-14): at
    # 0.2 the top 5 hold 19/34 wanted facts (the embedding store 15/34 at its 0.25); an off-topic prompt shares no word
    RECALL_FLOOR = 0.2
    SEARCH_FLOOR = 0.1
    DEDUPE_FLOOR = 0.1

    OUTBOX_KIND = "fact"

    def __init__(self, db_path: str, outbox: bool = False):
        self._db = _Db(db_path)
        self.db_path = db_path
        self._outbox = outbox

    async def add(self, user_id: str, facts: list[dict]) -> None:
        for fact in facts or []:
            await resolve_memory_operation(user_id, fact, self)

    async def upsert_fact(self, user_id: str, fact: dict, scope: FactScope = "user", working_dir: str | None = None) -> None:
        key, value = str(fact.get("key", "")).strip(), str(fact.get("value", "")).strip()
        if not key or not value:
            return
        row_id = fact_id(user_id, key, scope, working_dir)
        source = str(fact.get("source") or "user_message")

        def _write() -> None:
            now = int(time.time())
            with self._db.connect() as conn:
                conn.execute(
                    "INSERT INTO facts (id, user_id, scope, working_dir, key, value, source, created_at_epoch, "
                    "updated_at_epoch) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                    "value = excluded.value, source = excluded.source, updated_at_epoch = excluded.updated_at_epoch",
                    (row_id, user_id, scope, working_dir if scope == "project" else None, key, value, source, now, now))
                if self._outbox:
                    _enqueue(conn, self.OUTBOX_KIND, [row_id])

        await asyncio.to_thread(_write)

    async def query(self, user_id: str, query_text: str, top_k: int = 3, min_similarity: float = 0.0,
                    scope: FactScope | None = None, working_dir: str | None = None) -> list[dict]:
        where, params = _fact_where(user_id, scope, working_dir)

        def _query() -> list[dict]:
            with self._db.connect() as conn:
                scored, _ = _ranked(conn, "facts", lambda r: f"{r['key']} {r['value']}", query_text, where, params,
                                    pool=max(_POOL, top_k))
                found = [{**fact_fields(r), "similarity": s} for s, r in scored if s >= min_similarity][:top_k]
                if min_similarity <= 0 and len(found) < top_k:
                    seen = {r["id"] for _, r in scored}
                    found += [{**fact_fields(r), "similarity": 0.0} for r in _others(conn, "facts b", where, params, seen, top_k - len(found),
                                                        "b.updated_at_epoch DESC")]
                return found

        return await asyncio.to_thread(_query)

    async def list_all(self, user_id: str, scope: FactScope | None = None, working_dir: str | None = None) -> list[dict]:
        where, params = _fact_where(user_id, scope, working_dir)

        def _list() -> list[dict]:
            with self._db.connect() as conn:
                return [fact_fields(r) for r in conn.execute(f"SELECT * FROM facts b WHERE {where}", params)]

        facts = await asyncio.to_thread(_list)
        return sorted(facts, key=lambda f: (-f["updated_at_epoch"], f["scope"] != "project", f["key"]))

    async def delete(self, user_id: str, key: str, scope: FactScope = "user", working_dir: str | None = None) -> bool:
        row_id = fact_id(user_id, key, scope, working_dir)

        def _delete() -> bool:
            with self._db.connect() as conn:
                if self._outbox:  # even when the row is already gone: its index copy may not be
                    _enqueue(conn, self.OUTBOX_KIND, [row_id])
                return conn.execute("DELETE FROM facts WHERE id = ?", (row_id,)).rowcount > 0

        return await asyncio.to_thread(_delete)


# ---- episodes --------------------------------------------------------------------------------------------------------

def _episode_text(row: sqlite3.Row) -> str:
    """What the score compares a query with: the same text the FTS row indexes (type words included)."""
    kind = str(row["event_type"] or "").upper()
    words = f"{kind.lower()} {TYPE_WORDS[kind]}" if kind in TYPE_WORDS else kind
    return f"{words} {row['anchor_event']} {row['summary']} {row['tags']}"


def _epoch(row: sqlite3.Row) -> int:
    raw = row["created_at_epoch"]
    return int(raw) if isinstance(raw, (int, float)) else 0


def episode_fields(row: sqlite3.Row) -> dict:
    """An episode row as a query returns it, without its score."""
    raw_tags = row["tags"]
    return {"id": row["id"], "event_type": row["event_type"], "anchor_event": row["anchor_event"],
            "summary": row["summary"], "tags": json.loads(raw_tags) if isinstance(raw_tags, str) and raw_tags else [],
            "created_at_iso": row["created_at_iso"], "created_at_epoch": _epoch(row), "session_id": row["session_id"]}


class SQLiteEpisodeStore:
    """Episodes (`EpisodicMemoryStore`'s API) in SQLite, searched with FTS5."""

    # Sweep (2026-11-14): 0.15 recalls 17/21 episodes with 0/36 false hits (the embedding store: 17/21 with 2 false
    # hits at its 0.3); 0.2 recalls 16/21
    RECALL_FLOOR = 0.15
    SEARCH_FLOOR = 0.1
    # Rewordings of one moment share fewer words than an embedding sees, so the duplicate band starts lower; a
    # near-verbatim copy still scores above DUPLICATE
    SIMILAR = 0.3
    DUPLICATE = 0.85

    OUTBOX_KIND = "episode"

    def __init__(self, db_path: str, outbox: bool = False):
        self._db = _Db(db_path)
        self.db_path = db_path
        self._outbox = outbox

    async def add(self, episodes: list[dict]) -> None:
        if not episodes:
            return

        def _write() -> None:
            with self._db.connect() as conn:
                for ep in episodes:
                    m = ep["metadata"]
                    conn.execute(
                        "INSERT INTO episodes (id, user_id, session_id, working_dir, event_type, anchor_event, summary, "
                        "tags, created_at_epoch, created_at_iso) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(id) DO UPDATE SET event_type = excluded.event_type, "
                        "anchor_event = excluded.anchor_event, summary = excluded.summary, tags = excluded.tags",
                        (ep["id"], m.get("user_id"), m.get("session_id"), m.get("working_dir"), m.get("event_type"),
                         m.get("anchor_event"), m.get("summary"), m.get("tags") or "[]", m.get("created_at_epoch"),
                         m.get("created_at_iso")))
                if self._outbox:
                    _enqueue(conn, self.OUTBOX_KIND, [ep["id"] for ep in episodes])

        await asyncio.to_thread(_write)

    @staticmethod
    def _where(user_id: str, working_dir: str | None, event_type: str | None = None) -> tuple[str, list]:
        where, params = "b.user_id = ?", [user_id]
        if working_dir:
            where, params = where + " AND b.working_dir = ?", params + [working_dir]
        if event_type:
            where, params = where + " AND b.event_type = ?", params + [event_type]
        return where, params

    async def query(self, user_id: str, query_text: str, top_k: int = 3, working_dir: str | None = None,
                    recency_decay_rate: float | None = None, min_similarity: float | None = None, *,
                    exclude_ids: Collection[str] = (), stats: dict[str, Any] | None = None,
                    event_type: str | None = None) -> list[dict]:
        floor = self.RECALL_FLOOR if min_similarity is None else min_similarity
        excluded = set(exclude_ids)
        where, params = self._where(user_id, working_dir, event_type)
        now = int(time.time())

        def scored(similarity: float, row: sqlite3.Row) -> dict:
            rate = decay_rate(row["event_type"]) if recency_decay_rate is None else recency_decay_rate
            return {"score": similarity * math.exp(-rate * max(0.0, (now - _epoch(row)) / 86400.0)),
                    **episode_fields(row)}

        def _query() -> list[dict]:
            with self._db.connect() as conn:
                ranked, pool_full = _ranked(conn, "episodes", _episode_text, query_text, where, params)
                found = [scored(s, r) for s, r in ranked if s >= floor and r["id"] not in excluded]
                if stats is not None:
                    stats["matching"] = f"{len(found)}{'+' if pool_full else ''}"
                if floor <= 0 and len(found) < top_k:
                    seen = {r["id"] for _, r in ranked} | excluded
                    found += [scored(0.0, r) for r in _others(conn, "episodes b", where, params, seen,
                                                              top_k - len(found), "b.created_at_epoch DESC")]
                return found

        return rank_episodes(await asyncio.to_thread(_query))[:top_k]

    async def nearest(self, user_id: str, vector_text: str, working_dir: str,
                      min_similarity: float | None = None) -> dict | None:
        floor = self.SIMILAR if min_similarity is None else min_similarity
        where, params = self._where(user_id, working_dir)

        def _nearest() -> dict | None:
            with self._db.connect() as conn:
                ranked, _ = _ranked(conn, "episodes", _episode_text, vector_text, where, params)
            if not ranked or ranked[0][0] < floor:
                return None
            similarity, row = ranked[0]
            return {"id": row["id"], "similarity": similarity, "event_type": row["event_type"] or "",
                    "anchor_event": row["anchor_event"] or "", "summary": row["summary"] or ""}

        return await asyncio.to_thread(_nearest)

    async def prune(self, user_id: str, working_dir: str, cap: int = EPISODE_CAP_PER_PROJECT) -> list[str]:
        where, params = self._where(user_id, working_dir)

        def _prune() -> list[str]:
            with self._db.connect() as conn:
                rows = conn.execute(f"SELECT * FROM episodes b WHERE {where}", params).fetchall()
                excess = len(rows) - cap
                if excess <= 0:
                    return []
                now = int(time.time())
                candidates = sorted(
                    (math.exp(-decay_rate(r["event_type"]) * max(0.0, (now - _epoch(r)) / 86400.0)), _epoch(r), r["id"])
                    for r in rows if str(r["event_type"] or "").upper() != "DECISION")
                doomed = [episode_id for _, _, episode_id in candidates[:excess]]
                conn.executemany("DELETE FROM episodes WHERE id = ?", [(i,) for i in doomed])
                if self._outbox:
                    _enqueue(conn, self.OUTBOX_KIND, doomed)
                return doomed

        return await asyncio.to_thread(_prune)

    async def count(self, user_id: str, working_dir: str | None = None) -> int:
        where, params = self._where(user_id, working_dir)

        def _count() -> int:
            with self._db.connect() as conn:
                return int(conn.execute(f"SELECT count(*) FROM episodes b WHERE {where}", params).fetchone()[0])

        return await asyncio.to_thread(_count)

    async def list_all(self, user_id: str, working_dir: str | None = None) -> list[dict]:
        where, params = self._where(user_id, working_dir)

        def _list() -> list[dict]:
            with self._db.connect() as conn:
                return [{"id": r["id"], "event_type": r["event_type"] or "", "anchor_event": r["anchor_event"] or "",
                         "created_at_iso": r["created_at_iso"] or "", "created_at_epoch": _epoch(r)}
                        for r in conn.execute(f"SELECT * FROM episodes b WHERE {where}", params)]

        return sorted(await asyncio.to_thread(_list), key=lambda e: e["created_at_epoch"], reverse=True)

    async def delete(self, user_id: str, episode_id: str) -> bool:
        def _delete() -> bool:
            with self._db.connect() as conn:
                deleted = conn.execute("DELETE FROM episodes WHERE id = ? AND user_id = ?",
                                       (episode_id, user_id)).rowcount > 0
                # Queued when the row is gone now, even if it was already (its index copy may not be); never for
                # another user's episode, which stays
                if self._outbox and (deleted or not conn.execute(
                        "SELECT 1 FROM episodes WHERE id = ?", (episode_id,)).fetchone()):
                    _enqueue(conn, self.OUTBOX_KIND, [episode_id])
                return deleted

        return await asyncio.to_thread(_delete)
