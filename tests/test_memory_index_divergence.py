"""A broken Chroma index (HNSW/metadata divergence) must degrade memory, never break it.

chroma-core/chroma#7032: if a row exists in Chroma's SQLite metadata but not in the HNSW vector
segment, every filtered `query()` raises `InternalError: Error executing plan: Internal error:
Error finding id`. Recall and saving both go through a filtered `query()`, so one orphaned row used
to break saving outright.
"""
import asyncio

import pytest

from dostuff.lib.memory.chroma import is_index_divergence
from dostuff.lib.memory.safe_query import safe_query

# The exact message Chroma 1.5.x raises (no typed exception, no error code)
DIVERGENCE = "Error executing plan: Internal error: Error finding id"


def _stub_embedder(texts):
    """Two-dim stand-in for Chroma's embedding function, so the fallback can rank without the index.

    Assigned per instance as `collection._embedding_function`: a class method would be subject to
    name mangling and would not be found by the fallback. Matching is case-insensitive because
    queries arrive in lower case while stored documents are capitalised.
    """
    return [
        [1.0, 0.0] if "editor" in t.lower() or "neovim" in t.lower() else [0.0, 1.0]
        for t in texts
    ]


class BrokenCollection:
    """Mimics a diverged collection: query() walks the graph and fails, get() does not."""

    def __init__(self, rows):
        self.rows = rows
        self.queries = 0
        self._embedding_function = _stub_embedder

    def count(self):
        return len(self.rows)

    def query(self, query_texts, n_results, where=None):
        self.queries += 1
        raise RuntimeError(DIVERGENCE)

    def get(self, ids=None, where=None, include=None, limit=None):
        rows = self.rows if limit is None else self.rows[:limit]
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [r[2] for r in rows],
        }


def test_divergence_is_detected_by_its_message_only():
    assert is_index_divergence(RuntimeError(DIVERGENCE))
    assert is_index_divergence(RuntimeError("Internal error: Error finding id abc-123"))
    # Real failures must not be swallowed as if they were the divergence bug
    assert not is_index_divergence(RuntimeError("collection not found"))
    assert not is_index_divergence(KeyError("ids"))


def test_safe_query_falls_back_to_metadata_and_ranks_in_python():
    rows = [
        ("u:lang", "User's language is Rust", {"key": "lang"}),
        ("u:editor", "User's editor is Neovim", {"key": "editor"}),
        ("u:orphan", None, {"key": "orphan"}),  # the row missing from the HNSW graph
    ]
    collection = BrokenCollection(rows)

    results = safe_query(collection, "which editor", top_k=2, where={"user_id": "u"})

    assert collection.queries == 1  # it tried the index first
    assert results["ids"][0] == ["u:editor", "u:lang"]  # ranked by similarity, orphan dropped
    assert all(meta["key"] != "orphan" for meta in results["metadatas"][0])
    # distances are cosine distances, as query() returns them
    assert results["distances"][0][0] < results["distances"][0][1]


def test_safe_query_reraises_errors_that_are_not_the_divergence_bug():
    class OtherFailure(BrokenCollection):
        def query(self, query_texts, n_results, where=None):
            raise RuntimeError("disk is on fire")

    # A row must exist, or safe_query returns {} without querying at all
    broken = OtherFailure([("u:editor", "User's editor is Neovim", {"key": "editor"})])
    with pytest.raises(RuntimeError, match="disk is on fire"):
        safe_query(broken, "which editor", top_k=1, where={"user_id": "u"})


def test_safe_query_on_empty_collection_does_not_query():
    collection = BrokenCollection([])
    assert safe_query(collection, "anything", top_k=3, where={"user_id": "u"}) == {}
    assert collection.queries == 0


def test_semantic_store_query_survives_divergence():
    from dostuff.lib.memory.semantic_memory_store import SemanticMemoryStore

    store = SemanticMemoryStore.__new__(SemanticMemoryStore)
    store._collection = BrokenCollection([
        ("u:editor", "User's editor is Neovim", {"key": "editor", "value": "Neovim", "source": "user_message"}),
        ("u:lang", "User's language is Rust", {"key": "lang", "value": "Rust", "source": "user_message"}),
        ("u:orphan", None, {"key": "orphan"}),
    ])
    facts = asyncio.run(store.query("u", "which editor do I use", top_k=3, min_similarity=0.0))
    assert [f["key"] for f in facts] == ["editor", "lang"]


def test_episode_lookup_survives_divergence():
    from dostuff.lib.memory.episodic_memory_store import EpisodicMemoryStore

    meta = {
        "event_type": "DECISION", "anchor_event": "Use WAL", "summary": "Avoid locked DB",
        "created_at_iso": "2026-10-01T00:00:00Z", "created_at_epoch": 1759276800,
    }
    store = EpisodicMemoryStore.__new__(EpisodicMemoryStore)
    store._collection = BrokenCollection([("ep_1", "Editor decision: use WAL", meta)])
    assert asyncio.run(store.query("u", "editor", top_k=3))[0]["anchor_event"] == "Use WAL"
    # find_duplicate is the first step of save_episode: it must not raise, or the save never happens
    assert asyncio.run(store.find_duplicate("u", "editor", "/proj", min_similarity=0.0)) == "ep_1"


def test_fallback_drops_rows_with_no_metadata():
    """`get()` returns metadatas=None for orphan rows, and every caller assumes .get() works."""

    class NullMetaCollection(BrokenCollection):
        def get(self, ids=None, where=None, include=None, limit=None):
            return {
                "ids": ["u:orphan", "u:editor"],
                "documents": ["ghost", "User's editor is Neovim"],
                "metadatas": [None, {"key": "editor", "value": "Neovim", "source": "user_message"}],
            }

    results = safe_query(NullMetaCollection([("u:orphan", "ghost", None)]), "which editor", top_k=5,
                         where={"user_id": "u"})
    assert results["metadatas"] == [[{"key": "editor", "value": "Neovim", "source": "user_message"}]]

    # The episode path calls meta.get(...) on every row, which used to raise on the None
    from dostuff.lib.memory.episodic_memory_store import EpisodicMemoryStore

    store = EpisodicMemoryStore.__new__(EpisodicMemoryStore)
    store._collection = NullMetaCollection([("u:orphan", "ghost", None)])
    store._collection.rows = [("ep_1", "Editor decision", {
        "event_type": "DECISION", "anchor_event": "Use WAL", "summary": "s",
        "created_at_iso": "2026-10-01T00:00:00Z", "created_at_epoch": 1759276800,
    })]
    episodes = asyncio.run(store.query("u", "editor", top_k=3))
    # One episode came back (the None row dropped) instead of raising AttributeError
    assert len(episodes) == 1


def test_fallback_is_cached_so_a_recall_turn_does_not_re_embed():
    """A broken index stays broken until repaired, so the fallback must not be paid every turn."""
    import dostuff.lib.memory.safe_query as safe_query_module

    safe_query_module.clear_fallback_cache()
    calls = {"embed": 0}

    class Counting(BrokenCollection):
        def __init__(self, rows):
            super().__init__(rows)

            def counting_embedder(texts):
                calls["embed"] += 1
                return _stub_embedder(texts)

            self._embedding_function = counting_embedder

    rows = [("u:editor", "User's editor is Neovim", {"key": "editor"})]
    collection = Counting(rows)

    first = safe_query(collection, "which editor", top_k=1, where={"user_id": "u"})
    second = safe_query(collection, "which editor", top_k=1, where={"user_id": "u"})

    assert first == second
    assert calls["embed"] == 2  # one embedder call for query + docs, not four
    safe_query_module.clear_fallback_cache()


def test_the_probe_catches_an_orphan_that_is_not_the_first_row():
    """`get(include=["embeddings"], limit=1)` misses a late orphan, which reads as healthy.

    That is how a diverged mcp_tools_index once passed `dostuff memory check`.
    """
    pytest.importorskip("chromadb")
    import sqlite3
    import tempfile
    from pathlib import Path

    import chromadb

    from dostuff.helpers.memory.repair_index import check_collection

    tmp = Path(tempfile.mkdtemp(prefix="late_orphan_"))
    client = chromadb.PersistentClient(path=str(tmp))
    collection = client.create_collection("mcp_tools_index", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[f"srv:t{i}" for i in range(20)],
        documents=[f"tool {i}" for i in range(20)],
        metadatas=[{"server_name": "srv", "tool_name": f"t{i}"} for i in range(20)],
    )
    # Inject the orphan last, so a limit=1 probe never reaches it
    db = sqlite3.connect(str(tmp / "chroma.sqlite3"))
    try:
        segment = db.execute("select id from segments where scope='METADATA'").fetchone()[0]
        seq = db.execute("select coalesce(max(seq_id), 0) from embeddings").fetchone()[0]
        cursor = db.execute(
            "insert into embeddings (segment_id, embedding_id, seq_id, created_at) "
            "values (?, ?, ?, datetime('now'))",
            (segment, "srv:ghost", seq + 1),
        )
        for key, value in (("server_name", "srv"), ("tool_name", "ghost")):
            db.execute("insert into embedding_metadata (id, key, string_value) values (?, ?, ?)",
                       (cursor.lastrowid, key, value))
        db.commit()
    finally:
        db.close()

    del client, collection
    reopened = chromadb.PersistentClient(path=str(tmp)).get_collection("mcp_tools_index")
    # A limit=1 probe would pass here; the real check must not
    assert reopened.get(include=["embeddings"], limit=1) is not None
    error = check_collection(reopened)
    assert error is not None and "Error finding id" in error


def test_degraded_memory_is_announced_once_per_collection(monkeypatch):
    """A silent fallback looks like the agent forgetting things, so it must say so once."""
    import dostuff.lib.memory.safe_query as safe_query_module

    safe_query_module.clear_fallback_cache()
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr("dostuff.helpers.ui.emit.emit", lambda message, kind="system": seen.append((kind, message)))

    collection = BrokenCollection([("u:editor", "User's editor is Neovim", {"key": "editor"})])
    collection.name = "semantic_memory"

    safe_query(collection, "which editor", top_k=1, where={"user_id": "u"})
    safe_query(collection, "which editor", top_k=1, where={"user_id": "u"})
    safe_query(collection, "another question", top_k=1, where={"user_id": "u"})

    assert len(seen) == 1, "warned once, not on every query"
    kind, message = seen[0]
    assert kind == "error"
    assert "semantic_memory" in message and "dostuff memory repair" in message
    assert "Nothing was lost" in message
    safe_query_module.clear_fallback_cache()


def test_tool_search_survives_divergence():
    from dostuff.lib.mcp.mcp_tool_registry_store import MCPToolRegistryStore

    store = MCPToolRegistryStore.__new__(MCPToolRegistryStore)
    store._collection = BrokenCollection([(
        "gh:search", "Tool Name: search",
        {"server_name": "gh", "tool_name": "search", "description": "Search repos",
         "input_schema_json": "{}", "output_schema_json": None},
    )])
    tools = asyncio.run(store.search_tools("editor search", top_k=3))
    assert [t["name"] for t in tools] == ["search"]
