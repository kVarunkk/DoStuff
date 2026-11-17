"""The repair for a diverged Chroma index must keep every readable row and drop only the orphans.

This builds a real PersistentClient, injects a metadata row with no HNSW vector (which is exactly
what makes Chroma raise "Error finding id"), and checks that the rebuild brings the collection back.
Skipped when chromadb is not installed.
"""
import pytest

pytest.importorskip("chromadb")

import sqlite3  # noqa: E402


def _inject_orphan(persist_path, collection_name: str, orphan_id: str) -> None:
    """Add a metadata row for `orphan_id` while leaving it out of the HNSW vector segment."""
    db = sqlite3.connect(str(persist_path / "chroma.sqlite3"))
    try:
        metadata_segment = db.execute(
            "select s.id from segments s join collections c on s.collection = c.id "
            "where c.name = ? and s.scope = 'METADATA'",
            (collection_name,),
        ).fetchone()[0]
        next_seq = db.execute("select coalesce(max(seq_id), 0) from embeddings").fetchone()[0]
        cursor = db.execute(
            "insert into embeddings (segment_id, embedding_id, seq_id, created_at) "
            "values (?, ?, ?, datetime('now'))",
            (metadata_segment, orphan_id, next_seq + 1),
        )
        row_id = cursor.lastrowid
        for key, value in (("user_id", "u1"), ("key", "orphan"), ("value", "ghost")):
            db.execute(
                "insert into embedding_metadata (id, key, string_value) values (?, ?, ?)",
                (row_id, key, value),
            )
        db.commit()
    finally:
        db.close()


@pytest.fixture
def broken_store(tmp_path, monkeypatch):
    """A collection holding two good facts and one orphan, with a fresh client cache."""
    import chromadb

    from dostuff.lib.memory import chroma

    monkeypatch.setattr(chroma, "_clients", {})
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("semantic_memory", metadata={"hnsw:space": "cosine"})
    collection.upsert(
        ids=["u1:editor", "u1:lang"],
        documents=["User's editor is Neovim", "User's language is Rust"],
        metadatas=[{"user_id": "u1", "key": "editor"}, {"user_id": "u1", "key": "lang"}],
    )
    _inject_orphan(tmp_path, "semantic_memory", "u1:orphan")

    # Reopen so the client sees the injected row, as a later session would
    del client, collection
    monkeypatch.setattr(chroma, "_clients", {})
    fresh = chromadb.PersistentClient(path=str(tmp_path))
    return fresh, fresh.get_collection("semantic_memory")


def test_check_collection_reports_the_divergence(broken_store):
    from dostuff.helpers.memory.repair_index import check_collection

    _, collection = broken_store
    error = check_collection(collection, {"user_id": "u1"})
    assert error is not None and "Error finding id" in error


def test_rebuild_collection_restores_filtered_queries_and_drops_only_orphans(broken_store, tmp_path):
    from dostuff.helpers.memory.repair_index import check_collection, dump_rows, rebuild_collection

    client, collection = broken_store
    ids, documents, metadatas, skipped = dump_rows(collection)
    assert ids == ["u1:editor", "u1:lang"]  # the orphan has no document, so it is not dumpable
    assert skipped == {"orphans": 1, "missing_metadata": 0}
    assert all(isinstance(doc, str) for doc in documents)

    summary = rebuild_collection(client, "semantic_memory", backup_dir=tmp_path / "backups")

    assert summary["kept"] == 2 and summary["orphans"] == 1 and summary["error"] is None
    assert summary["before"] == 3
    rebuilt = client.get_collection("semantic_memory")
    assert rebuilt.count() == 2
    hits = rebuilt.query(query_texts=["editor"], n_results=2, where={"user_id": "u1"})
    assert hits["ids"][0] == ["u1:editor", "u1:lang"]
    assert check_collection(rebuilt, {"user_id": "u1"}) is None
    # The staging collection must not be left behind
    assert [c.name for c in client.list_collections()] == ["semantic_memory"]


def test_rebuild_survives_a_row_with_no_metadata(broken_store, tmp_path):
    """A row added without metadata is legal in Chroma but cannot be re-added next to metadated rows.

    Before this was handled, `add()` raised "Expected metadata to be a non-empty dict" *after* the
    collection had been deleted, destroying the whole collection.
    """
    from dostuff.helpers.memory.repair_index import rebuild_collection

    client, collection = broken_store
    collection.add(ids=["u1:nometa"], documents=["row saved without any metadata"])

    summary = rebuild_collection(client, "semantic_memory", backup_dir=tmp_path / "backups")

    assert summary["error"] is None
    assert summary["missing_metadata"] == 1
    # Not lost: the good row is back and the unreadable one is in the backup
    assert client.get_collection("semantic_memory").count() == summary["kept"] == 2
    backup = (tmp_path / "backups" / "semantic_memory.json").read_text(encoding="utf-8")
    assert "u1:nometa" in backup


def test_failed_staging_leaves_the_original_collection_intact(broken_store, monkeypatch):
    """The delete only happens once the replacement is built and verified."""
    import dostuff.helpers.memory.repair_index as repair_index

    client, collection = broken_store
    before = collection.count()
    monkeypatch.setattr(repair_index, "stage_collection", lambda *a, **k: 0)  # simulate a partial re-add

    with pytest.raises(RuntimeError, match="staged 0 of"):
        repair_index.rebuild_collection(client, "semantic_memory")

    assert client.get_collection("semantic_memory").count() == before
    assert [c.name for c in client.list_collections()] == ["semantic_memory"]


def test_check_collection_catches_a_break_only_the_filtered_query_sees(broken_store):
    """An orphan carrying `server_name` breaks `search_tools` while an unfiltered query still works.

    This is why `check` cannot use `where=None` for mcp_tools_index: it would report it healthy.
    """
    from dostuff.helpers.memory.repair_index import check_collection

    client, collection = broken_store
    collection.add(ids=["u1:tool"], documents=["Tool Name: search"],
                   metadatas=[{"server_name": "srv", "tool_name": "search"}])
    error = check_collection(collection)
    assert error is not None and "Error finding id" in error


def test_memory_store_recall_works_again_after_repair(broken_store):
    import asyncio

    from dostuff.helpers.memory.repair_index import rebuild_collection
    from dostuff.lib.memory.semantic_memory_store import SemanticMemoryStore

    client, collection = broken_store
    store = SemanticMemoryStore.__new__(SemanticMemoryStore)
    store._collection = collection
    # While broken, recall degrades instead of raising
    assert [f["key"] for f in asyncio.run(store.query("u1", "editor", top_k=2, min_similarity=0.0))] == [
        "editor", "lang"
    ]

    rebuild_collection(client, "semantic_memory")
    store._collection = client.get_collection("semantic_memory")
    assert [f["key"] for f in asyncio.run(store.query("u1", "editor", top_k=2, min_similarity=0.0))] == [
        "editor", "lang"
    ]
