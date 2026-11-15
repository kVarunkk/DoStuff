"""Hybrid as the default backend: memories of the old chroma store brought into memory.db once, and the embedding
index checked and repaired against memory.db (`dostuff memory check|repair`, and the TUI at startup)."""
import asyncio
import json
import sqlite3
import time

import pytest

pytest.importorskip("chromadb")

from typer.testing import CliRunner  # noqa: E402

from dostuff.lib.memory import migrate  # noqa: E402
from dostuff.lib.memory.episodic_memory_store import EpisodicMemoryStore  # noqa: E402
from dostuff.lib.memory.hybrid_memory_store import HybridEpisodeStore, HybridFactStore  # noqa: E402
from dostuff.lib.memory.semantic_memory_store import SemanticMemoryStore  # noqa: E402
from dostuff.lib.memory.sqlite_memory_store import SQLiteFactStore  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def _episode(i, anchor, wd="/p"):
    meta = {"user_id": "u", "session_id": "s", "working_dir": wd, "event_type": "DECISION", "anchor_event": anchor,
            "summary": f"summary {i}", "tags": json.dumps(["t"]), "created_at_epoch": 1_700_000_000 + i,
            "created_at_iso": "2023-11-14"}
    return {"id": f"e{i}", "vector_text": f"- EVENT TYPE: DECISION\n  - ANCHOR EVENT: {anchor}", "metadata": meta}


@pytest.fixture
def old_store(tmp_path):
    """A data dir as the chroma backend left it: two facts (one project) and two episodes."""
    chroma_path = str(tmp_path / "chroma")
    facts = SemanticMemoryStore(persist_path=chroma_path)
    run(facts.upsert_fact("u", {"key": "editor", "value": "Uses Neovim", "source": "user_message"}))
    run(facts.upsert_fact("u", {"key": "tests", "value": "Run pytest with -q"}, scope="project", working_dir="/p"))
    run(EpisodicMemoryStore(persist_path=chroma_path).add([_episode(1, "Chose SQLite"), _episode(2, "Dropped Redis")]))
    return tmp_path


def _hybrid(tmp_path):
    return (HybridFactStore(str(tmp_path / "memory.db"), str(tmp_path / "memory-index")),
            HybridEpisodeStore(str(tmp_path / "memory.db"), str(tmp_path / "memory-index")))


def _import(tmp_path, queue=True):
    from dostuff.lib.memory.sqlite_memory_store import _Db
    _Db(str(tmp_path / "memory.db"))
    return migrate.import_from_chroma(str(tmp_path / "memory.db"), str(tmp_path / "chroma"), queue_for_index=queue)


def test_old_memories_are_brought_in_with_their_ids_scopes_and_times(old_store):
    result = _import(old_store)
    assert (result.status, result.facts, result.episodes) == ("imported", 2, 2)
    assert "2 saved facts and 2 episodes" in result.message()
    facts, episodes = _hybrid(old_store)
    by_key = {f["key"]: f for f in run(facts.keyword.list_all("u"))}
    assert by_key["editor"]["value"] == "Uses Neovim" and by_key["editor"]["scope"] == "user"
    assert by_key["tests"]["scope"] == "project" and by_key["tests"]["working_dir"] == "/p"
    assert by_key["editor"]["created_at_epoch"] > 0
    eps = {e["id"]: e for e in run(episodes.keyword.list_all("u", working_dir="/p"))}
    assert set(eps) == {"e1", "e2"} and eps["e1"]["created_at_epoch"] == 1_700_000_001
    # Queued, so the first search indexes them: a paraphrase finds the imported fact through the embedding side
    found = run(facts.query("u", "which text editor do I use", top_k=3, min_similarity=facts.RECALL_FLOOR))
    assert "editor" in [f["key"] for f in found]


def test_import_runs_once_so_a_memory_forgotten_after_it_stays_forgotten(old_store):
    _import(old_store)
    facts, _ = _hybrid(old_store)
    assert run(facts.delete("u", "editor"))
    assert _import(old_store).status == "done"
    assert "editor" not in {f["key"] for f in run(facts.list_all("u"))}


def test_import_keeps_a_newer_value_already_in_memory_db_and_leaves_the_old_store(old_store):
    db = str(old_store / "memory.db")
    run(SQLiteFactStore(db).upsert_fact("u", {"key": "editor", "value": "Uses Helix"}))
    _import(old_store)
    assert {f["key"]: f["value"] for f in run(SQLiteFactStore(db).list_all("u"))}["editor"] == "Uses Helix"
    old = SemanticMemoryStore(persist_path=str(old_store / "chroma"))
    assert {f["key"] for f in run(old.list_all("u"))} == {"editor", "tests"}


def test_no_old_store_and_no_chromadb_are_told_apart(tmp_path, old_store, monkeypatch):
    empty = tmp_path / "fresh"
    empty.mkdir()
    assert _import(empty).status == "none" and _import(empty).message() == ""
    monkeypatch.setattr(migrate.chroma, "chroma_available", lambda: False)
    result = _import(old_store)
    assert result.status == "needs-chromadb" and 'pip install "dostuff[memory]"' in result.message()
    monkeypatch.undo()
    assert _import(old_store).status == "imported"  # not marked done while it couldn't read the store


def test_failed_import_is_retried_and_does_not_stop_startup(old_store, monkeypatch):
    def broken(_path):
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(migrate, "_read_chroma", broken)
    result = _import(old_store)
    assert result.status == "failed" and "not a database" in result.message()
    monkeypatch.undo()
    assert _import(old_store).status == "imported"


def test_sqlite_backend_import_queues_nothing_for_an_index(old_store):
    _import(old_store, queue=False)
    with sqlite3.connect(old_store / "memory.db") as conn:
        assert conn.execute("SELECT count(*) FROM index_outbox").fetchone()[0] == 0


def test_switching_from_sqlite_to_hybrid_indexes_the_rows_saved_before(tmp_path):
    db = str(tmp_path / "memory.db")
    run(SQLiteFactStore(db).upsert_fact("u", {"key": "py_errors", "value": "Raises ValueError for invalid input"}))
    facts, _ = _hybrid(tmp_path)
    report = run(facts.check_index())
    assert (report.rows, report.indexed, report.missing, report.ok) == (1, 0, 1, False)
    assert "1 not indexed" in report.describe()
    after = run(facts.repair_index())
    assert after.ok and after.indexed == 1
    found = run(facts.query("u", "what exception do I throw on illegal values?", top_k=3,
                            min_similarity=facts.RECALL_FLOOR))
    assert [f["key"] for f in found][:1] == ["py_errors"]


def test_check_finds_stale_and_forgotten_rows_and_repair_fixes_both(tmp_path):
    facts, episodes = _hybrid(tmp_path)
    run(facts.upsert_fact("u", {"key": "editor", "value": "Uses Neovim"}))
    run(facts.upsert_fact("u", {"key": "shell", "value": "Uses fish"}))
    run(episodes.add([_episode(1, "Chose SQLite")]))
    # Behind the index's back, as an older dostuff (or a crash before the outbox existed) would leave it
    with sqlite3.connect(tmp_path / "memory.db") as conn:
        conn.execute("UPDATE facts SET value = 'Uses Helix' WHERE key = 'editor'")
        conn.execute("DELETE FROM facts WHERE key = 'shell'")
        conn.execute("UPDATE episodes SET summary = 'changed' WHERE id = 'e1'")
    report = run(facts.check_index())
    assert (report.stale, report.extra) == (1, 1)
    assert run(episodes.check_index()).stale == 1
    assert run(facts.repair_index()).ok and run(episodes.repair_index()).ok
    assert {f["key"]: f["value"] for f in run(facts.embedding.list_all("u"))} == {"editor": "Uses Helix"}


def test_broken_index_is_rebuilt_from_memory_db(tmp_path, monkeypatch):
    from dostuff.helpers.memory import repair_index
    facts, _ = _hybrid(tmp_path)
    run(facts.upsert_fact("u", {"key": "editor", "value": "Uses Neovim"}))
    old_collection = facts.embedding._collection
    calls = []

    def check(collection, where=None):
        calls.append(collection)
        return "Error finding id 7" if collection is old_collection else None

    monkeypatch.setattr(repair_index, "check_collection", check)
    assert run(facts.check_index()).broken == "Error finding id 7"
    after = run(facts.repair_index())
    assert after.ok and facts.embedding._collection is not old_collection
    assert {f["key"] for f in run(facts.embedding.list_all("u"))} == {"editor"}


def test_repair_reports_an_index_it_cannot_write(tmp_path, monkeypatch):
    run(SQLiteFactStore(str(tmp_path / "memory.db")).upsert_fact("u", {"key": "editor", "value": "Uses Neovim"}))
    facts, _ = _hybrid(tmp_path)

    async def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(facts.embedding, "upsert_fact", boom)
    with pytest.raises(RuntimeError, match="could not be written"):
        run(facts.repair_index())


# ---- the default, the CLI and consolidate ----------------------------------------------------------------------------

@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """get_config() pointed at tmp_path, backend left at its default, store caches cleared."""
    from dostuff.config import get_config
    from dostuff.lib.memory import stores
    cfg = get_config()
    monkeypatch.setattr(type(cfg), "get_data_dir", lambda self, *a, **k: tmp_path)
    monkeypatch.setattr(cfg, "get_user_id", lambda *a, **k: "u")
    for cached in (stores._imported, stores._hybrid_facts, stores._hybrid_episodes, stores._sqlite_facts,
                   stores._sqlite_episodes):
        cached.cache_clear()
    yield tmp_path
    for cached in (stores._imported, stores._hybrid_facts, stores._hybrid_episodes, stores._sqlite_facts,
                   stores._sqlite_episodes):
        cached.cache_clear()


def test_hybrid_is_the_default_backend(tmp_path, monkeypatch):
    import yaml

    from dostuff.config import Config
    monkeypatch.chdir(tmp_path)  # no project config.yaml in the way
    cases = [({}, "hybrid"), ({"memory": {}}, "hybrid"), ({"memory": {"backend": "nonsense"}}, "hybrid"),
             ({"memory": {"backend": "chroma"}}, "chroma"), ({"memory": {"backend": "sqlite"}}, "sqlite")]
    for i, (raw, backend) in enumerate(cases):
        path = tmp_path / f"config{i}.yaml"
        path.write_text(yaml.safe_dump(raw))
        assert Config(path).memory_backend == backend, raw


def test_first_start_on_the_default_backend_brings_in_the_old_store(old_store, data_dir):
    from dostuff.lib.memory import stores
    store = stores.get_semantic_store()
    assert isinstance(store, HybridFactStore)
    assert {f["key"] for f in run(store.list_all("u"))} == {"editor", "tests"}
    assert stores.chroma_import_result().status == "imported"


def test_memory_check_and_repair_on_hybrid(data_dir):
    from dostuff.cli import app
    db = str(data_dir / "memory.db")
    run(SQLiteFactStore(db).upsert_fact("u", {"key": "editor", "value": "Uses Neovim"}))  # saved under sqlite
    runner = CliRunner()
    checked = runner.invoke(app, ["memory", "check"])
    assert checked.exit_code == 1, checked.output
    assert "keyword index: ok" in checked.output and "1 not indexed" in checked.output
    assert "dostuff memory repair" in checked.output
    repaired = runner.invoke(app, ["memory", "repair"])
    assert repaired.exit_code == 0, repaired.output
    assert "facts: ok (1 saved, all indexed)" in repaired.output
    assert runner.invoke(app, ["memory", "check"]).exit_code == 0


def test_memory_repair_rebuilds_a_broken_keyword_index(data_dir, monkeypatch):
    from dostuff.cli import app
    from dostuff.lib.memory import chroma
    monkeypatch.setattr(chroma, "chroma_available", lambda: False)  # hybrid = sqlite: keywords only
    db = str(data_dir / "memory.db")
    run(SQLiteFactStore(db).upsert_fact("u", {"key": "editor", "value": "Uses Neovim"}))
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO facts_fts(facts_fts, rowid, key, value) VALUES ('delete', 1, 'editor', 'Uses Neovim')")
    runner = CliRunner()
    checked = runner.invoke(app, ["memory", "check"])
    assert checked.exit_code == 1 and "keyword index: BROKEN" in checked.output, checked.output
    assert "embedding index: off" in checked.output
    repaired = runner.invoke(app, ["memory", "repair"])
    assert repaired.exit_code == 0 and "keyword index: rebuilt" in repaired.output, repaired.output
    found = run(SQLiteFactStore(db).query("u", "neovim", top_k=3, min_similarity=0.1))
    assert [f["key"] for f in found] == ["editor"]


def test_reembed_is_refused_off_the_chroma_backend(data_dir):
    from dostuff.cli import app
    result = CliRunner().invoke(app, ["memory", "repair", "--reembed"])
    assert result.exit_code == 2 and "chroma backend" in result.output


def test_consolidate_uses_the_configured_store(data_dir, monkeypatch):
    from dostuff.cli import _consolidate_facts
    from dostuff.helpers.memory import consolidate
    seen = []

    async def propose(store, user_id, here):
        seen.append(store)
        return []

    monkeypatch.setattr(consolidate, "propose", propose)
    _consolidate_facts(yes=True)
    assert isinstance(seen[0], HybridFactStore)


def test_import_marker_written_with_counts(old_store):
    _import(old_store)
    with sqlite3.connect(old_store / "memory.db") as conn:
        marker = json.loads(conn.execute("SELECT value FROM meta WHERE key = 'chroma_import'").fetchone()[0])
    assert (marker["facts"], marker["episodes"]) == (2, 2) and marker["at"] <= time.time()
