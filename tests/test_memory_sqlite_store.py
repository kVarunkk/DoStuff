"""The sqlite memory backend (memory.backend: sqlite): FTS5 keyword search behind the Chroma stores' API."""
import asyncio
import json
import sqlite3
import time

import pytest

from dostuff.lib.memory.sqlite_memory_store import (
    SQLiteEpisodeStore, SQLiteFactStore, content_words, fts5_available,
)

pytestmark = pytest.mark.skipif(not fts5_available(), reason="sqlite3 built without FTS5")


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def facts(tmp_path):
    return SQLiteFactStore(str(tmp_path / "memory.db"))


@pytest.fixture
def episodes(tmp_path):
    return SQLiteEpisodeStore(str(tmp_path / "memory.db"))


def _ep(i, text, event_type="LEARNING", wd="/p", days_old=0, summary=None):
    now = int(time.time()) - days_old * 86400
    return {"id": f"e{i}", "metadata": {"user_id": "u", "session_id": "s", "working_dir": wd, "event_type": event_type,
                                         "anchor_event": text, "summary": summary or text, "tags": json.dumps([]),
                                         "created_at_epoch": now, "created_at_iso": ""}}


def test_content_words_split_keys_and_drop_stopwords():
    assert content_words("Which py_all_exports do I use?") == ["py", "all", "exports", "use"] or \
        content_words("Which py_all_exports do I use?") == ["py", "exports", "use"]
    assert content_words("hi, thanks!") == []


def test_fact_upsert_keeps_created_and_updates_value(facts, monkeypatch):
    clock = iter([100, 200])
    monkeypatch.setattr(time, "time", lambda: next(clock))
    run(facts.upsert_fact("u", {"key": "editor", "value": "Uses Vim"}))
    run(facts.upsert_fact("u", {"key": "editor", "value": "Uses Neovim"}))
    [fact] = run(facts.list_all("u"))
    assert (fact["value"], fact["created_at_epoch"], fact["updated_at_epoch"]) == ("Uses Neovim", 100, 200)


def test_fact_scopes_list_and_delete(facts):
    run(facts.upsert_fact("u", {"key": "linter", "value": "Uses ruff"}))
    run(facts.upsert_fact("u", {"key": "linter", "value": "flake8 only"}, scope="project", working_dir="/a"))
    run(facts.upsert_fact("u", {"key": "db", "value": "Postgres 16"}, scope="project", working_dir="/b"))
    assert {f["value"] for f in run(facts.list_all("u", working_dir="/a"))} == {"Uses ruff", "flake8 only"}
    assert [f["key"] for f in run(facts.list_all("u", scope="project", working_dir="/b"))] == ["db"]
    assert run(facts.delete("u", "linter", scope="project", working_dir="/a"))
    assert not run(facts.delete("u", "linter", scope="project", working_dir="/a"))
    assert [f["value"] for f in run(facts.list_all("u", working_dir="/a"))] == ["Uses ruff"]


def test_fact_query_ranks_by_shared_words_and_respects_floor(facts):
    run(facts.upsert_fact("u", {"key": "keyboard_layout", "value": "Types on a Colemak keyboard layout"}))
    run(facts.upsert_fact("u", {"key": "coffee", "value": "Drinks oat-milk flat whites"}))
    run(facts.upsert_fact("u", {"key": "tests", "value": "Runs tests with pytest"}))
    got = run(facts.query("u", "Which keyboard layout do I type on?", top_k=3, min_similarity=0.1))
    assert [f["key"] for f in got] == ["keyboard_layout"]  # "type" finds "Types": stemmed
    assert run(facts.query("u", "hi", top_k=3, min_similarity=0.1)) == []
    # No floor: the nearest rows are returned however far, like the embedding store
    assert len(run(facts.query("u", "hi", top_k=3, min_similarity=0.0))) == 3


def test_fact_query_never_crosses_projects(facts):
    run(facts.upsert_fact("u", {"key": "db", "value": "Staging uses PostgreSQL"}, scope="project", working_dir="/b"))
    assert run(facts.query("u", "PostgreSQL", top_k=5, working_dir="/a")) == []
    assert [f["key"] for f in run(facts.query("u", "PostgreSQL", top_k=5, working_dir="/b"))] == ["db"]


def test_episode_query_floor_type_filter_exclude_and_stats(episodes):
    run(episodes.add([_ep(1, "Enabled WAL journal mode for the SQLite store", "DECISION"),
                      _ep(2, "'database is locked' fixed with a busy timeout on sqlite connect"),
                      _ep(3, "Bumped requests to 2.32", "EVENT"),
                      _ep(4, "SQLite WAL in another project", wd="/other")]))
    stats: dict = {}
    got = run(episodes.query("u", "why WAL mode on the sqlite store?", top_k=3, working_dir="/p", stats=stats))
    assert got[0]["id"] == "e1" and "e4" not in {e["id"] for e in got} and stats["matching"] in ("1", "2")
    assert run(episodes.query("u", "hello there", top_k=3, working_dir="/p")) == []
    typed = run(episodes.query("u", "sqlite", top_k=3, working_dir="/p", event_type="LEARNING"))
    assert [e["id"] for e in typed] == ["e2"]
    assert "e1" not in {e["id"] for e in run(episodes.query("u", "WAL sqlite", top_k=3, working_dir="/p",
                                                             exclude_ids={"e1"}))}


def test_episode_typed_search_without_floor_lists_every_episode_of_the_type(episodes):
    run(episodes.add([_ep(1, "Chose MIT licence", "DECISION"), _ep(2, "Dropped Python 3.8", "DECISION")]))
    got = run(episodes.query("u", "what did we decide about tests?", top_k=5, working_dir="/p", min_similarity=-1.0,
                             recency_decay_rate=0.0, event_type="DECISION"))
    assert {e["id"] for e in got} == {"e1", "e2"}


def test_episode_old_decision_outranks_old_event(episodes):
    run(episodes.add([_ep(1, "sqlite WAL decision", "DECISION", days_old=200),
                      _ep(2, "sqlite WAL event", "EVENT", days_old=200)]))
    assert [e["id"] for e in run(episodes.query("u", "sqlite WAL", top_k=2, working_dir="/p"))] == ["e1", "e2"]


def test_episode_nearest_copy_scores_one(episodes):
    run(episodes.add([_ep(1, "Per-test tmp_path fixed the macOS CI flake")]))
    near = run(episodes.nearest("u", "LEARNING Per-test tmp_path fixed the macOS CI flake", "/p"))
    assert near["id"] == "e1" and near["similarity"] > episodes.DUPLICATE
    assert run(episodes.nearest("u", "Chose the MIT licence", "/p")) is None


def _decided(episodes) -> list[str]:
    """The episodes found by "what did we decide", which shares no word with the type name DECISION: it is found only
    through the type words the keyword index holds for it."""
    return [e["id"] for e in run(episodes.query("u", "what did we decide", top_k=3, working_dir="/p",
                                                min_similarity=0.01, recency_decay_rate=0.0))]


def test_episode_type_words_pass_the_integrity_check_and_survive_a_rebuild(episodes):
    """The keyword index holds each type with its everyday words; the integrity check and a rebuild must read the same
    text, or every saved episode reads as a broken index and `memory repair` strips the type words."""
    run(episodes.add([_ep(1, "Leftover cents go to the payer", "DECISION"), _ep(2, "Bumped requests", "EVENT")]))
    assert episodes._db.fts_problems() == []
    assert _decided(episodes) == ["e1"]
    episodes._db.rebuild_fts()
    assert episodes._db.fts_problems() == []
    assert _decided(episodes) == ["e1"]
    run(episodes.delete("u", "e1"))
    assert episodes._db.fts_problems() == [] and _decided(episodes) == []


def test_episode_index_from_before_the_type_words_view_is_rebuilt_on_open(tmp_path):
    """A memory.db whose episode index reads the episodes table directly (before the fix), already rebuilt by
    `memory repair` so its type words are gone: opening it recreates the index with them."""
    db = str(tmp_path / "memory.db")
    run(SQLiteEpisodeStore(db).add([_ep(1, "Leftover cents go to the payer", "DECISION")]))
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            DROP TABLE episodes_vocab; DROP TABLE episodes_fts;
            CREATE VIRTUAL TABLE episodes_fts USING fts5(
                event_type, anchor_event, summary, tags, content='episodes', tokenize='porter unicode61');
            CREATE VIRTUAL TABLE episodes_vocab USING fts5vocab(episodes_fts, 'row');
            INSERT INTO episodes_fts(episodes_fts) VALUES ('rebuild');""")
    reopened = SQLiteEpisodeStore(db)
    assert reopened._db.fts_problems() == []
    assert _decided(reopened) == ["e1"]
    assert run(reopened.query("u", "payer", top_k=3, working_dir="/p", min_similarity=0.01))[0]["id"] == "e1"


def test_episode_prune_count_list_delete(episodes):
    run(episodes.add([_ep(1, "a decision", "DECISION", days_old=300), _ep(2, "old event", "EVENT", days_old=300),
                      _ep(3, "new event", "EVENT")]))
    assert run(episodes.prune("u", "/p", cap=1)) == ["e2", "e3"]  # never the DECISION; the most faded event first
    assert run(episodes.count("u", "/p")) == 1
    assert [e["id"] for e in run(episodes.list_all("u"))] == ["e1"]
    assert run(episodes.delete("u", "e1")) and not run(episodes.delete("u", "e1"))


def test_backend_setting_selects_sqlite_without_chromadb(tmp_path, monkeypatch):
    from dostuff.config import get_config
    from dostuff.lib.memory import backend, chroma, stores
    cfg = get_config()
    monkeypatch.setattr(type(cfg), "get_data_dir", lambda self, *a, **k: tmp_path)
    monkeypatch.setattr(chroma, "chroma_available", lambda: False)
    monkeypatch.setattr(cfg, "memory_backend", "chroma")
    assert not backend.memory_available() and stores.get_semantic_store() is None
    monkeypatch.setattr(cfg, "memory_backend", "sqlite")
    assert backend.memory_available()
    assert isinstance(stores.get_semantic_store(), SQLiteFactStore)
    assert isinstance(stores.get_episodic_store(), SQLiteEpisodeStore)
    assert (tmp_path / "memory.db").exists()
