"""The hybrid memory backend (memory.backend: hybrid): SQLite of record, searched by keywords and embeddings."""
import asyncio
import json
import sqlite3
import time

import pytest

pytest.importorskip("chromadb")

from dostuff.lib.memory.hybrid_memory_store import HybridEpisodeStore, HybridFactStore  # noqa: E402
from dostuff.lib.memory.sqlite_memory_store import SQLiteEpisodeStore, SQLiteFactStore  # noqa: E402


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def facts(tmp_path):
    return HybridFactStore(str(tmp_path / "memory.db"), str(tmp_path / "memory-index"))


@pytest.fixture
def episodes(tmp_path):
    return HybridEpisodeStore(str(tmp_path / "memory.db"), str(tmp_path / "memory-index"))


def _ep(i, anchor, summary, event_type="DECISION", wd="/p"):
    meta = {"user_id": "u", "session_id": "s", "working_dir": wd, "event_type": event_type, "anchor_event": anchor,
            "summary": summary, "tags": json.dumps([]), "created_at_epoch": int(time.time()), "created_at_iso": ""}
    return {"id": f"e{i}", "vector_text": f"- EVENT TYPE: {event_type}\n  - ANCHOR EVENT: {anchor}\n  - SUMMARY: {summary}",
            "metadata": meta}


FACTS = [("py_errors", "Raises ValueError with a message naming the bad argument for invalid input"),
         ("keyboard", "Types on a Dvorak keyboard layout"),
         ("editor", "Uses Neovim with the LazyVim distribution")]


def _seed(facts):
    for key, value in FACTS:
        run(facts.upsert_fact("u", {"key": key, "value": value}))


def test_paraphrase_found_by_embeddings_where_keywords_find_nothing(facts):
    _seed(facts)
    prompt = "what exception do I throw on illegal values?"
    assert run(facts.keyword.query("u", prompt, top_k=3, min_similarity=SQLiteFactStore.RECALL_FLOOR)) == []
    found = run(facts.query("u", prompt, top_k=3, min_similarity=facts.RECALL_FLOOR))
    assert [f["key"] for f in found][:1] == ["py_errors"]


def test_keyword_hit_kept_and_unrelated_prompt_finds_nothing(facts):
    _seed(facts)
    assert [f["key"] for f in run(facts.query("u", "dvorak", top_k=3, min_similarity=facts.RECALL_FLOOR))][:1] == \
        ["keyboard"]
    assert run(facts.query("u", "hi", top_k=3, min_similarity=facts.RECALL_FLOOR)) == []


def test_each_fact_once_and_store_of_record_is_sqlite(facts):
    _seed(facts)
    found = run(facts.query("u", "ValueError for invalid input", top_k=5, min_similarity=0.0))
    assert len({f["key"] for f in found}) == len(found)
    assert {f["key"] for f in run(facts.list_all("u"))} == {k for k, _ in FACTS}
    assert run(facts.delete("u", "keyboard"))
    assert run(facts.embedding.query("u", "Dvorak keyboard layout", top_k=3, min_similarity=0.5)) == []
    assert "keyboard" not in {f["key"] for f in run(facts.list_all("u"))}


def test_broken_embedding_index_falls_back_to_keywords(facts, monkeypatch):
    _seed(facts)

    async def boom(*a, **k):
        raise RuntimeError("index divergence")

    monkeypatch.setattr(facts.embedding, "query", boom)
    monkeypatch.setattr(facts.embedding, "upsert_fact", boom)
    run(facts.upsert_fact("u", {"key": "shell", "value": "Uses fish as the login shell"}))
    assert [f["key"] for f in run(facts.query("u", "fish shell", top_k=3, min_similarity=facts.RECALL_FLOOR))] == \
        ["shell"]


def test_episode_union_and_reworded_duplicate(episodes):
    run(episodes.add([_ep(1, "Enabled WAL mode on the SQLite store", "Concurrent writers hit 'database is locked'"),
                      _ep(2, "Chose ruff over flake8", "flake8 kept fighting black over line length")]))
    found = run(episodes.query("u", "why do parallel writes to the db stop failing?", top_k=3, working_dir="/p"))
    assert [e["id"] for e in found][:1] == ["e1"]
    stats: dict = {}
    run(episodes.query("u", "linter choice", top_k=3, working_dir="/p", stats=stats))
    assert stats["matching"]
    same = run(episodes.nearest("u", "- EVENT TYPE: DECISION\n  - ANCHOR EVENT: Enabled WAL mode on the SQLite store\n"
                                     "  - SUMMARY: Concurrent writers hit 'database is locked'", "/p"))
    assert same and same["id"] == "e1" and same["similarity"] >= episodes.DUPLICATE
    assert run(episodes.nearest("u", "Bought a new desk lamp", "/p")) is None
    assert run(episodes.count("u", "/p")) == 2 and run(episodes.delete("u", "e2")) and run(episodes.count("u", "/p")) == 1


# ---- outbox: SQLite and the index can't share a transaction --------------------------------------------------------

def _pending(store) -> list[tuple[str, str]]:
    with sqlite3.connect(store.db_path) as conn:
        return [tuple(r) for r in conn.execute("SELECT kind, row_id FROM index_outbox ORDER BY seq")]


def _index_keys(facts) -> set[str]:
    return {f["key"] for f in run(facts.embedding.list_all("u"))}


async def _boom(*a, **k):
    raise RuntimeError("index down")


def test_write_queues_and_drains_the_index(facts):
    run(facts.upsert_fact("u", {"key": "shell", "value": "Uses fish as the login shell"}))
    assert _index_keys(facts) == {"shell"} and _pending(facts) == []


def test_index_down_on_save_keeps_the_change_queued_until_it_lands(facts, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(facts.embedding, "upsert_fact", _boom)
        run(facts.upsert_fact("u", {"key": "shell", "value": "Uses fish as the login shell"}))
    assert [f["key"] for f in run(facts.list_all("u"))] == ["shell"]  # saved: SQLite committed
    assert _pending(facts) == [("fact", "u:shell")] and _index_keys(facts) == set()
    run(facts.upsert_fact("u", {"key": "editor", "value": "Uses Helix"}))  # the next write drains the backlog
    assert _index_keys(facts) == {"shell", "editor"} and _pending(facts) == []


def test_forgotten_fact_left_in_the_index_is_never_recalled_and_is_removed_later(facts, monkeypatch):
    _seed(facts)
    prompt = "what exception do I throw on illegal values?"  # found by the embedding side only
    with monkeypatch.context() as m:
        m.setattr(facts.embedding, "delete_ids", _boom)
        assert run(facts.delete("u", "py_errors"))
    assert "py_errors" in _index_keys(facts)  # the orphan
    assert "py_errors" not in [f["key"] for f in run(facts.query("u", prompt, top_k=3, min_similarity=0.0))]
    assert _pending(facts) == [("fact", "u:py_errors")]
    assert run(facts.outbox.drain())
    assert "py_errors" not in _index_keys(facts) and _pending(facts) == []


def test_update_the_index_missed_reads_the_current_value(facts, monkeypatch):
    _seed(facts)
    with monkeypatch.context() as m:
        m.setattr(facts.embedding, "upsert_fact", _boom)
        run(facts.upsert_fact("u", {"key": "editor", "value": "Uses Neovim with the LazyVim distribution, in tmux"}))
    stale = {f["key"]: f["value"] for f in run(facts.embedding.list_all("u"))}["editor"]
    assert "tmux" not in stale
    found = {f["key"]: f["value"] for f in run(facts.query("u", "LazyVim", top_k=3, min_similarity=0.0))}
    assert found["editor"].endswith("in tmux")


def test_crash_between_commit_and_drain_is_caught_up_by_the_next_session(tmp_path):
    db, index = str(tmp_path / "memory.db"), str(tmp_path / "memory-index")
    first = HybridFactStore(db, index)
    run(first.keyword.upsert_fact("u", {"key": "shell", "value": "Uses fish as the login shell"}))  # no drain: "crash"
    second = HybridFactStore(db, index)
    assert _pending(second) == [("fact", "u:shell")]
    assert [f["key"] for f in run(second.query("u", "which login shell", top_k=3, min_similarity=0.0))] == ["shell"]
    assert _pending(second) == [] and _index_keys(second) == {"shell"}


def test_failed_drain_is_not_retried_by_every_search(facts, monkeypatch):
    calls = []

    async def counting_boom(*a, **k):
        calls.append(1)
        raise RuntimeError("index down")

    monkeypatch.setattr(facts.embedding, "upsert_fact", counting_boom)
    run(facts.upsert_fact("u", {"key": "shell", "value": "Uses fish as the login shell"}))
    run(facts.query("u", "fish", top_k=3, min_similarity=0.0))
    run(facts.query("u", "fish", top_k=3, min_similarity=0.0))
    assert len(calls) == 1  # the searches waited out _RETRY_AFTER
    run(facts.upsert_fact("u", {"key": "editor", "value": "Uses Helix"}))  # a write tries again
    assert len(calls) == 2


def test_entry_written_during_a_drain_stays_queued(facts):
    run(facts.keyword.upsert_fact("u", {"key": "shell", "value": "Uses fish"}))
    pending = run(facts.keyword._db.outbox_pending("fact", 10))
    run(facts.keyword.upsert_fact("u", {"key": "shell", "value": "Uses zsh"}))  # lands after the drain read
    run(facts.keyword._db.outbox_done("fact", pending))
    assert _pending(facts) == [("fact", "u:shell")]


def test_deleted_episode_left_in_the_index_does_not_block_saving_it_again(episodes, monkeypatch):
    run(episodes.add([_ep(1, "Enabled WAL mode on the SQLite store", "Concurrent writers hit 'database is locked'")]))
    text = ("- EVENT TYPE: DECISION\n  - ANCHOR EVENT: Enabled WAL mode on the SQLite store\n"
            "  - SUMMARY: Concurrent writers hit 'database is locked'")
    with monkeypatch.context() as m:
        m.setattr(episodes.embedding, "delete_ids", _boom)
        # the keyword side would find the row too, so it's taken out of the index as well as SQLite
        assert run(episodes.delete("u", "e1"))
    assert run(episodes.embedding.nearest("u", text, "/p")) is not None  # the orphan
    assert run(episodes.nearest("u", text, "/p")) is None
    assert run(episodes.query("u", "WAL mode database is locked", top_k=3, working_dir="/p")) == []


def test_pruned_episodes_leave_the_index(episodes):
    run(episodes.add([_ep(i, f"Event number {i}", f"summary {i}", event_type="EVENT") for i in range(3)]))
    doomed = run(episodes.prune("u", "/p", cap=1))
    assert len(doomed) == 2 and _pending(episodes) == []
    left = run(episodes.embedding.query("u", "Event number", top_k=5, working_dir="/p", min_similarity=0.0))
    assert {e["id"] for e in left} == {"e0", "e1", "e2"} - set(doomed)


def test_sqlite_backend_alone_writes_no_outbox(tmp_path):
    store = SQLiteFactStore(str(tmp_path / "memory.db"))
    run(store.upsert_fact("u", {"key": "shell", "value": "Uses fish"}))
    run(store.delete("u", "shell"))
    assert _pending(store) == []


def test_backend_setting_selects_hybrid_and_keywords_without_chromadb(tmp_path, monkeypatch):
    from dostuff.config import get_config
    from dostuff.lib.memory import backend, chroma, stores
    cfg = get_config()
    monkeypatch.setattr(type(cfg), "get_data_dir", lambda self, *a, **k: tmp_path)
    monkeypatch.setattr(cfg, "memory_backend", "hybrid")
    assert backend.memory_available()
    assert isinstance(stores.get_semantic_store(), HybridFactStore)
    assert isinstance(stores.get_episodic_store(), HybridEpisodeStore)
    monkeypatch.setattr(chroma, "chroma_available", lambda: False)
    assert backend.memory_available()
    assert isinstance(stores.get_semantic_store(), SQLiteFactStore)
    assert isinstance(stores.get_episodic_store(), SQLiteEpisodeStore)
