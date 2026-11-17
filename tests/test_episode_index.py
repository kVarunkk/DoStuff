"""The episode index step: one line per saved episode of this project, so a past decision is in view even when the
prompt shares nothing with it (memory.backend: sqlite has no embeddings to bridge a paraphrase)."""
import asyncio
import time

import pytest

from dostuff.helpers.agent import turn_context as tc
from tests.test_loop import _interactive, _msg, _tc

DAY = 86400
NOW = 1_800_000_000
USER = "u1"


def _ep(i, kind, anchor, days_old=0):
    return {"id": f"e{i}", "event_type": kind, "anchor_event": anchor, "created_at_epoch": NOW - days_old * DAY}


def test_lasting_types_first_then_by_recency_weight():
    eps = [_ep(1, "EVENT", "CI flaked", 30), _ep(2, "DECISION", "WAL mode", 300), _ep(3, "LEARNING", "busy timeout", 20),
           _ep(4, "CORRECTION", "use tabs", 2), _ep(5, "EVENT", "old bump", 200)]
    order = [e["id"] for e in tc.select_episode_index(eps, now=NOW)]
    assert order[:2] == ["e4", "e2"]  # decisions/corrections newest first, however old
    assert order[2:] == ["e3", "e1", "e5"]  # a 20-day lesson outweighs a 30-day event: it fades slower


def test_budget_cuts_and_text_says_so(monkeypatch):
    monkeypatch.setattr(tc, "EPISODE_INDEX_TOKEN_BUDGET", 20)
    eps = [_ep(i, "DECISION", f"decision number {i} about the store", i) for i in range(5)]
    text, fingerprint = tc.episode_index_content(eps, "/work/proj-a", now=NOW)
    assert 'project="proj-a"' in text and 'saved="5"' in text and len(fingerprint) < 5
    assert "Older ones are not listed" in text
    assert "- [DECISION " in text and "search_memory" in text


def test_nothing_saved_no_text():
    assert tc.episode_index_content([], "/p") == ("", ())


def test_fingerprint_changes_with_anchor_and_counts_as_memory():
    a = tc.episode_index_content([_ep(1, "DECISION", "WAL")], "/p", now=NOW)
    b = tc.episode_index_content([_ep(1, "DECISION", "WAL mode")], "/p", now=NOW)
    assert a[1] != b[1]
    step = tc.make_episode_index_step(*a)
    assert tc.delivered_episode_index([step]) == a[1] and tc.memory_sections(step) == [step["content"]]


class FakeEpisodes:
    def __init__(self, episodes=()):
        self.episodes = list(episodes)
        self.dirs: list = []

    async def list_all(self, user_id, working_dir=None):
        assert user_id == USER
        self.dirs.append(working_dir)
        return list(self.episodes)


def _index_steps(history):
    return [s for s in history if "episode_index" in (s.get("meta") or {})]


def _run(monkeypatch, episodes, responses, **kw):
    from tests.test_memory_delivery import FakeFacts
    return _interactive(monkeypatch, responses, memory_store=FakeFacts(), memory_user_id=USER,
                        facts_project="/work/proj-a", episodic_store=episodes, **kw)


def test_first_call_sends_index_once(monkeypatch):
    store = FakeEpisodes([_ep(1, "DECISION", "Enabled WAL journal mode", 10)])

    async def read(name, args, **kw):
        return "x"
    _, working, calls = _run(monkeypatch, store, [_msg(tool_calls=[_tc("read_file", '{"path": "a"}', "c1")]),
                                                  _msg(content="ok")], dispatch=read)
    [step] = _index_steps(working)
    assert "Enabled WAL journal mode" in step["content"] and step in calls[0]["history"]
    assert store.dirs[0] is not None  # this directory's episodes (episodic_scope: project)


def test_new_episode_resends_and_none_saved_sends_nothing(monkeypatch):
    _, working, _ = _run(monkeypatch, FakeEpisodes(), [_msg(content="ok")])
    assert _index_steps(working) == []
    store = FakeEpisodes([_ep(1, "DECISION", "WAL")])
    working.append({"role": "user", "content": "again"})
    _, working, _ = _run(monkeypatch, store, [_msg(content="ok")], working=working)
    store.episodes.append(_ep(2, "LEARNING", "busy timeout"))
    working.append({"role": "user", "content": "and again"})
    _, working, _ = _run(monkeypatch, store, [_msg(content="ok")], working=working)
    steps = _index_steps(working)
    assert len(steps) == 2 and "busy timeout" in steps[-1]["content"]


def test_subagent_run_without_store_gets_no_index(monkeypatch):
    from tests.test_memory_delivery import FakeFacts
    _, working, _ = _interactive(monkeypatch, [_msg(content="ok")], memory_store=FakeFacts(), memory_user_id=USER,
                                 facts_project="/work/proj-a")
    assert _index_steps(working) == []


def test_sqlite_type_words_find_a_decision_by_decide(tmp_path):
    from dostuff.lib.memory.sqlite_memory_store import SQLiteEpisodeStore, fts5_available
    if not fts5_available():
        pytest.skip("no FTS5")
    store = SQLiteEpisodeStore(str(tmp_path / "m.db"))
    meta = {"user_id": USER, "working_dir": "/p", "event_type": "DECISION", "anchor_event": "Enabled WAL journal mode",
            "summary": "Readers were blocked by writers", "tags": "[]", "created_at_epoch": int(time.time()),
            "created_at_iso": ""}
    asyncio.run(store.add([{"id": "e1", "metadata": meta}]))
    got = asyncio.run(store.query(USER, "what did we decide about the database?", top_k=3, working_dir="/p",
                                  min_similarity=0.01))
    assert [e["id"] for e in got] == ["e1"]
