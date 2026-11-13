"""search_memory: the model's own lookup past the caps (memory recall spec D12, T3.1).

The facts step lists 50 facts and a turn recalls at most 3 overflow facts and 3 episodes; a prompt that needs more
(S10 in the live eval) can only get them by asking. Real Chroma in a temp dir; skipped without chromadb.
"""
import asyncio
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("chromadb")

from dostuff.lib.memory import episodic_memory_store as ems  # noqa: E402
from dostuff.lib.memory import semantic_memory_store as sms  # noqa: E402
from dostuff.tools.memory import search_memory as sm  # noqa: E402

USER = "u1"
DAY = 86400


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Both stores in a temp dir, cwd in project A, and the tool pointed at them."""
    from dostuff.lib.memory import chroma

    monkeypatch.setattr(chroma, "_clients", {})
    facts = sms.SemanticMemoryStore(persist_path=str(tmp_path / "chroma"))
    episodes = ems.EpisodicMemoryStore(persist_path=str(tmp_path / "chroma"))
    proj_a, proj_b = tmp_path / "projA", tmp_path / "projB"
    proj_a.mkdir()
    proj_b.mkdir()
    monkeypatch.chdir(proj_a)
    cfg = SimpleNamespace(get_user_id=lambda: USER, episodic_scope="project")
    monkeypatch.setattr(sm, "get_config", lambda: cfg)
    monkeypatch.setattr(sm, "get_semantic_store", lambda: facts)
    monkeypatch.setattr(sm, "get_episodic_store", lambda: episodes)
    return SimpleNamespace(facts=facts, episodes=episodes, a=str(proj_a.resolve()), b=str(proj_b.resolve()), cfg=cfg)


def _facts(store, rows):
    """Bulk-write facts as upsert_fact would, with chosen ages: rows are (key, value, days_ago, scope, working_dir)."""
    now = int(time.time())
    ids, docs, metas = [], [], []
    for key, value, days, scope, wd in rows:
        ts = now - days * DAY
        meta = {"user_id": USER, "key": key, "value": value, "scope": scope, "source": "user_message",
                "created_at_epoch": ts, "updated_at_epoch": ts}
        if scope == "project":
            meta["working_dir"] = wd
        ids.append(sms.fact_id(USER, key, scope, wd))
        docs.append(value)
        metas.append(meta)
    store._collection.upsert(ids=ids, documents=docs, metadatas=metas)


def _episodes(store, rows):
    """rows are (id, event_type, anchor, summary, days_ago, working_dir)."""
    now = int(time.time())
    eps = []
    for eid, etype, anchor, summary, days, wd in rows:
        ts = now - days * DAY
        eps.append({"id": eid, "vector_text": f"- EVENT TYPE: {etype}\n  - ANCHOR EVENT: {anchor}\n  - SUMMARY: {summary}",
                    "metadata": {"user_id": USER, "session_id": "s", "working_dir": wd, "created_at_epoch": ts,
                                 "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                                 "event_type": etype, "anchor_event": anchor, "summary": summary, "tags": "[]"}})
    asyncio.run(store.add(eps))


def _search(query, **kw):
    return asyncio.run(sm.search_memory(query, **kw))


def _over_the_cap(env):
    """52 newer user fillers (so the user facts below are past the 50) and 31 newer project fillers in A (so A's
    oldest project fact is past the 30 project slots), plus the facts the tests look for."""
    fill_u = [(f"hobby_{i:02d}", f"Keeps a collection of {i} houseplants", 1 + i % 40, "user", None) for i in range(52)]
    fill_p = [(f"note_{i:02d}", f"In this project, note {i} is filed under docs/notes", 1 + i % 20, "project", env.a)
              for i in range(31)]
    _facts(env.facts, fill_u + fill_p + [
        ("shell", "Uses PowerShell on Windows", 0, "user", None),                       # newest: in the facts step
        ("db_dislike", "Dislikes working with MongoDB", 400, "user", None),             # past the 50
        ("ticket", "Commit messages start with the ticket id TKV-<number>", 300, "project", env.a),  # past the 30
        ("ticket", "Commit messages start with the ticket id ABC-<number>", 300, "project", env.b),  # other project
    ])


def test_search_memory_scopes(env):
    _over_the_cap(env)
    _episodes(env.episodes, [
        ("ep_a", "LEARNING", "Tests that shared one tinykv.db file interfered", "Each test gets its own database.", 30,
         env.a),
        ("ep_b", "LEARNING", "Tests that shared one database file interfered in project B", "Fixed in B.", 30, env.b),
    ])

    facts = _search("Which databases do I dislike working with?", kind="facts")
    assert "MongoDB" in facts

    ticket = _search("commit message ticket id", kind="facts")
    assert "TKV-" in ticket and "ABC-" not in ticket  # this project's fact only
    assert "[project]" in ticket

    # A fact the facts step already lists is not searched: it is in context already
    assert "PowerShell" not in _search("Which shell do I use? PowerShell", kind="facts")

    eps = _search("tests sharing a database file", kind="episodes")
    assert "tinykv.db" in eps and "project B" not in eps

    both = _search("tests sharing a database file")  # kind="all": both sections
    assert "Episodes" in both and "Facts" in both


def test_search_memory_below_the_cap_says_facts_are_listed(env):
    _facts(env.facts, [("db_dislike", "Dislikes working with MongoDB", 400, "user", None),
                       ("ticket", "Commit messages start with TKV-<number>", 300, "project", env.a)])
    out = _search("Which databases do I dislike?", kind="facts")
    assert "MongoDB" not in out
    assert "All 2 saved facts are already listed" in out


def test_search_memory_episodes_user_scope_searches_every_directory(env):
    env.cfg.episodic_scope = "user"
    _episodes(env.episodes, [("ep_b", "LEARNING", "Tests that shared one database file interfered in project B",
                              "Fixed in B.", 30, env.b)])
    assert "project B" in _search("tests sharing a database file", kind="episodes")


def test_search_memory_floor_and_no_decay(env):
    _episodes(env.episodes, [
        # Old and closest: with the turn's decay a 400-day-old LEARNING keeps ~2% of its score; here it leads
        ("ep_old", "LEARNING", "sqlite3 'database is locked' fixed with a busy timeout",
         "Two writers raised database is locked; sqlite3.connect(path, timeout=5) fixed it.", 400, env.a),
        ("ep_new", "EVENT", "Looked at SQLite settings", "Read through the SQLite connection options.", 1, env.a),
        ("ep_far", "DECISION", "Chose the MIT licence", "We picked MIT over Apache-2.0 for simplicity.", 1, env.a),
    ])
    out = _search("sqlite database is locked error", kind="episodes")
    assert "busy timeout" in out
    assert out.index("busy timeout") < out.index("SQLite settings")  # relevance, not recency
    assert "MIT" not in out  # under the 0.2 floor
    # Each result carries its type and date, so the model weighs age itself
    assert "LEARNING" in out and time.strftime("%Y-%m-%d", time.gmtime(time.time() - 400 * DAY)) in out


def test_search_memory_limit_and_nothing_found(env):
    _episodes(env.episodes, [(f"ep_{i}", "LEARNING", f"Test lesson {i}: tests sharing a database file broke",
                              f"Lesson {i}.", 10 + i, env.a) for i in range(8)])
    out = _search("tests sharing a database file", kind="episodes", limit=2)
    assert out.count("Test lesson") == 2
    assert "Nothing" in _search("the colour of the moon on Tuesdays", kind="episodes")
    with pytest.raises(ValueError):
        _search("   ")


def test_search_memory_memory_off(monkeypatch):
    monkeypatch.setattr(sm, "get_semantic_store", lambda: None)
    monkeypatch.setattr(sm, "get_episodic_store", lambda: None)
    with pytest.raises(RuntimeError, match="memory is off"):
        _search("anything")


# Registration ---------------------------------------------------------------------------------------------------

def test_search_memory_is_a_memory_tool_offered_only_with_memory(monkeypatch):
    from dostuff.lib.memory import chroma
    from dostuff.tools.definitions import MEMORY_TOOLS, TOOL_MAP, default_tool_names
    assert TOOL_MAP["search_memory"].kind == "read"
    assert "search_memory" in MEMORY_TOOLS
    monkeypatch.setattr(chroma, "chroma_available", lambda: False)
    assert "search_memory" not in default_tool_names()


def test_search_memory_plan_mode_allowed():
    from dostuff.agent.permissions import PermissionConfig, check
    for mode in ("ask", "allowlist", "auto", "plan"):
        decision = check("search_memory", "read", {"query": "q"}, None, PermissionConfig(mode=mode))  # type: ignore[arg-type]
        assert decision.action == "allow", mode


def test_search_memory_not_for_subagents():
    from dostuff.agent.subagents import RESTRICTED_SUBAGENT_TOOLS
    assert "search_memory" in RESTRICTED_SUBAGENT_TOOLS


def test_prompts_point_at_search_memory():
    from dostuff.helpers.agent.constants import MEMORY_INSTRUCTIONS
    from dostuff.helpers.agent.turn_context import EPISODES_MORE_NOTE
    assert "search_memory" in MEMORY_INSTRUCTIONS
    assert "search_memory" in EPISODES_MORE_NOTE


def _many_overflow_facts(env, n):
    """52 newer fillers fill the 50; then `n` older style facts, all past the cap and all matching 'code style'."""
    fill = [(f"hobby_{i:02d}", f"Keeps a collection of {i} houseplants", 1 + i % 40, "user", None) for i in range(52)]
    style = [(f"style_{i:02d}", f"Python code style rule {i}: format helper functions this way", 200 + i, "user", None)
             for i in range(n)]
    _facts(env.facts, fill + style)


def test_search_memory_facts_default_ten_up_to_twenty(env):
    _many_overflow_facts(env, 25)
    assert _search("Python code style rules", kind="facts").count("code style rule") == 10
    assert _search("Python code style rules", kind="facts", limit=20).count("code style rule") == 20
    assert _search("Python code style rules", kind="facts", limit=99).count("code style rule") == 20


def test_search_memory_episodes_default_five_up_to_ten(env):
    _episodes(env.episodes, [(f"ep_{i}", "LEARNING", f"Test lesson {i}: tests sharing a database file broke",
                              f"Lesson {i}.", 10 + i, env.a) for i in range(12)])
    assert _search("tests sharing a database file", kind="episodes").count("Test lesson") == 5
    assert _search("tests sharing a database file", kind="episodes", limit=99).count("Test lesson") == 10


def test_search_memory_says_when_more_facts_match(env):
    _many_overflow_facts(env, 25)
    out = _search("Python code style rules", kind="facts", limit=20)  # 25 match, 20 shown
    assert "More facts match" in out and "narrower" in out


def test_search_memory_no_more_note_when_all_matches_shown(env):
    _many_overflow_facts(env, 3)
    out = _search("Python code style rules", kind="facts")
    assert out.count("code style rule") == 3 and "More facts match" not in out


# event_type filter ----------------------------------------------------------------------------------------------
# "What did I correct you on?" names a kind of memory, not its words: a correction worded by what happened ("removed
# an any from the cart reducer") scores under the 0.2 floor against it. With the type given, every episode of that
# type is ranked without the floor; today's unfiltered results follow, so a wrong type never loses what the search
# would have found without it (offline benchmark plan/2026-11-11-memory-recall/t2_search, 2026-11-13).

def _corrections_and_others(env):
    _episodes(env.episodes, [
        ("c_any", "CORRECTION", "User removed an any I added to the cart reducer",
         "Use a discriminated union for cart actions instead.", 20, env.a),
        ("c_log", "CORRECTION", "User rewrote my f-string log lines as structlog key-values",
         "log.info('loaded', rows=n), not log.info(f'loaded {n} rows').", 25, env.a),
        ("c_other", "CORRECTION", "User changed my tab indentation to spaces in project B", "Spaces.", 25, env.b),
        ("d_uv", "DECISION", "Switched the pipeline from Poetry to uv", "Lock resolution took minutes.", 10, env.a),
        ("d_css", "DECISION", "Dropped Tailwind for CSS modules", "Class lists got unreadable.", 12, env.a),
    ])


def test_search_memory_event_type_ranks_that_type_without_the_floor(env):
    _corrections_and_others(env)
    query = "things the user corrected me on"
    plain = _search(query, kind="episodes")
    assert "cart reducer" not in plain  # under the floor without the filter: the case this exists for
    out = _search(query, kind="episodes", event_type="CORRECTION")
    assert "cart reducer" in out and "structlog" in out
    assert "project B" not in out  # still this project's episodes only
    typed = out.split("Other episodes")[0]
    assert "Poetry" not in typed and "Tailwind" not in typed  # the typed section holds that type only


def test_search_memory_wrong_event_type_keeps_the_unfiltered_results(env):
    _episodes(env.episodes, [
        ("l_lock", "LEARNING", "sqlite3 'database is locked' fixed with a busy timeout",
         "Two writers raised database is locked; timeout=5 fixed it.", 40, env.a),
        ("d_mit", "DECISION", "Chose the MIT licence", "We picked MIT over Apache-2.0.", 5, env.a),
    ])
    query = "sqlite database is locked error"
    assert "busy timeout" in _search(query, kind="episodes")
    out = _search(query, kind="episodes", event_type="DECISION")  # wrong type for this question
    assert "busy timeout" in out and "Other episodes" in out
    assert out.count("busy timeout") == 1  # shown once, not in both sections


def test_search_memory_event_type_none_saved(env):
    _corrections_and_others(env)
    out = _search("what did we learn", kind="episodes", event_type="LEARNING")
    assert "No LEARNING episodes" in out


def test_search_memory_event_type_rejects_bad_input(env):
    with pytest.raises(ValueError, match="event_type"):
        _search("anything", kind="episodes", event_type="IDEA")
    with pytest.raises(ValueError, match="episodes"):
        _search("anything", kind="facts", event_type="DECISION")


def test_episode_store_query_filters_by_event_type(env):
    _corrections_and_others(env)
    got = asyncio.run(env.episodes.query(USER, "cart reducer", top_k=10, working_dir=env.a, min_similarity=-1.0,
                                         recency_decay_rate=0.0, event_type="DECISION"))
    assert {e["event_type"] for e in got} == {"DECISION"} and len(got) == 2
