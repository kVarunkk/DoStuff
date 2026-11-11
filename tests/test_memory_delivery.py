"""Saved facts reach the model in a facts step (plan/2026-11-11-memory-recall REQ-2, evals.md T2).

The loop is driven with scripted model replies (`_interactive` from test_loop.py); the store is an in-memory
fake with the real store's `list_all` scope semantics, so no Chroma is needed.
"""
import asyncio

import litellm

from dostuff.agent import loop as loop_module
from dostuff.agent.call_agent import provider_messages
from dostuff.helpers.agent.turn_context import make_context_step
from conftest import fake_response
from tests.test_loop import _interactive, _msg, _tc, Adapter

USER = "u1"
PROJ_A = "/work/proj-a"
PROJ_B = "/work/proj-b"


class FakeFacts:
    """`list_all` like SemanticMemoryStore's: user facts plus `working_dir`'s project facts, newest first."""

    def __init__(self, facts=(), error=None):
        self.facts = [dict(f) for f in facts]
        self.error = error
        self.reads = 0

    def add(self, key, value, scope="user", working_dir=None, updated=0):
        fact = {"key": key, "value": value, "source": "user_message", "scope": scope,
                "created_at_epoch": updated, "updated_at_epoch": updated}
        if scope == "project":
            fact["working_dir"] = working_dir
        self.facts = [f for f in self.facts
                      if (f["key"], f["scope"], f.get("working_dir")) != (key, scope, fact.get("working_dir"))]
        self.facts.append(fact)
        return self

    def forget(self, key):
        self.facts = [f for f in self.facts if f["key"] != key]

    async def list_all(self, user_id, scope=None, working_dir=None):
        self.reads += 1
        if self.error is not None:
            raise self.error
        assert user_id == USER
        return [f for f in sorted(self.facts, key=lambda f: -f["updated_at_epoch"])
                if f["scope"] == "user" or f.get("working_dir") == working_dir]


def _facts_steps(history):
    return [s for s in history if "facts" in (s.get("meta") or {})]


def _run(monkeypatch, store, responses, project=PROJ_A, **kw):
    return _interactive(monkeypatch, responses, memory_store=store, memory_user_id=USER, facts_project=project, **kw)


def _read_call(call_id="c1"):
    return _tc("read_file", '{"path": "a.py"}', call_id)


async def _read_dispatch(name, args, **kw):
    return "x"


def test_first_turn_sends_facts_step(monkeypatch):
    store = FakeFacts().add("name", "Admiral Nemo", updated=2).add("linter", "ruff", "project", PROJ_A, updated=1)
    _, working, calls = _run(monkeypatch, store, [_msg(content="ok")])
    [step] = _facts_steps(working)
    assert calls[0]["history"][-1] == step
    assert '<user_facts shown="1" saved="1">\n- name: Admiral Nemo\n</user_facts>' in step["content"]
    assert '<project_facts project="proj-a" shown="1" saved="1">\n- linter: ruff\n</project_facts>' in step["content"]
    # It rides with the prompt in one user message, after it
    last = provider_messages(calls[0]["history"])[-1]
    assert last["role"] == "user" and last["content"].index("hi") < last["content"].index("<user_facts")


def test_no_facts_no_step(monkeypatch):
    _, working, _ = _run(monkeypatch, FakeFacts(), [_msg(content="ok")])
    assert _facts_steps(working) == []


def test_unchanged_facts_no_new_step_and_messages_byte_identical(monkeypatch):
    store = FakeFacts().add("name", "Nemo")
    _, working, calls = _run(monkeypatch, store, [_msg(tool_calls=[_read_call()]), _msg(content="ok")],
                             dispatch=_read_dispatch)
    assert len(_facts_steps(working)) == 1
    first, second = (provider_messages(c["history"]) for c in calls)
    assert second[:len(first)] == first  # the second call only extends the first: the cached prefix holds
    # A later turn on the same history adds nothing either
    working.append({"role": "user", "content": "again"})
    _, working, _ = _run(monkeypatch, store, [_msg(content="ok")], working=working)
    assert len(_facts_steps(working)) == 1


def test_save_midturn_resends_on_next_call(monkeypatch):
    store = FakeFacts().add("name", "Nemo", updated=1)

    async def save(name, args, **kw):
        store.add("editor", "Neovim", updated=2)
        return "Saved memory 'editor': Neovim"
    _, working, calls = _run(monkeypatch, store, [_msg(tool_calls=[_read_call()]), _msg(content="ok")],
                             dispatch=save)
    assert len(_facts_steps(calls[0]["history"])) == 1
    newest = _facts_steps(calls[1]["history"])[-1]
    assert calls[1]["history"][-1] == newest
    assert "- editor: Neovim\n- name: Nemo" in newest["content"]  # the full list, newest first
    assert "This replaces any earlier list of saved facts." in newest["content"]


def test_forget_resends(monkeypatch):
    store = FakeFacts().add("name", "Nemo").add("editor", "Neovim")
    _, working, _ = _run(monkeypatch, store, [_msg(content="ok")])
    store.forget("editor")
    working.append({"role": "user", "content": "again"})
    _, working, _ = _run(monkeypatch, store, [_msg(content="ok")], working=working)
    steps = _facts_steps(working)
    assert len(steps) == 2 and "editor" not in steps[-1]["content"] and "- name: Nemo" in steps[-1]["content"]
    # Forgetting the last one says so, rather than leaving the old list standing
    store.forget("name")
    working.append({"role": "user", "content": "and again"})
    _, working, _ = _run(monkeypatch, store, [_msg(content="ok")], working=working)
    assert "No facts are saved any more." in _facts_steps(working)[-1]["content"]


def test_facts_step_redelivered_after_compaction(monkeypatch):
    store = FakeFacts().add("name", "Nemo")
    _, working, _ = _run(monkeypatch, store, [_msg(content="ok")])
    # Compaction keeps only recent plain steps: the facts step went with the summarised part
    kept = [s for s in working if not _facts_steps([s])]
    kept.append({"role": "user", "content": "next"})
    _, working, calls = _run(monkeypatch, store, [_msg(content="ok")], working=kept)
    assert len(_facts_steps(calls[0]["history"])) == 1


def test_facts_step_redelivered_after_overflow_compaction(monkeypatch):
    store = FakeFacts().add("name", "Nemo")

    async def fake_compact(steps, keep_budget, notes, limit):
        return [{"role": "user", "content": "hi"}], "SUMMARY"  # the facts step went with the old steps
    monkeypatch.setattr(loop_module, "compact_context", fake_compact)
    overflow = litellm.ContextWindowExceededError("x", "m", "openai")
    _, _, calls = _run(monkeypatch, store, [overflow, _msg(content="ok")])
    assert len(calls) == 2 and len(_facts_steps(calls[1]["history"])) == 1
    assert calls[1]["history"][-1] == _facts_steps(calls[1]["history"])[0]


def test_project_facts_isolated_by_working_dir(monkeypatch):
    store = FakeFacts().add("linter", "ruff", "project", PROJ_A).add("linter", "eslint", "project", PROJ_B)
    _, working, _ = _run(monkeypatch, store, [_msg(content="ok")], project=PROJ_B)
    [step] = _facts_steps(working)
    assert "eslint" in step["content"] and "ruff" not in step["content"]
    assert '<project_facts project="proj-b"' in step["content"]


def test_store_error_reports_once_turn_runs(monkeypatch):
    store = FakeFacts(error=RuntimeError("index broken"))
    adapter = Adapter()
    result, working, calls = _run(monkeypatch, store, [_msg(tool_calls=[_read_call()]), _msg(content="ok")],
                                  dispatch=_read_dispatch, adapter=adapter)
    assert result == "ok" and len(calls) == 2 and _facts_steps(working) == []
    errors = [d for e, d in adapter.events if e == "error" and "index broken" in str(d)]
    assert len(errors) == 1 and store.reads == 2


def test_no_chromadb_no_step_no_tools(monkeypatch):
    from dostuff.lib.memory import chroma
    from dostuff.tools.definitions import MEMORY_TOOLS, default_tool_names
    _, working, _ = _interactive(monkeypatch, [_msg(content="ok")])  # no store passed: memory is off
    assert _facts_steps(working) == []
    monkeypatch.setattr(chroma, "chroma_available", lambda: False)
    assert not MEMORY_TOOLS & set(default_tool_names())


def test_facts_step_after_the_turns_bundle(monkeypatch):
    store = FakeFacts().add("name", "Nemo")
    working = [make_context_step("<turn-context>ENV</turn-context>"), {"role": "user", "content": "hi"}]
    _, working, _ = _run(monkeypatch, store, [_msg(content="ok")], working=working)
    assert working[0]["content"] == "<turn-context>ENV</turn-context>" and working[1]["content"] == "hi"
    assert working[2] == _facts_steps(working)[0]


def test_facts_read_off_the_event_loop(monkeypatch):
    """The store read is awaited, so a slow read doesn't block the TUI (the real store uses to_thread)."""
    store = FakeFacts().add("name", "Nemo")
    assert asyncio.iscoroutinefunction(store.list_all)
    _run(monkeypatch, store, [_msg(content="ok")])
    assert store.reads == 1


# T1.11 the bundle: overflow facts only ------------------------------------------------------------------------

class QueryRecorder(FakeFacts):
    def __init__(self, ranked=()):
        super().__init__()
        self.ranked = list(ranked)  # what a similarity query returns, best first
        self.queries = []

    async def query(self, user_id, query_text, top_k=3, min_similarity=0.0, scope=None, working_dir=None):
        self.queries.append({"top_k": top_k, "min_similarity": min_similarity, "scope": scope,
                             "working_dir": working_dir})
        return self.ranked[:top_k]


class NoEpisodes:
    async def query(self, **kwargs):
        return []


def _bundle(store, monkeypatch, tmp_path):
    from dostuff.agent import instructions
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(instructions, "build_environment_block", lambda: "")
    return asyncio.run(instructions.build_turn_context("which linter?", memory_store=store,
                                                       episodic_store=NoEpisodes(), user_id=USER))


def test_bundle_does_not_query_facts_below_cap(monkeypatch, tmp_path):
    store = QueryRecorder()
    for i in range(50):
        store.add(f"k{i:02d}", f"v{i}", updated=i)
    bundle = _bundle(store, monkeypatch, tmp_path)
    assert store.queries == [] and "facts>" not in bundle


def test_over_cap_newest_50_and_overflow_in_bundle(monkeypatch, tmp_path):
    from dostuff.helpers.agent.turn_context import render_facts_step, select_injected_facts
    store = QueryRecorder()
    for i in range(55):
        store.add(f"k{i:02d}", f"v{i}", updated=i)
    here = str(tmp_path.resolve())
    store.add("linter", "ruff", "project", here, updated=0)
    # The query ranks a shown fact first: it is already in the facts step, so it is not repeated
    by_key = {f["key"]: f for f in store.facts}
    store.ranked = [by_key[k] for k in ("k54", "k02", "linter", "k00", "k01", "k03")]

    shown, overflow = select_injected_facts(asyncio.run(store.list_all(USER, working_dir=here)))
    assert len(shown) == 50 and {f["key"] for f in overflow} == {"k00", "k01", "k02", "k03", "k04", "linter"}
    text = render_facts_step(shown, saved_user=55, saved_project=1, project="p")
    assert '<user_facts shown="50" saved="55">' in text and '<project_facts project="p" shown="0" saved="1">' in text

    bundle = _bundle(store, monkeypatch, tmp_path)
    [q] = store.queries
    assert q["min_similarity"] == 0.25 and q["working_dir"] == here and q["scope"] is None
    assert "<other_relevant_facts>\n- k02: v2\n- linter: ruff\n- k00: v0\n</other_relevant_facts>" in bundle
    assert "k54" not in bundle and "<relevant_user_facts>" not in bundle


# T1.12 typed subagents get facts, read-only (D14) -----------------------------------------------------------

def _subagent_store():
    return (FakeFacts().add("name", "Admiral Nemo", updated=2)
            .add("linter", "ruff", "project", PROJ_A, updated=1)
            .add("linter", "eslint", "project", PROJ_B, updated=1))


def _run_subagents(agents, fake_model, store, *starts):
    """Start each (marker, kwargs) as a subagent of a main session whose facts come from `store` in PROJ_A."""
    from dostuff.helpers.agent.turn_context import FactsSource, facts_source

    async def go():
        facts_source.set(FactsSource(store, USER, PROJ_A))
        for marker, kwargs in starts:
            fake_model.script(marker, fake_response("report"))
            kwargs = {"owner": "s1", "agent_type": "general-purpose", "description": "d", **kwargs}
            run = await agents.start(prompt=f"{marker} look into it", **kwargs)
            await run.done
    asyncio.run(go())


def _user_texts(call):
    return "\n".join(str(m.get("content")) for m in call["messages"] if m.get("role") == "user")


def test_typed_subagent_gets_facts_step(agents, fake_model, agents_home):
    _run_subagents(agents, fake_model, _subagent_store(), ("[A]", {}))
    first = _user_texts(fake_model.calls_for("[A]")[0])
    assert '<user_facts shown="1" saved="1">\n- name: Admiral Nemo\n</user_facts>' in first
    assert '<project_facts project="proj-a" shown="1" saved="1">\n- linter: ruff\n</project_facts>' in first
    assert "eslint" not in first


def test_omit_memory_type_gets_none(agents, fake_model, agents_home):
    from dostuff.agent.subagents import definitions
    (agents_home / ".dostuff" / "agents" / "quiet.md").write_text(
        "---\nname: quiet\ndescription: Keeps to itself.\nomit_memory: true\n---\nYou work alone.\n", encoding="utf-8")
    definitions.reload()
    assert definitions.get_definitions()["quiet"].omit_memory
    assert definitions.BUILTIN_TYPES["explore"].omit_memory and definitions.BUILTIN_TYPES["plan"].omit_memory
    assert not definitions.BUILTIN_TYPES["general-purpose"].omit_memory
    _run_subagents(agents, fake_model, _subagent_store(), ("[E]", {"agent_type": "explore"}),
                   ("[P]", {"agent_type": "plan"}), ("[Q]", {"agent_type": "quiet"}))
    for marker in ("[E]", "[P]", "[Q]"):
        assert "facts>" not in _user_texts(fake_model.calls_for(marker)[0]), marker


def test_subagent_no_episodes_no_save_tools(agents, fake_model, agents_home):
    _run_subagents(agents, fake_model, _subagent_store(), ("[A]", {}))
    for call in fake_model.calls_for("[A]"):
        texts = _user_texts(call)
        assert "<past_episodes>" not in texts and "<other_relevant_facts>" not in texts
        tools = {t["function"]["name"] for t in call.get("tools") or []}
        assert tools and not tools & {"save_memory", "save_episode"}


def test_worktree_subagent_uses_main_project_facts(agents, fake_model, agents_home, tmp_path, monkeypatch):
    from dostuff.lib.worktree import create_worktree
    from tests.test_subagents_run import _git_repo
    repo = _git_repo(tmp_path / "repo")
    monkeypatch.chdir(repo)
    store = _subagent_store().add("linter", "black", "project", str(repo), updated=1)

    from dostuff.helpers.agent.turn_context import FactsSource, facts_source

    async def go():
        facts_source.set(FactsSource(store, USER, str(repo)))
        fake_model.script("[W]", fake_response("report"))
        run = await agents.start(owner="s1", agent_type="general-purpose", description="d",
                                 prompt="[W] look", worktree=create_worktree("agent-facts", repo))
        await run.done
    asyncio.run(go())
    first = _user_texts(fake_model.calls_for("[W]")[0])
    assert f'<project_facts project="{repo.name}"' in first and "- linter: black" in first


def test_fork_unchanged(agents, fake_model, agents_home):
    """A fork inherits the main conversation, facts step included, and gets no second one."""
    from dostuff.agent.subagents.fork import ForkContext
    from dostuff.helpers.agent.turn_context import FactsSource, facts_source, facts_step_content, make_facts_step
    store = _subagent_store()
    text, fingerprint = facts_step_content(asyncio.run(store.list_all(USER, working_dir=PROJ_A)), PROJ_A)
    main_history = [{"role": "user", "content": "hi"}, make_facts_step(text, fingerprint),
                    {"role": "assistant", "content": "hello"}]
    store.add("editor", "Neovim", updated=3)  # saved since: only a fork that read facts itself would see it

    async def go():
        facts_source.set(FactsSource(store, USER, PROJ_A))
        fake_model.script("[F]", fake_response("report"))
        run = await agents.start(owner="s1", agent_type="fork", description="d", prompt="[F] look",
                                 fork=ForkContext("sys", "", main_history, ["read_file"], None))
        await run.done
    asyncio.run(go())
    texts = _user_texts(fake_model.calls_for("[F]")[0])
    assert texts.count("<user_facts") == 1 and "Neovim" not in texts
