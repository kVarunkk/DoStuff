"""Agent transcripts, send_message and /resume (subagents spec)."""
import asyncio
import subprocess

import pytest

from conftest import FakeAdapter, fake_response, fake_tool_call
from dostuff.agent import call_agent as call_agent_module
from dostuff.helpers.tools.project_root import project_root

FOOTER = "agent_id: {id} (send_message to continue)"


def _agent(**kwargs):
    from dostuff.tools.agent import agent
    kwargs.setdefault("description", "find x")
    kwargs.setdefault("mcp_client", object())
    kwargs.setdefault("session_id", "s1")
    kwargs.setdefault("turn_id", "t1")
    return agent(**kwargs)


def _send(**kwargs):
    from dostuff.tools.send_message import send_message
    kwargs.setdefault("mcp_client", object())
    kwargs.setdefault("session_id", "s1")
    kwargs.setdefault("turn_id", "t2")
    return send_message(**kwargs)


def _tick(call_id):
    return fake_response(tool_calls=[fake_tool_call("get_current_datetime", {}, call_id)])


# --- T5.4 send_message to a finished agent ----------------------------------------------------

def test_send_message_resumes(agents, fake_model, agent_type):
    from dostuff.helpers.agent.notices import get_notice_queue
    agent_type("capped", max_iterations=2)
    # run 1 uses its whole cap (two tool calls, then the wrap-up); run 2 gets a fresh one
    fake_model.script("[A]", _tick("c1"), _tick("c2"), fake_response("partial 1"), _tick("c3"),
                      fake_response("report 2"), fake_response("report 3"))

    async def go():
        first = await _agent(prompt="[A] find x", subagent_type="capped", adapter=FakeAdapter())
        [run] = agents.runs()
        assert first.startswith("[subagent did not finish") and first.endswith(FOOTER.format(id=run.id))
        before = list(run.history)
        calls_before = len(fake_model.calls_for("[A]"))
        second = await _send(to=run.id, message="and y?", adapter=FakeAdapter())
        resumed_call = fake_model.calls_for("[A]")[calls_before]
        assert resumed_call["messages"][1:] == before + [{"role": "user", "content": "and y?"}]
        assert second == f"report 2\n\n{FOOTER.format(id=run.id)}"
        assert run.status == "completed" and agents.runs() == [run]

        third = await _send(to=run.id, message="[bg] more", run_in_background=True, adapter=FakeAdapter())
        assert third == (f'Started agent "{run.id}" (capped) in the background. '
                         "You'll get its report as a notice; don't poll.")
        await run.done
        [notice] = get_notice_queue().drain("s1")
        assert notice.id == run.id and notice.report == "report 3" and notice.status == "completed"
        return run

    asyncio.run(go())


@pytest.mark.parametrize("ended", ["failed", "stopped", "interrupted"])
def test_send_message_after_any_end(agents, fake_model, ended):
    fake_model.script("[A]", fake_response("report 1"), fake_response("report 2"))

    async def go():
        await _agent(prompt="[A] x", adapter=FakeAdapter())
        [run] = agents.runs()
        run.status = ended  # as a model error, a stop or a restart would leave it
        result = await _send(to=run.id, message="again", adapter=FakeAdapter())
        assert result == f"report 2\n\n{FOOTER.format(id=run.id)}"
        assert run.status == "completed"
    asyncio.run(go())


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("a\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "init")
    monkeypatch.chdir(root)
    return root.resolve()


def _root_spy(monkeypatch, fake_model):
    """Record the subagent's project root and last user message at every model call."""
    seen: list = []
    fake = fake_model.acompletion

    async def spy(**kwargs):
        seen.append((project_root(), kwargs["messages"][-1]))
        return await fake(**kwargs)
    monkeypatch.setattr(call_agent_module, "acompletion", spy)
    return seen


def test_send_message_worktree(agents, fake_model, repo, monkeypatch):
    from dostuff.agent.permissions import SessionPermissions
    seen = _root_spy(monkeypatch, fake_model)
    perms = SessionPermissions(mode="auto")
    fake_model.script("[N]", fake_response("nothing changed"), fake_response("again"))
    write = fake_tool_call("write_file", {"path": "new.txt", "content": "hi\n"})
    fake_model.script("[C]", fake_response(tool_calls=[write]), fake_response("wrote"), fake_response("more"))

    async def go():
        # changed nothing: the worktree is removed, and the next run gets a new one of HEAD
        await _agent(prompt="[N] x", isolation="worktree", adapter=FakeAdapter(), permissions=perms)
        [quiet] = agents.runs()
        first_root = seen[-1][0]
        assert first_root != repo and not first_root.exists() and not quiet.worktree.path.exists()
        await _send(to=quiet.id, message="look again", adapter=FakeAdapter(), permissions=perms)
        root, last = seen[-1]
        assert root != repo and root.parent == first_root.parent  # a new worktree under .dostuff/worktrees
        assert "worktree" in last["content"] and "removed" in last["content"]
        assert last["content"].endswith("look again")

        # changed files: the same worktree is kept and reused
        await _agent(prompt="[C] x", isolation="worktree", adapter=FakeAdapter(), permissions=perms)
        busy = next(r for r in agents.runs() if r.id != quiet.id)
        kept = busy.worktree.path
        assert (kept / "new.txt").exists()
        await _send(to=busy.id, message="and more", adapter=FakeAdapter(), permissions=perms)
        root, last = seen[-1]
        assert root == kept and last == {"role": "user", "content": "and more"}

    asyncio.run(go())


def test_send_message_owner_and_unknown(agents, fake_model):
    fake_model.script("[A]", fake_response("report 1"))

    async def go():
        await _agent(prompt="[A] x", adapter=FakeAdapter())
        [run] = agents.runs()
        other = await _send(to=run.id, message="hi", session_id="s2", adapter=FakeAdapter())
        assert other == f'[error] No agent "{run.id}" in this session. It has no agents.'
        unknown = await _send(to="zzz", message="hi", adapter=FakeAdapter())
        assert unknown.startswith('[error] No agent "zzz" in this session. Its agents: ')
        assert f"{run.id} (find x, general-purpose, completed)" in unknown
    asyncio.run(go())


# --- T5.5 send_message to a running agent -----------------------------------------------------

async def _until(predicate, timeout: float = 3.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")


def test_send_message_inbox(agents, fake_model, tmp_path):
    import time

    from dostuff.agent.permissions import SessionPermissions
    from dostuff.lib.memory.session_store import SQLiteSessionStore
    agents.max_concurrent = 2
    # A is held in the call that gives its final answer; the messages arrive meanwhile
    gate_a = fake_model.gate("[A]", fake_response("first answer"))
    fake_model.script("[A]", fake_response("after m2"))
    # W asks to write a file and waits for approval
    fake_model.script("[W]", fake_response(tool_calls=[fake_tool_call("write_file", {"path": "w.txt", "content": "x"})]),
                      fake_response("wrote"))
    fake_model.script("[Q]", fake_response("q done"))

    async def go():
        store = SQLiteSessionStore(str(tmp_path / "sessions.db"))
        agents.attach_store(store)
        parent = FakeAdapter()
        try:
            from dostuff.helpers.tools.project_root import use_project_root
            with use_project_root(tmp_path):
                a = await agents.start(owner="s1", agent_type="general-purpose", description="a", prompt="[A] x",
                                       adapter=parent)
                w = await agents.start(owner="s1", agent_type="general-purpose", description="w", prompt="[W] x",
                                       adapter=parent, permissions=SessionPermissions(mode="ask"))
                q = await agents.start(owner="s1", agent_type="general-purpose", description="q", prompt="[Q] x",
                                       adapter=parent)
                await _until(lambda: bool(fake_model.calls_for("[A]")) and w.status == "waiting_approval")
                assert q.status == "queued"
                start = time.monotonic()
                for run, text in ((a, "m1"), (a, "m2"), (w, "wm"), (q, "qm")):
                    assert await _send(to=run.id, message=text, adapter=parent) == f'Queued for "{run.id}".'
                assert time.monotonic() - start < 0.5  # returns at once

                gate_a.set()
                await a.done
                parent.answer(0, "yes")
                await w.done
                await q.done
            await agents.flush()
            return a, w, q, await store.load_agent_steps(a.id)
        finally:
            await store.close()

    a, w, q, stored = asyncio.run(go())
    # A: the messages come after its last step, in order, and it keeps going instead of ending
    second = fake_model.calls_for("[A]")[1]["messages"]
    assert second[-3:] == [{"role": "assistant", "content": "first answer"},
                           {"role": "user", "content": "m1"}, {"role": "user", "content": "m2"}]
    assert a.result == "after m2" and a.inbox == []
    assert [s["content"] for s in stored if s["role"] == "user"] == ["[A] x", "m1", "m2"]
    # W: after the tool result of the call that waited for approval
    after = fake_model.calls_for("[W]")[1]["messages"]
    assert after[-2]["role"] == "tool" and after[-1] == {"role": "user", "content": "wm"}
    # Q: it was queued, so the message comes right after its task
    first_q = fake_model.calls_for("[Q]")[0]["messages"]
    assert first_q[1:] == [{"role": "user", "content": "[Q] x"}, {"role": "user", "content": "qm"}]


# --- T5.6 /resume and restart -----------------------------------------------------------------

def _row(agent_id, status, **over):
    row = {"id": agent_id, "session_id": "s1", "agent_type": "general-purpose", "description": f"d {agent_id}",
           "model": "m", "status": status, "background": False, "worktree": None, "branch": None,
           "started_at": "2026-11-01T10:00:00", "ended_at": None, "tokens": 5, "cost_usd": 0.0,
           "result": None, "compaction_notes": ""}
    return {**row, **over}


def test_resume_marks_interrupted(agents, fake_model, tmp_path):
    from dostuff.lib.memory.session_store import SQLiteSessionStore

    async def go():
        store = SQLiteSessionStore(str(tmp_path / "sessions.db"))
        try:
            for agent_id, status in (("r1", "running"), ("q1", "queued"), ("w1", "waiting_approval"),
                                     ("c1", "completed")):
                await store.upsert_agent_run(_row(agent_id, status, ended_at="2026-11-01T10:05:00"
                                                  if status == "completed" else None, result="rep"
                                                  if status == "completed" else None))
                await store.append_agent_step(agent_id, {"role": "user", "content": f"task {agent_id}"})
            await store.upsert_agent_run(_row("x1", "running", session_id="other"))
            loaded = await agents.load("s1", store)
            again = await agents.load("s1", store)  # loading twice adds nothing
            rows = {r["id"]: r["status"] for r in await store.load_agent_runs("s1")}
            return loaded, again, rows
        finally:
            await store.close()

    loaded, again, rows = asyncio.run(go())
    statuses = {r.id: r.status for r in loaded}
    assert statuses == {"r1": "interrupted", "q1": "interrupted", "w1": "interrupted", "c1": "completed"}
    assert again == [] and {r.id for r in agents.runs("s1")} == set(statuses)
    assert rows == statuses  # the store says so too
    c1 = agents.get("c1")
    assert c1.history == [{"role": "user", "content": "task c1"}] and c1.result == "rep" and c1.tokens == 5
    assert all(r.task is None for r in agents.runs()) and fake_model.calls == []  # nothing starts on load


def test_continue_after_restart(agents, fake_model, tmp_path, monkeypatch):
    from dostuff.agent.subagents import registry as agents_module
    from dostuff.lib.memory.session_store import SQLiteSessionStore
    db = str(tmp_path / "sessions.db")
    fake_model.script("[A]", fake_response("report 1"), fake_response("report 2"))

    async def first_process():
        store = SQLiteSessionStore(db)
        agents.attach_store(store)
        try:
            await _agent(prompt="[A] find x", adapter=FakeAdapter())
            [run] = agents.runs()
            await agents.flush()
            await store.compact_agent_steps(run.id, 1, "notes about x")  # as a compaction would
            return run.id
        finally:
            await store.close()

    agent_id = asyncio.run(first_process())

    async def second_process():
        fresh = agents_module.AgentRegistry()
        monkeypatch.setattr(agents_module, "_agent_registry", fresh)
        store = SQLiteSessionStore(db)
        fresh.attach_store(store)
        try:
            await fresh.load("s1", store)
            result = await _send(to=agent_id, message="[A] and y?", adapter=FakeAdapter())
            await fresh.flush()
            return result, fresh, await store.load_agent_steps(agent_id)
        finally:
            await store.close()

    result, fresh, transcript = asyncio.run(second_process())
    assert result == f"report 2\n\n{FOOTER.format(id=agent_id)}"
    call = fake_model.calls[-1]
    assert "notes about x" in call["messages"][0]["content"]
    assert call["messages"][1:] == [{"role": "assistant", "content": "report 1"},
                                    {"role": "user", "content": "[A] and y?"}]
    assert [s["content"] for s in transcript] == ["[A] find x", "report 1", "[A] and y?", "report 2"]
    assert fresh.get(agent_id).status == "completed"


# --- T5.8 main prompt: follow-ups -------------------------------------------------------------

def test_main_prompt_follow_ups(monkeypatch):
    """The delegating section says to continue an agent with send_message instead of starting over, that a
    running agent reads it before its next step, and not to poll; in every mode (D13)."""
    from dostuff.agent.instructions import build_base_instructions
    from dostuff.config import get_config
    for wake in (True, False):
        monkeypatch.setattr(get_config(), "agents_wake_on_end", wake)
        for interactive in (True, False):
            base, _ = build_base_instructions(interactive=interactive)
            text = base.split("## Delegating to subagents", 1)[1].split("\n## ", 1)[0]
            assert "send_message" in text and "agent_id" in text
            assert "instead of starting a new agent" in text
            assert "running" in text and "next step" in text
            assert "Don't poll" in text
