"""Background agents: notices, stop, approvals and Ctrl+B at registry level (subagents spec)."""
import asyncio

from dostuff.tools.bash.jobs import Notice as JobNotice


def _job_notice(job_id: str) -> JobNotice:
    return JobNotice(id=job_id, status="exited", code=0, runtime=1.0, killed_by=None, stdout_tail=[], stderr_tail=[])


# --- T4.1 shared notice queue ------------------------------------------------------------------

def test_notice_queue():
    from dostuff.helpers.agent.notices import AgentNotice, NoticeQueue
    q = NoticeQueue()
    job = _job_notice("bg-1")
    agent = AgentNotice(id="a1b2c3", agent_type="explore", description="find x", status="completed",
                        runtime=130.0, tokens=34_000, report="found it")
    assert job.kind == "job" and agent.kind == "agent"
    q.put("s1", job)
    q.put("s1", agent)
    q.put("s2", _job_notice("bg-9"))
    assert q.pending("s1") == [job, agent]
    assert q.pending("s1") == [job, agent]  # peeking leaves them
    assert q.drain("s1") == [job, agent] and q.drain("s1") == []
    assert [n.id for n in q.pending("s2")] == ["bg-9"]  # other owners untouched
    q.put("s1", agent)
    q.transfer("s1", "s3")
    assert q.pending("s1") == [] and q.drain("s3") == [agent]
    q.transfer("nobody", "s3")  # nothing to move: no error
    assert q.drain("s3") == []


def test_job_registry_uses_shared_queue(registry):
    """JobRegistry's notice API is unchanged, and only ever sees job notices."""
    from dostuff.helpers.agent.notices import AgentNotice, get_notice_queue
    queue = get_notice_queue()
    agent = AgentNotice(id="a1", agent_type="explore", description="d", status="completed", runtime=1.0,
                        tokens=0, report="r")
    queue.put("s1", agent)
    queue.put("s1", _job_notice("bg-1"))
    assert [n.id for n in registry.pending_notices("s1")] == ["bg-1"]
    assert [n.id for n in registry.drain_notices("s1")] == ["bg-1"]
    assert queue.drain("s1") == [agent]  # the job drain left the agent notice


def test_format_notices_both_kinds():
    from dostuff.helpers.agent.notices import AgentNotice, format_notices
    agent = AgentNotice(id="a1", agent_type="explore", description="d", status="completed", runtime=130.0,
                        tokens=34_000, report="found it")
    text = format_notices([_job_notice("bg-1"), agent])
    assert text.startswith('<background-job id="bg-1"')
    assert ('<background-agent id="a1" type="explore" description="d" status="completed" runtime="2m 10s" '
            'tokens="34k">') in text
    assert text.endswith("found it\n</background-agent>")


def test_loop_drains_both_kinds(fake_model, agents):
    """The loop puts job and agent notices into one user step before the model call (D7)."""
    from dostuff.agent.loop import loop
    from dostuff.helpers.agent.notices import AgentNotice, get_notice_queue
    get_notice_queue().put("s1", _job_notice("bg-1"))
    get_notice_queue().put("s1", AgentNotice(id="a1", agent_type="explore", description="d", status="completed",
                                             runtime=1.0, tokens=0, report="r"))
    history: list = []
    asyncio.run(loop(session_id="s1", turn_id="t", user_text="hi", dynamic_system_instruction="sys",
                     mcp_client=None, working_history=history, turn_type="interactive_loop",
                     stream=False))  # type: ignore[arg-type]
    notes = [s["content"] for s in fake_model.calls[0]["messages"] if "<background-" in str(s.get("content"))]
    assert len(notes) == 1 and "<background-job" in notes[0] and "<background-agent" in notes[0]
    assert get_notice_queue().pending("s1") == []


# --- T4.2 run_in_background=True ---------------------------------------------------------------

from conftest import FakeAdapter, FakeStep, fake_response, fake_tool_call  # noqa: E402


async def _until(predicate, timeout: float = 3.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")


def _agent(**kwargs):
    from dostuff.tools.agent import agent
    kwargs.setdefault("description", "find x")
    kwargs.setdefault("mcp_client", object())
    kwargs.setdefault("session_id", "s1")
    kwargs.setdefault("turn_id", "t1")
    return agent(**kwargs)


def test_background_start_result(agents, fake_model):
    import time
    agents.max_concurrent = 1
    gate_a = fake_model.gate("[A]", fake_response("report A"))
    gate_b = fake_model.gate("[B]", fake_response("report B"))

    async def go():
        start = time.monotonic()
        result = await _agent(prompt="[A] x", run_in_background=True, adapter=FakeAdapter())
        elapsed = time.monotonic() - start
        [run] = agents.runs()
        assert result == (f'Started agent "{run.id}" (general-purpose) in the background. '
                          "You'll get its report as a notice; don't poll.")
        assert elapsed < 0.2 and run.background and run.status == "running"
        queued = await _agent(prompt="[B] x", run_in_background=True, adapter=FakeAdapter())
        run_b = [r for r in agents.runs() if r is not run][0]
        assert queued == (f'Started agent "{run_b.id}" (general-purpose) in the background (queued: '
                          f"{agents.max_concurrent} agents are already running). You'll get its report as a "
                          "notice; don't poll.")
        assert run_b.status == "queued"
        gate_a.set()
        gate_b.set()
        await _until(lambda: run.status == run_b.status == "completed")
    asyncio.run(go())


def test_background_run_outlives_main_loop(agents, fake_model):
    from dostuff.agent.loop import loop
    call = fake_tool_call("agent", {"description": "bg", "prompt": "[B] x", "run_in_background": True}, "c1")
    fake_model.script("[MAIN]", fake_response(tool_calls=[call]), fake_response("main done"))
    gate = fake_model.gate("[B]", fake_response("report B"))

    async def go():
        working: list = [{"role": "user", "content": "[MAIN] go"}]
        result = await loop(session_id="s1", turn_id="t1", user_text="[MAIN] go", dynamic_system_instruction="",
                            mcp_client=object(), working_history=working, turn_type="interactive_loop",  # type: ignore[arg-type]
                            stream=False, adapter=FakeAdapter())
        assert result == "main done"
        [run] = agents.runs()
        assert run.status == "running"  # the main turn ended, the run didn't
        [tool] = [s for s in working if s["role"] == "tool"]
        assert tool["content"].startswith(f'Started agent "{run.id}"')
        gate.set()
        await _until(lambda: run.status == "completed")
    asyncio.run(go())


def test_background_notice(agents, fake_model):
    from dostuff.helpers.agent.notices import format_notices, get_notice_queue
    fake_model.script("[A]", fake_response("found it"))
    fake_model.script("[F]", ValueError("boom"))
    fake_model.script("[G]", fake_response("foreground report"))

    async def go():
        ok = await agents.start(owner="s1", agent_type="explore", description="find x", prompt="[A] x",
                                background=True)
        await _until(lambda: ok.status == "completed")
        bad = await agents.start(owner="s1", agent_type="explore", description="break", prompt="[F] x",
                                 background=True)
        await _until(lambda: bad.status == "failed")
        fg = await agents.start(owner="s1", agent_type="explore", description="fg", prompt="[G] x")
        await _until(lambda: fg.status == "completed")
        return ok, bad
    ok, bad = asyncio.run(go())
    notices = get_notice_queue().drain("s1")
    assert [(n.kind, n.id, n.status) for n in notices] == [("agent", ok.id, "completed"), ("agent", bad.id, "failed")]
    assert all(n.wake for n in notices)
    text = format_notices(notices[:1], max_chars=5)
    assert text.startswith(f'<background-agent id="{ok.id}" type="explore" description="find x" status="completed" '
                           'runtime="0s" tokens="')
    assert "found\n[… report truncated]" in text
    assert "boom" in notices[1].report


def test_background_notice_reaches_model(agents, fake_model):
    """Two background agents end while a foreground one runs: the main model's next call gets one user step
    with both blocks, appended before the call."""
    from dostuff.agent.loop import loop
    calls = [
        fake_tool_call("agent", {"description": "b1", "prompt": "[B] x", "run_in_background": True}, "c1"),
        fake_tool_call("agent", {"description": "b2", "prompt": "[D] x", "run_in_background": True}, "c2"),
        fake_tool_call("agent", {"description": "fg", "prompt": "[C] x"}, "c3"),
    ]
    fake_model.script("[MAIN]", fake_response(tool_calls=calls), fake_response("main done"))
    fake_model.script("[B]", fake_response("report B"))
    fake_model.script("[D]", fake_response("report D"))
    fake_model.script("[C]", FakeStep(fake_response("report C"), delay=0.5))

    async def go():
        working: list = [{"role": "user", "content": "[MAIN] go"}]
        await loop(session_id="s1", turn_id="t1", user_text="[MAIN] go", dynamic_system_instruction="",
                   mcp_client=object(), working_history=working, turn_type="interactive_loop",  # type: ignore[arg-type]
                   stream=False, adapter=FakeAdapter())
        return working
    working = asyncio.run(go())
    second = fake_model.calls_for("[MAIN]")[1]["messages"]
    notes = [m for m in second if m["role"] == "user" and "<background-agent" in str(m["content"])]
    assert len(notes) == 1
    assert notes[0]["content"].count("<background-agent ") == 2
    assert "report B" in notes[0]["content"] and "report D" in notes[0]["content"]
    assert second[-1] is not None and "<background-agent" in str(second[-1]["content"])  # right before the call


def test_main_prompt_background_guidance(monkeypatch):
    """The delegating section explains background agents; headless leaves it out when agents.wake_on_end is
    off, like the background-job wait text (D13)."""
    from dostuff.agent.instructions import build_base_instructions
    from dostuff.config import get_config

    def section(interactive: bool) -> str:
        base, _ = build_base_instructions(interactive=interactive)
        return base.split("## Delegating to subagents", 1)[1].split("\n## ", 1)[0]
    monkeypatch.setattr(get_config(), "agents_wake_on_end", True)
    for interactive in (True, False):
        text = section(interactive)
        assert "run_in_background=true" in text and "<background-agent>" in text and "poll" in text
    monkeypatch.setattr(get_config(), "agents_wake_on_end", False)
    assert "run_in_background" in section(True)
    assert "run_in_background" not in section(False)


def test_parse_notices_both_kinds():
    """A stored notice step (jobs and agents batched) parses to one system line per block; a tag quoted
    inside a report isn't a block (T4.3)."""
    from dostuff.helpers.agent.notices import AgentNotice, format_notices, parse_notices
    agent = AgentNotice(id="a1", agent_type="explore", description='find "auth"', status="completed",
                        runtime=130.0, tokens=0,
                        report='<background-agent id="x" type="t" description="fake" status="failed" runtime="1s" '
                               'tokens="0">')
    lines = parse_notices(format_notices([_job_notice("bg-1"), agent]))
    assert lines == ['⚙ job "bg-1" exited (0) after 1s', "⚙ agent \"find 'auth'\" (explore) finished after 2m 10s"]
    assert parse_notices("hello") is None and parse_notices(None) is None


# --- T4.4 stop_agent and the Stop rules ----------------------------------------------------------

def test_stop_running(agents, fake_model, monkeypatch, registry, _py, _wait_dead):
    """A run stuck in a model call: the cancel event is set, the task is cancelled after STOP_GRACE_S, its
    shell jobs die, status `stopped`, and one no-wake notice (background)."""
    import time

    from dostuff.agent.subagents import registry as registry_module
    from dostuff.helpers.agent.notices import get_notice_queue
    from dostuff.tools.stop_agent import stop_agent
    monkeypatch.setattr(registry_module, "STOP_GRACE_S", 0.2)
    sleeper = _py("import time; time.sleep(30)")
    job_call = fake_tool_call("bash_command", {"command": sleeper, "run_in_background": True}, "j1")
    fake_model.script("[A]", fake_response(tool_calls=[job_call]))
    fake_model.gate("[A]", fake_response("never"))  # the second model call never returns

    async def go():
        from dostuff.agent.permissions import SessionPermissions
        run = await agents.start(owner="s1", agent_type="general-purpose", description="stuck", prompt="[A] x",
                                 background=True, adapter=FakeAdapter(), permissions=SessionPermissions(mode="auto"))
        await _until(lambda: len(fake_model.calls_for("[A]")) == 2)
        [job] = [j for j in registry.jobs() if j.owner == f"s1/sub/{run.id}"]
        start = time.monotonic()
        result = await stop_agent(agent_id=run.id, session_id="s1")
        return run, job, result, time.monotonic() - start
    run, job, result, elapsed = asyncio.run(go())
    assert result == f'Stopped agent "{run.id}" (general-purpose).'
    assert run.status == "stopped" and run.cancel.is_set() and run.task.done() and run.stopped_by == "model"
    assert 0.15 < elapsed < 2.0
    assert job.killed_by == "owner_ended" and _wait_dead(job.pid, 5)
    [notice] = get_notice_queue().drain("s1")
    assert (notice.id, notice.status, notice.wake) == (run.id, "stopped", False)


def test_stop_edge_cases(agents, fake_model):
    from dostuff.helpers.agent.notices import get_notice_queue
    from dostuff.tools.stop_agent import stop_agent
    gate = fake_model.gate("[B]", fake_response("never"))

    async def go():
        done = await agents.start(owner="s1", agent_type="explore", description="quick", prompt="[A] x")
        await done.done
        other = await agents.start(owner="s2", agent_type="explore", description="theirs", prompt="[A] y")
        await other.done
        assert await stop_agent(agent_id=done.id, session_id="s1") == (
            f'Agent "{done.id}" already completed; nothing to stop.')
        unknown = await stop_agent(agent_id="nope", session_id="s1")
        assert unknown == f'[error] No agent "nope" in this session. Its agents: {done.id} (quick, explore, completed).'
        assert await stop_agent(agent_id=other.id, session_id="s1") == (
            f'[error] No agent "{other.id}" in this session. Its agents: {done.id} (quick, explore, completed).')
        assert await stop_agent(agent_id="nope", session_id="s3") == (
            '[error] No agent "nope" in this session. It has no agents.')
        bg = await agents.start(owner="s1", agent_type="explore", description="bg", prompt="[B] x", background=True)
        await _until(lambda: bool(fake_model.calls_for("[B]")))
        await agents.stop_all("session_end")
        assert bg.status == "stopped" and bg.stopped_by == "session_end"
        gate.set()
    asyncio.run(go())
    assert get_notice_queue().drain("s1") == []  # session end: nobody is left to read it


def test_stop_foreground_one_step(agents, fake_model, monkeypatch):
    """A foreground run stopped from outside: its waiting tool call gives one tool step saying so."""
    from dostuff.agent.loop import loop
    from dostuff.agent.subagents import registry as registry_module
    from dostuff.tools.stop_agent import stop_agent
    monkeypatch.setattr(registry_module, "STOP_GRACE_S", 0.1)
    call = fake_tool_call("agent", {"description": "fg", "prompt": "[A] x"}, "c1")
    fake_model.script("[MAIN]", fake_response(tool_calls=[call]), fake_response("main done"))
    gate = fake_model.gate("[A]", fake_response("never"))

    async def go():
        working: list = [{"role": "user", "content": "[MAIN] go"}]
        main = asyncio.create_task(loop(
            session_id="s1", turn_id="t1", user_text="[MAIN] go", dynamic_system_instruction="", mcp_client=object(),  # type: ignore[arg-type]
            working_history=working, turn_type="interactive_loop", stream=False, adapter=FakeAdapter()))
        await _until(lambda: bool(fake_model.calls_for("[A]")))
        [run] = agents.runs()
        await stop_agent(agent_id=run.id, session_id="s1")
        assert await main == "main done"
        gate.set()
        return working, run
    working, run = asyncio.run(go())
    tools = [s for s in working if s["role"] == "tool"]
    assert len(tools) == 1 and f'agent "{run.id}" was stopped' in tools[0]["content"]


# --- T4.5 approvals queue in order ---------------------------------------------------------------

class SerialAdapter(FakeAdapter):
    """A parent whose prompts show one at a time, like the TUI's `_ask_lock`: `asks` records them as shown."""

    def __init__(self):
        super().__init__()
        self.lock = asyncio.Lock()

    async def ask(self, message, **kwargs):
        async with self.lock:
            return await super().ask(message, **kwargs)


def _writer(fake_model, marker: str, path: str) -> None:
    fake_model.script(marker, fake_response(tool_calls=[fake_tool_call("write_file", {"path": path, "content": "x"})]),
                      fake_response(f"done {marker}"))


def test_approvals_in_order(agents, fake_model, tmp_path):
    from dostuff.agent.permissions import SessionPermissions
    from dostuff.helpers.tools.project_root import use_project_root
    for m in "ABC":
        _writer(fake_model, f"[{m}]", f"{m}.txt")

    async def go():
        parent = SerialAdapter()
        perms = SessionPermissions(mode="ask")
        runs = []
        with use_project_root(tmp_path):
            for m in "ABC":
                runs.append(await agents.start(owner="s1", agent_type="general-purpose", description=f"write {m}",
                                               prompt=f"[{m}] x", adapter=parent, permissions=perms))
                await _until(lambda: runs[-1].status == "waiting_approval")
            main = asyncio.create_task(parent.ask("main agent: Write file"))
            await asyncio.sleep(0.05)
            assert len(parent.asks) == 1  # one prompt at a time
            for i, m in enumerate("ABC"):
                assert parent.asks[i][0].startswith(f"[agent: write {m} · general-purpose] ")
                assert runs[i].status == "waiting_approval"
                parent.answer(i, "yes")
                await _until(lambda: runs[i].status == "completed")
                await _until(lambda: len(parent.asks) == i + 2)
            assert parent.asks[3][0] == "main agent: Write file"
            parent.answer(3, "yes")
            await main
        return runs
    runs = asyncio.run(go())
    assert [(tmp_path / f"{m}.txt").exists() for m in "ABC"] == [True] * 3


def test_waiting_run_blocks_nobody(agents, fake_model, tmp_path, monkeypatch):
    from dostuff.agent.loop import loop
    from dostuff.agent.permissions import SessionPermissions
    from dostuff.agent.subagents import registry as registry_module
    from dostuff.helpers.tools.project_root import use_project_root
    monkeypatch.setattr(registry_module, "STOP_GRACE_S", 0.2)
    (tmp_path / "r.txt").write_text("hello", encoding="utf-8")
    _writer(fake_model, "[A]", "a.txt")
    fake_model.script("[B]", fake_response(tool_calls=[fake_tool_call("read_file", {"path": "r.txt"})]),
                      fake_response("done B"))
    fake_model.script("[MAIN]", fake_response(tool_calls=[fake_tool_call("read_file", {"path": "r.txt"})]),
                      fake_response("main done"))

    async def go():
        parent = SerialAdapter()
        perms = SessionPermissions(mode="ask")
        with use_project_root(tmp_path):
            a = await agents.start(owner="s1", agent_type="general-purpose", description="w", prompt="[A] x",
                                   adapter=parent, permissions=perms, background=True)
            await _until(lambda: a.status == "waiting_approval")
            b = await agents.start(owner="s1", agent_type="general-purpose", description="r", prompt="[B] x",
                                   adapter=parent, permissions=perms, background=True)
            working: list = [{"role": "user", "content": "[MAIN] go"}]
            result = await loop(session_id="s1", turn_id="t1", user_text="[MAIN] go", dynamic_system_instruction="",
                                mcp_client=object(), working_history=working, turn_type="interactive_loop",  # type: ignore[arg-type]
                                stream=False, adapter=parent, permissions=perms)
            assert result == "main done"
            await _until(lambda: b.status == "completed")
            await asyncio.sleep(2 * registry_module.STOP_GRACE_S)
            assert a.status == "waiting_approval"  # no timeout
            parent.answer(0, "yes")
            await _until(lambda: a.status == "completed")
    asyncio.run(go())


def test_always_grant_shared(agents, fake_model, tmp_path):
    from dostuff.agent.permissions import SessionPermissions
    from dostuff.helpers.tools.project_root import use_project_root
    _writer(fake_model, "[A]", "a.txt")
    _writer(fake_model, "[B]", "b.txt")

    async def go():
        parent = SerialAdapter()
        perms = SessionPermissions(mode="ask")
        with use_project_root(tmp_path):
            a = await agents.start(owner="s1", agent_type="general-purpose", description="a", prompt="[A] x",
                                   adapter=parent, permissions=perms)
            await _until(lambda: len(parent.asks) == 1)
            parent.answer(0, "always")
            await a.done
            b = await agents.start(owner="s1", agent_type="general-purpose", description="b", prompt="[B] x",
                                   adapter=parent, permissions=perms)
            await b.done
        assert len(parent.asks) == 1
    asyncio.run(go())
    assert (tmp_path / "b.txt").exists()


# --- T4.8 Ctrl+B moves foreground agents to the background (registry level) -----------------------

def test_detach_foreground(agents, fake_model):
    from dostuff.helpers.agent.notices import get_notice_queue
    agents.max_concurrent = 1
    gate = fake_model.gate("[A]", fake_response("report A"))
    fake_model.script("[Q]", fake_response("report Q"))

    async def go():
        parent = FakeAdapter()
        call = asyncio.create_task(_agent(prompt="[A] x", adapter=parent))
        # The second call starts only once [A] holds the one slot: each call reloads definitions in a worker
        # thread first, so two calls started together reach the registry in either order (macOS CI saw [Q] win)
        await _until(lambda: bool(fake_model.calls_for("[A]")))
        queued_call = asyncio.create_task(_agent(prompt="[Q] x", description="later", adapter=parent))
        await _until(lambda: len(agents.runs()) == 2)
        run, queued = sorted(agents.runs(), key=lambda r: r.description != "find x")
        assert queued.status == "queued"
        moved = agents.detach_foreground("s1")
        assert {r.id for r in moved} == {run.id, queued.id}
        # Both calls return without the gate (still closed): a call Ctrl+B didn't release would hang here.
        # No wall-clock bound: a loaded runner made `< 0.5 s` flaky
        result, queued_result = await asyncio.wait_for(asyncio.gather(call, queued_call), timeout=10)
        assert not gate.is_set()
        assert result == f'Agent "{run.id}" moved to the background; you\'ll get its report as a notice.'
        assert queued_result.startswith(f'Agent "{queued.id}" moved to the background')
        assert run.background and run.status == "running"
        parent.cancel()  # ESC after that doesn't stop it
        await asyncio.sleep(0.05)
        assert not run.cancel.is_set()
        gate.set()
        await _until(lambda: run.status == queued.status == "completed")
        assert agents.detach_foreground("s1") == []  # nothing left to move
        return run, queued
    run, queued = asyncio.run(go())
    notices = get_notice_queue().drain("s1")
    assert {n.id for n in notices} == {run.id, queued.id}
    assert {n.report for n in notices} == {"report A", "report Q"}


# --- T4.10 no agent's jobs outlive the process ---------------------------------------------------

_ATEXIT_CHILD = r'''
import asyncio, sys
import os
sys.path[:0] = [sys.argv[1], os.path.dirname(sys.argv[1])]  # conftest, then this checkout
from conftest import FakeModel, fake_response, fake_tool_call
from dostuff.agent import call_agent, loop as loop_module
from dostuff.agent.permissions import SessionPermissions
from dostuff.agent.subagents.registry import get_agent_registry
from dostuff.tools.bash.jobs import get_registry
model = FakeModel()
model.script("[A]", fake_response(tool_calls=[fake_tool_call("bash_command", {"command": sys.argv[2],
                                                                              "run_in_background": True})]))
model.gate("[A]", fake_response("never"))
call_agent.acompletion = model.acompletion
loop_module.get_model_token_limit = lambda: 100_000

async def main():
    await get_agent_registry().start(owner="s1", agent_type="general-purpose", description="d", prompt="[A] x",
                                     background=True, permissions=SessionPermissions(mode="auto"))
    while not get_registry().jobs():
        await asyncio.sleep(0.05)
    print(get_registry().jobs()[0].pid, flush=True)

loop = asyncio.new_event_loop()
loop.run_until_complete(main())
sys.exit(0)  # no cleanup: the run's task is never cancelled, only atexit runs
'''


def test_atexit_no_agent_jobs_left(tmp_path, _py, _wait_dead):
    import subprocess
    import sys
    from pathlib import Path
    tests_dir = Path(__file__).resolve().parent
    script = tmp_path / "child.py"
    script.write_text(_ATEXIT_CHILD, encoding="utf-8")
    env = {**__import__("os").environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path),
           "MODEL": "openai/gpt-4o-mini", "OPENAI_API_KEY": "sk-test"}
    out = subprocess.run([sys.executable, str(script), str(tests_dir), _py("import time; time.sleep(30)")],
                         capture_output=True, text=True, timeout=60, cwd=tests_dir.parent, env=env)
    assert out.returncode == 0, out.stderr
    pid = int(out.stdout.strip().splitlines()[-1])
    assert _wait_dead(pid, 5)
