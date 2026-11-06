"""Subagent registry, foreground runs, adapter, prompt, the `agent` tool and concurrency (subagents spec)."""
import asyncio
import logging
import re
from types import SimpleNamespace

import pytest

from conftest import FakeAdapter, FakeStep, fake_response, fake_tool_call


def test_imports():
    from dostuff.agent.subagents.registry import AgentRegistry, AgentRun, get_agent_registry
    assert isinstance(get_agent_registry(), AgentRegistry)
    assert AgentRun.__dataclass_fields__["status"].default == "queued"


# --- T2.1: the code lives under agent/subagents/ -------------------------------------------------

def test_modules_moved():
    from dostuff.agent.subagents import RESTRICTED_SUBAGENT_TOOLS
    from dostuff.agent.subagents.adapter import SubagentAdapter
    from dostuff.agent.subagents.prompt import SUBAGENT_BASE, build_subagent_prompt
    from dostuff.agent.subagents.run import _close_worktree, _digest, _partial_report, run_agent_loop
    assert "delete_file" in RESTRICTED_SUBAGENT_TOOLS
    assert callable(SubagentAdapter) and SUBAGENT_BASE and callable(build_subagent_prompt)
    assert all(callable(f) for f in (_close_worktree, _digest, _partial_report, run_agent_loop))


# --- T2.2: AgentRegistry.start and the run lifecycle ----------------------------------------------

async def _until(predicate, timeout: float = 3.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")


def _start(agents, prompt: str, **kwargs):
    kwargs.setdefault("owner", "s1")
    kwargs.setdefault("agent_type", "general-purpose")
    kwargs.setdefault("description", "find x")
    return agents.start(prompt=prompt, **kwargs)


def test_start_lifecycle(agents, fake_model):
    async def go():
        gate = fake_model.gate("[A]", fake_response("the report"))
        run = await _start(agents, "[A] look", background=False)
        assert re.fullmatch(r"[0-9a-f]{6}", run.id)
        await _until(lambda: run.status == "running")
        assert run.ended is None and not run.done.done()
        gate.set()
        assert await run.done == "the report"
        assert run.status == "completed" and run.result == "the report" and run.ended is not None
        assert agents.get(run.id) is run
        ids = set()
        for i in range(50):
            ids.add((await _start(agents, f"[n{i}] x")).id)
        await asyncio.gather(*(r.done for r in agents.runs()))
        assert len(ids) == 50
    asyncio.run(go())


def test_start_failed_run(agents, fake_model, monkeypatch, caplog):
    from dostuff.agent import loop as loop_module
    from dostuff.agent.subagents import registry as registry_module
    monkeypatch.setattr(loop_module, "retry_delay", lambda *a: 0.0)

    async def go():
        # the model call fails: the run is failed, the partial-report text is kept (pinned in T1.2)
        fake_model.script("[A]", ValueError("model blew up"), fake_response("wrap-up"))
        run = await _start(agents, "[A] x")
        assert "model blew up" in await run.done and run.status == "failed"

        # the run itself raises: failed, `[error] …`, traceback logged once
        def boom(*a, **k):
            raise RuntimeError("bug in the harness")
        monkeypatch.setattr(registry_module, "build_subagent_prompt", boom)
        with caplog.at_level(logging.ERROR):
            run = await _start(agents, "[B] x")
            result = await run.done
        assert result.startswith("[error]") and "bug in the harness" in result and run.status == "failed"
        logged = [r for r in caplog.records if r.exc_info and "bug in the harness" in str(r.exc_info[1])]
        assert len(logged) == 1
    asyncio.run(go())


def test_run_outlives_caller(agents, fake_model):
    async def go():
        gate = fake_model.gate("[A]", fake_response("still done"))
        run = await _start(agents, "[A] x")

        async def caller():
            return await asyncio.shield(run.done)
        task = asyncio.create_task(caller())
        await _until(lambda: run.status == "running")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)
        assert run.status == "running"
        gate.set()
        assert await run.done == "still done" and run.status == "completed"
    asyncio.run(go())


# --- T2.3: each run has its own adapter state -----------------------------------------------------

def test_adapter_own_cancel(agents, fake_model, monkeypatch):
    from dostuff.agent.permissions import SessionPermissions
    from dostuff.config import get_config
    monkeypatch.setattr(get_config(), "subagent_max_iterations", 500)  # busy runs must outlast the test
    perms = SessionPermissions(mode="auto")
    tick = fake_tool_call("get_current_datetime", {})

    def busy(marker):  # keeps calling a tool, 0.05 s per model call, until cancelled
        fake_model.script(marker, *[FakeStep(fake_response(tool_calls=[tick]), delay=0.05) for _ in range(200)])

    async def go():
        parent = FakeAdapter()
        busy("[A]"), busy("[B]"), busy("[C]")
        a = await _start(agents, "[A] x", adapter=parent, permissions=perms, turn_id="t1")
        b = await _start(agents, "[B] x", adapter=parent, permissions=perms, turn_id="t1")
        c = await _start(agents, "[C] x", adapter=parent, permissions=perms, turn_id="t1", background=True)
        await _until(lambda: all(r.status == "running" for r in (a, b, c)))
        a.cancel.set()
        await a.done
        assert a.status == "stopped" and b.status == "running" and c.status == "running"
        parent.cancel()  # ESC in the main turn: foreground runs stop, the background one doesn't
        await b.done
        assert b.status == "stopped"
        await asyncio.sleep(0.2)
        assert c.status == "running"
        agents.cancel_turn("t1")  # the explicit form; still leaves a background run alone
        await asyncio.sleep(0.1)
        assert c.status == "running"
        c.cancel.set()
        await c.done
        assert c.status == "stopped"
    asyncio.run(go())


def test_adapter_prefix_and_usage(agents, fake_model):
    from dostuff.agent.permissions import SessionPermissions
    perms = SessionPermissions(mode="auto")
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120, prompt_tokens_details=None)
    tick = fake_tool_call("get_current_datetime", {})
    for marker in ("[A]", "[B]"):
        fake_model.script(marker, fake_response(tool_calls=[tick], usage=usage), fake_response("ok", usage=usage))

    async def go():
        fg_parent, bg_parent = FakeAdapter(), FakeAdapter()
        fg = await _start(agents, "[A] x", adapter=fg_parent, permissions=perms)
        bg = await _start(agents, "[B] x", adapter=bg_parent, permissions=perms, background=True,
                          description="bg one")
        await asyncio.gather(fg.done, bg.done)
        return fg, bg, fg_parent, bg_parent
    fg, bg, fg_parent, bg_parent = asyncio.run(go())

    assert any(k == "tool_call" for k, _ in fg_parent.events)
    assert all(d.startswith("[agent: find x] ") for k, d in fg_parent.events if k in ("tool_call", "tool_result"))
    assert not any(k == "status" for k, _ in fg_parent.events)
    assert [d["total_tokens"] for k, d in fg_parent.events if k == "usage"] == [120, 120]
    assert fg.tokens == 240 and bg.tokens == 240
    # background: no tool lines, but usage (session cost) and its start / end lines reach the parent
    assert not any(k in ("tool_call", "tool_result") for k, _ in bg_parent.events)
    assert [d["total_tokens"] for k, d in bg_parent.events if k == "usage"] == [120, 120]
    texts = [d for k, d in bg_parent.events if k == "system" and isinstance(d, str)]
    assert any("bg one" in t and "started" in t for t in texts)
    assert any("bg one" in t and "finished" in t for t in texts)


# --- T2.4: concurrency limit and FIFO queue -------------------------------------------------------

def test_queue_fifo(agents, fake_model):
    agents.max_concurrent = 2

    async def go():
        gates = {m: fake_model.gate(m, fake_response(f"report {m}")) for m in ("[1]", "[2]", "[3]", "[4]")}
        runs = [await _start(agents, f"{m} x") for m in gates]
        await _until(lambda: [r.status for r in runs] == ["running", "running", "queued", "queued"])
        assert not fake_model.calls_for("[3]")
        gates["[1]"].set()
        assert await runs[0].done == "report [1]"
        await _until(lambda: runs[2].status == "running")
        assert runs[3].status == "queued"
        # a run waiting for approval still holds its slot
        agents._set_status(runs[1], "waiting_approval")
        await asyncio.sleep(0.05)
        assert runs[3].status == "queued"
        agents._set_status(runs[1], "running")
        for m in ("[2]", "[3]", "[4]"):
            gates[m].set()
        assert await runs[3].done == "report [4]"  # the queued foreground caller gets its report
        await asyncio.gather(*(r.done for r in runs))
    asyncio.run(go())


def test_queue_stop_queued(agents, fake_model):
    agents.max_concurrent = 1

    async def go():
        gate = fake_model.gate("[1]", fake_response("one"))
        first = await _start(agents, "[1] x")
        queued = await _start(agents, "[2] x")
        await _until(lambda: first.status == "running" and queued.status == "queued")
        await agents.stop(queued, "user")
        assert queued.status == "stopped" and not fake_model.calls_for("[2]")
        assert first.status == "running"
        gate.set()
        assert await first.done == "one"
    asyncio.run(go())


# --- T2.5: the subagent system prompt -------------------------------------------------------------

def _git_repo(root):
    import subprocess
    root.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@example.com"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    (root / "AGENTS.md").write_text("Always follow RULE-7Q.\n")
    skill = root / ".dostuff" / "skills" / "zz"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: skill-zz9\ndescription: a test skill\n---\nbody\n")
    (root / ".gitignore").write_text(".dostuff/\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True, capture_output=True)
    return root.resolve()


def test_subagent_prompt_has_project_rules(agents, fake_model, monkeypatch, tmp_path, agents_home):
    from dostuff.agent.instructions import skills_summary
    from dostuff.agent.subagents.prompt import SUBAGENT_BASE
    from dostuff.helpers.skills.discover_skills import discover_skills
    from dostuff.lib.worktree import create_worktree
    repo = _git_repo(tmp_path / "repo")
    monkeypatch.chdir(repo)

    def system_of(marker):
        return fake_model.calls_for(marker)[0]["messages"][0]["content"]

    fake_model.script("[A]", fake_response("a"))
    fake_model.script("[B]", fake_response("b"))

    async def go():
        plain = await _start(agents, "[A] x")
        await plain.done
        wt = create_worktree("agent-p1", repo)
        isolated = await _start(agents, "[B] x", worktree=wt)
        await isolated.done
        return wt
    wt = asyncio.run(go())

    for marker, cwd in (("[A]", repo), ("[B]", wt.path)):
        text = system_of(marker)
        parts = [SUBAGENT_BASE.strip(), "Research and multi-step tasks", f"cwd: {cwd}\n",
                 "## Project Instructions", "RULE-7Q", "skill-zz9"]
        if marker == "[B]":
            parts.insert(2, f"git worktree at {wt.path}")
        positions = [text.find(p) for p in parts]
        assert -1 not in positions, (marker, [p for p, i in zip(parts, positions) if i == -1])
        assert positions == sorted(positions), marker
        assert "<relevant_user_facts>" not in text and "<past_episodes>" not in text
        assert skills_summary(discover_skills()) in text  # the same block as the main prompt, `dir:` included
        assert "  dir: " in text and "  location: " not in text
    assert "git worktree" not in system_of("[A]")


def test_subagent_prompt_built_once(agents, fake_model, monkeypatch):
    from dostuff.agent.permissions import SessionPermissions
    from dostuff.agent.subagents import registry as registry_module
    calls = []
    real = registry_module.build_subagent_prompt

    async def spy(*args, **kwargs):
        calls.append(args)
        return await real(*args, **kwargs)
    monkeypatch.setattr(registry_module, "build_subagent_prompt", spy)
    tick = fake_tool_call("get_current_datetime", {})
    fake_model.script("[A]", fake_response(tool_calls=[tick]), fake_response(tool_calls=[tick]), fake_response("ok"))

    async def go():
        run = await _start(agents, "[A] x", permissions=SessionPermissions(mode="auto"))
        return await run.done
    assert asyncio.run(go()) == "ok"
    assert len(fake_model.calls_for("[A]")) == 3 and len(calls) == 1


# --- T2.6: the `agent` tool replaces delegate_to_subagent -----------------------------------------

def _agent(**kwargs):
    from dostuff.tools.agent import agent
    kwargs.setdefault("description", "find x")
    kwargs.setdefault("mcp_client", object())
    kwargs.setdefault("session_id", "s1")
    kwargs.setdefault("turn_id", "t1")
    return agent(**kwargs)


def test_agent_tool_foreground(agents, fake_model, agent_type):
    agent_type("narrow", tools=["read_file", "run_code"])
    fake_model.script("[A]", fake_response("the report"))
    fake_model.script("[B]", fake_response("narrow report"))

    async def go():
        return await _agent(prompt="[A] x"), await _agent(prompt="[B] x", subagent_type="narrow")
    plain, narrow = asyncio.run(go())
    report, footer = plain.rsplit("\n\n", 1)
    assert report == "the report"
    assert re.fullmatch(r"agent_id: [0-9a-f]{6} \(send_message to continue\)", footer)
    assert footer.split()[1] in {r.id for r in agents.runs()}
    warning, body, footer = narrow.split("\n\n")
    assert warning.startswith("[warning]") and "run_code" in warning
    assert body == "narrow report" and footer.startswith("agent_id: ")


def test_agent_tool_unknown_type(agents, fake_model):
    result = asyncio.run(_agent(prompt="x", subagent_type="nope"))
    assert result.startswith("[error]") and "nope" in result and "general-purpose" in result
    result = asyncio.run(_agent(prompt="x", isolation="docker"))
    assert result.startswith("[error] Unknown isolation 'docker'")
    assert agents.runs() == []


def test_agent_tool_schema():
    from dostuff.tools.definitions import HARNESS_PARAMS, TOOL_MAP, TOOL_SCHEMAS
    params = TOOL_SCHEMAS["agent"]["function"]["parameters"]
    assert set(params["properties"]) == {"description", "prompt", "subagent_type", "model", "run_in_background",
                                         "isolation"}
    assert all(p.get("description") for p in params["properties"].values())
    assert set(params.get("required", [])) == {"description", "prompt"}
    assert "delegate_to_subagent" not in TOOL_SCHEMAS
    assert TOOL_MAP["agent"].kind == "read"
    assert not ({"session_id", "owner", "adapter"} & set(params["properties"])) and "owner" in HARNESS_PARAMS


def test_agent_tool_never_asks(agents, fake_model, agent_type, tmp_path, monkeypatch):
    from dostuff.agent.permissions import SessionPermissions
    from dostuff.agent.run_tool import run_tool
    monkeypatch.chdir(tmp_path)
    agent_type("writer", tools=["write_file"])
    write = fake_tool_call("write_file", {"path": "out.txt", "content": "hi"})
    fake_model.script("[A]", fake_response(tool_calls=[write]), fake_response("wrote it"))
    parent = FakeAdapter(auto="yes")

    async def go():
        return await run_tool("agent", {"description": "write", "prompt": "[A] x", "subagent_type": "writer"},
                              mcp_client=object(), session_id="s1", turn_id="t1", adapter=parent,  # type: ignore[arg-type]
                              permissions=SessionPermissions(mode="ask"))
    result = asyncio.run(go())
    assert result.startswith("wrote it")
    assert len(parent.asks) == 1 and parent.asks[0][0].startswith("[agent: write · writer] ")
    assert (tmp_path / "out.txt").read_text() == "hi"


def test_restricted_tools_d12(agents, fake_model, agent_type):
    from dostuff.agent.subagents import RESTRICTED_SUBAGENT_TOOLS
    d12 = {"agent", "send_message", "stop_agent", "delete_file", "save_memory", "save_episode", "todo_write",
           "read_settings", "edit_config", "set_mcp_server", "remove_mcp_server"}
    assert d12 <= RESTRICTED_SUBAGENT_TOOLS
    agent_type("greedy", tools=["read_file", *sorted(d12)])
    fake_model.script("[A]", fake_response("ok"))
    result = asyncio.run(_agent(prompt="[A] x", subagent_type="greedy"))
    [call] = fake_model.calls_for("[A]")
    assert {t["function"]["name"] for t in call["tools"]} == {"read_file"}
    warning = result.split("\n\n")[0]
    assert all(name in warning for name in d12)
    # the general-purpose default set never has them either
    fake_model.script("[B]", fake_response("ok"))
    asyncio.run(_agent(prompt="[B] x"))
    assert not d12 & {t["function"]["name"] for t in fake_model.calls_for("[B]")[0]["tools"]}


# --- T2.7: foreground agents run in parallel ------------------------------------------------------

def test_parallel_foreground_timing(agents, fake_model):
    import time

    from dostuff.agent.loop import loop
    calls = [fake_tool_call("agent", {"description": f"d{m}", "prompt": f"[{m}] x"}, f"c{m}") for m in "ABC"]
    fake_model.script("[MAIN]", fake_response(tool_calls=calls), fake_response("all done"))
    for m in "ABC":
        fake_model.script(f"[{m}]", FakeStep(fake_response(f"report {m}"), delay=0.5))
    working: list = [{"role": "user", "content": "[MAIN] go"}]
    tool_phase: dict = {}

    async def go():
        real = fake_model.acompletion

        async def timed(**kwargs):  # the main model's second call marks the end of the tool phase
            if "[MAIN]" in str(kwargs["messages"][1].get("content")) and fake_model.calls_for("[MAIN]"):
                tool_phase["end"] = time.monotonic()
            return await real(**kwargs)
        from dostuff.agent import call_agent as call_agent_module
        call_agent_module.acompletion = timed
        try:
            start = time.monotonic()
            result = await loop(session_id="s1", turn_id="t1", user_text="[MAIN] go", dynamic_system_instruction="",
                                mcp_client=object(), working_history=working, turn_type="interactive_loop",  # type: ignore[arg-type]
                                stream=False, adapter=FakeAdapter())
            return result, start
        finally:
            call_agent_module.acompletion = real
    result, start = asyncio.run(go())
    assert result == "all done"
    assert tool_phase["end"] - start < 1.0  # sequential would be >= 1.5 s
    tools = [s for s in working if s["role"] == "tool"]
    assert [s["tool_call_id"] for s in tools] == ["cA", "cB", "cC"]
    assert [s["content"].split("\n\n")[0] for s in tools] == ["report A", "report B", "report C"]


# --- T2.8: ESC and shell jobs ---------------------------------------------------------------------

def test_esc_stops_foreground_runs(agents, fake_model):
    import time

    from dostuff.agent.loop import CANCELLED_TURN, loop
    from dostuff.agent.run_tool import CANCELLED_RESULT
    from dostuff.agent.subagents.registry import STOP_GRACE_S
    calls = [fake_tool_call("agent", {"description": f"d{m}", "prompt": f"[{m}] x"}, f"c{m}") for m in "AB"]
    fake_model.script("[MAIN]", fake_response(tool_calls=calls))
    working: list = [{"role": "user", "content": "[MAIN] go"}]
    parent = FakeAdapter()

    async def go():
        for m in "AB":
            fake_model.gate(f"[{m}]", fake_response("never"))
        turn = asyncio.create_task(loop(
            session_id="s1", turn_id="t1", user_text="[MAIN] go", dynamic_system_instruction="",
            mcp_client=object(), working_history=working, turn_type="interactive_loop",  # type: ignore[arg-type]
            stream=False, adapter=parent))
        # both runs are inside their (never-ending) model call
        await _until(lambda: all(fake_model.calls_for(f"[{m}]") for m in "AB"))
        start = time.monotonic()
        parent.cancel()
        result = await turn
        await asyncio.gather(*(r.done for r in agents.runs()))
        return result, time.monotonic() - start
    result, elapsed = asyncio.run(go())
    assert result == CANCELLED_TURN
    assert [r.status for r in agents.runs()] == ["stopped", "stopped"]
    assert elapsed < STOP_GRACE_S + 1.0
    tools = [s for s in working if s["role"] == "tool"]
    assert [s["tool_call_id"] for s in tools] == ["cA", "cB"]
    assert all(s["content"] == CANCELLED_RESULT for s in tools), [s["content"] for s in tools]


def test_run_jobs_killed_on_end(agents, fake_model, monkeypatch, registry, _py, _wait_dead):
    from dostuff.agent import loop as loop_module
    from dostuff.agent.permissions import SessionPermissions
    monkeypatch.setattr(loop_module, "retry_delay", lambda *a: 0.0)
    sleeper = _py("import time; time.sleep(30)")
    perms = SessionPermissions(mode="auto")
    from dostuff.agent.subagents.definitions import BUILTIN_TYPES, AgentDef
    monkeypatch.setitem(BUILTIN_TYPES, "sh", AgentDef(name="sh", description="d", tools=("bash_command",)))

    def start_call(job_id):
        args = {"command": sleeper, "run_in_background": True, "background_id": job_id}
        return fake_response(tool_calls=[fake_tool_call("bash_command", args)])

    def check(run, job_id):
        [job] = [j for j in registry.jobs() if j.id == job_id]
        assert job.owner == f"s1/sub/{run.id}"
        assert registry.owner_label(job.owner) == f"subagent: {run.description}"
        assert _wait_dead(job.pid, 5) and job.status == "killed" and job.killed_by == "owner_ended"
        assert registry.drain_notices(job.owner) == [] and registry.drain_notices("s1") == []

    async def go():
        fake_model.script("[ok]", start_call("ok"), fake_response("done"))
        ok = await _start(agents, "[ok] x", agent_type="sh", description="ok run", permissions=perms)
        assert await ok.done == "done" and ok.status == "completed"
        check(ok, "ok")

        fake_model.script("[err]", start_call("err"), ValueError("model blew up"), fake_response("wrap-up"))
        err = await _start(agents, "[err] x", agent_type="sh", description="err run", permissions=perms)
        assert "model blew up" in await err.done and err.status == "failed"
        check(err, "err")

        fake_model.script("[stop]", start_call("stop"))
        fake_model.gate("[stop]", fake_response("never"))
        stopped = await _start(agents, "[stop] x", agent_type="sh", description="stop run", permissions=perms)
        await _until(lambda: any(j.id == "stop" for j in registry.jobs()))
        await asyncio.sleep(0.1)
        await agents.stop(stopped, "user")
        assert stopped.status == "stopped"
        check(stopped, "stop")
    asyncio.run(go())


def test_esc_before_first_model_call(agents, fake_model):
    """A run can see the ESC at its own first check, before the main loop cancels the call: same result."""
    from dostuff.agent.run_tool import CANCELLED_RESULT
    parent = FakeAdapter()

    async def go():
        parent.cancel()
        return await _agent(prompt="[A] x", adapter=parent)
    result = asyncio.run(go())
    assert result.split("\n\n")[0] == CANCELLED_RESULT and result.split("\n\n")[1].startswith("agent_id: ")
    assert [r.status for r in agents.runs()] == ["stopped"] and not fake_model.calls_for("[A]")


def test_manual_skill_not_in_subagent_prompt(tmp_path, monkeypatch):
    from dostuff.agent.subagents.definitions import load_definitions
    from dostuff.agent.subagents.prompt import _sections
    from dostuff.helpers.skills import discover_skills as ds
    for name, extra in (("visible", ""), ("manual", "disable-model-invocation: true\n")):
        folder = tmp_path / "skills" / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: the {name} one\n{extra}---\nbody\n",
                                         encoding="utf-8")
    monkeypatch.setattr(ds, "SKILLS_DIRS", [str(tmp_path / "skills")])
    text = _sections(load_definitions(tmp_path, tmp_path).types["general-purpose"], None)
    assert "the visible one" in text and "the manual one" not in text
