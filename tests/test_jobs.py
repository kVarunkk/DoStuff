"""Shell job registry (plan/2026-10-31-background-shell/spec.md WS-1, tasks.md Phase 2)."""
import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from dostuff.tools.bash import jobs as jobs_module
from dostuff.tools.bash.jobs import POLL_S, JobError, format_notice, find_orphans, prune_logs


def run(coro):
    return asyncio.run(coro)


async def until(cond, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(POLL_S / 4)


def _read_pids(path: Path, n: int = 1, timeout: float = 5.0) -> list[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            pids = [int(p) for p in path.read_text().split()]
            if len(pids) >= n:
                return pids
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise AssertionError(f"pid file {path} never got {n} pids")


_SLEEP = "import time; time.sleep(30)"
_TREE = """
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
open(r"{pids}", "w").write(f"{{os.getpid()}} {{child.pid}}")
time.sleep(30)
"""


# --- T0.1 --------------------------------------------------------------------------------------

def test_imports():
    from dostuff.tools.bash.jobs import Job, JobRegistry, get_registry
    assert isinstance(get_registry(), JobRegistry)
    assert get_registry() is get_registry()
    assert "uid" in Job.__dataclass_fields__


# --- T2.1 start: files, not pipes --------------------------------------------------------------

def test_start_writes_separate_files(registry, jobs_home, _py):
    cmd = _py("import sys; sys.stdout.buffer.write(b'out\\n'); sys.stderr.buffer.write(b'err\\n')")

    async def scenario():
        job = await registry.start(cmd, "", "s1", background=True, job_id="")
        assert job.status == "running" and job.pid > 0 and job.uid
        assert job.out_path == jobs_home / "s1" / f"{job.uid}.out"
        assert job.err_path == jobs_home / "s1" / f"{job.uid}.err"
        await registry.wait(job, 10)
        return job
    job = run(scenario())
    assert job.out_path.read_bytes() == b"out\n"
    assert job.err_path.read_bytes() == b"err\n"
    moved = job.out_path.with_suffix(".moved")
    os.replace(job.out_path, moved)  # no handle left open in the parent (would fail on Windows)
    job.err_path.unlink()


def test_start_spawn_failure_registers_nothing(registry, jobs_home, tmp_path, monkeypatch):
    from dostuff.tools.bash import bash_command as bash_module

    async def scenario():
        with pytest.raises(JobError, match="working directory"):
            await registry.start("echo hi", "no-such-dir", "s1", background=True)
        monkeypatch.setattr(bash_module, "select_shell", lambda: ("bash", [str(tmp_path / "nope.exe"), "-c"]))
        with pytest.raises(OSError):
            await registry.start("echo hi", "", "s1", background=True)
    run(scenario())
    assert registry.jobs() == []
    assert not list(jobs_home.glob("*/*.out")) and not list(jobs_home.glob("*/*.err"))


# --- T2.2 watcher records exit and jobs.json ---------------------------------------------------

def test_watcher_records_exit(registry, jobs_home, _py):
    async def scenario():
        quick = await registry.start("exit 3", "", "s1", background=True)
        slow = await registry.start(_py(_SLEEP), "", "s1", background=True, job_id="slow")
        await until(lambda: quick.status == "exited", 2)
        assert quick.exit_code == 3 and quick.ended is not None
        index = json.loads((jobs_home / "s1" / "jobs.json").read_text(encoding="utf-8"))
        by_id = {e["id"]: e for e in index}
        assert {"id", "pid", "command", "started"} <= by_id[quick.id].keys()
        assert by_id[quick.id]["ended"] is not None and by_id[quick.id]["command"] == "exit 3"
        assert by_id["slow"]["ended"] is None and by_id["slow"]["pid"] == slow.pid
        await registry.kill_all()
    run(scenario())


# --- T2.3 ids, collisions, reuse, limit --------------------------------------------------------

def test_job_ids(registry):
    async def scenario():
        a = await registry.start("exit 0", "", "s1", background=True)
        b = await registry.start("exit 0", "", "s1", background=True)
        c = await registry.start("exit 0", "", "s2", background=True)
        d = await registry.start("exit 0", "", "s1", background=True, job_id="Dev")
        fg = await registry.start("exit 0", "", "s1", background=False)
        assert (a.id, b.id, c.id, d.id, fg.id) == ("bg-1", "bg-2", "bg-1", "dev", "bg-3")
        for bad in ("-x", "a b", "x" * 33):
            with pytest.raises(ValueError, match=r"\[a-z0-9\]"):
                await registry.start("exit 0", "", "s1", background=True, job_id=bad)
        await until(lambda: all(j.ended for j in registry.jobs()))
        e = await registry.start("exit 0", "", "s1", background=True)
        assert e.id == "bg-4"  # ended ids are never handed out again
        await registry.wait(e, 5)
    run(scenario())


def test_job_id_collision_and_reuse(registry, _py):
    async def scenario():
        first = await registry.start(_py(_SLEEP), "", "s1", background=True, job_id="dev")
        with pytest.raises(JobError) as err:
            await registry.start("echo again", "", "s1", background=True, job_id="dev")
        msg = str(err.value)
        assert f"pid {first.pid}" in msg and "already running" in msg
        assert "kill it with kill_shell or pick another background_id" in msg
        assert first.status == "running" and registry.get("s1", "dev") is first
        await registry.kill(first, by="model")
        second = await registry.start("exit 0", "", "s1", background=True, job_id="dev")
        assert second.uid != first.uid and registry.get("s1", "dev") is second
        assert [j.id for j in registry.jobs("s1")] == ["dev"]
        await registry.wait(second, 5)
    run(scenario())


def test_job_limit(registry, _py):
    async def scenario():
        cmd = _py(_SLEEP)
        started = [await registry.start(cmd, "", "s1", background=True) for _ in range(8)]
        with pytest.raises(JobError) as err:
            await registry.start(cmd, "", "s1", background=True)
        assert all(f'"{j.id}"' in str(err.value) for j in started)
        await registry.start(cmd, "", "s1", background=False)
        await registry.start(cmd, "", "s2", background=True)
        await registry.kill_all()
    run(scenario())


# --- T2.4 read_new -----------------------------------------------------------------------------

def test_read_new_incremental(registry, tmp_path, _py):
    flag = tmp_path / "flag"
    cmd = _py(f"""
        import os, sys, time
        sys.stdout.write("a\\n"); sys.stdout.flush()
        sys.stderr.write("e1\\n"); sys.stderr.flush()
        while not os.path.exists(r"{flag}"):
            time.sleep(0.02)
        sys.stdout.write("b\\n"); sys.stdout.flush()
    """)

    async def scenario():
        job = await registry.start(cmd, "", "s1", background=True)
        await until(lambda: job.out_path.stat().st_size > 0 and job.err_path.stat().st_size > 0)
        first = registry.read_new(job)
        assert (first.stdout, first.stderr) == ("a\n", "e1\n")
        second = registry.read_new(job)
        assert (second.stdout, second.stderr) == ("", "")
        flag.write_text("go")
        await registry.wait(job, 5)
        third = registry.read_new(job)
        assert (third.stdout, third.stderr) == ("b\n", "")
    run(scenario())


def test_read_new_split_multibyte(registry, tmp_path, _py):
    cmd = _py("""
        import sys, time
        for byte in "✓".encode("utf-8"):
            sys.stdout.buffer.write(bytes([byte])); sys.stdout.flush()
            time.sleep(0.15)
    """)

    async def scenario():
        job = await registry.start(cmd, "", "s1", background=True)
        parts = []
        while job.ended is None:
            parts.append(registry.read_new(job).stdout)
            assert "�" not in parts[-1]
            await asyncio.sleep(0.05)
        parts.append(registry.read_new(job).stdout)
        assert "".join(parts) == "✓"
    run(scenario())


def test_read_new_filter_consumes(registry, _py):
    cmd = _py("print('ok 1'); print('ERROR one'); print('ok 2'); print('ERROR two')")

    async def scenario():
        job = await registry.start(cmd, "", "s1", background=True)
        await registry.wait(job, 5)
        with pytest.raises(JobError, match="regex"):
            registry.read_new(job, filter="(")
        assert job.out_offset == 0
        assert registry.read_new(job, filter="ERROR").stdout == "ERROR one\nERROR two\n"
        assert registry.read_new(job).stdout == ""
    run(scenario())


def test_read_new_cap(registry, _py):
    cap = 200
    cmd = _py(f"""
        import sys
        for i in range({cap // 10 * 10}):
            sys.stdout.buffer.write(b"line %05d\\n" % i)
    """)

    async def scenario():
        job = await registry.start(cmd, "", "s1", background=True)
        await registry.wait(job, 5)
        total = job.out_path.stat().st_size
        text = registry.read_new(job, cap=cap).stdout
        assert len(text) <= cap
        assert text.endswith(f"line {cap // 10 * 10 - 1:05d}\n")
        header, rest = text.split("\n", 1)
        assert header == f"[… {total - len(rest.encode())} earlier bytes skipped; full log: {job.out_path}]"
    run(scenario())


# --- T2.5 wait_for -----------------------------------------------------------------------------

def test_wait_for_outcomes(registry, monkeypatch, _py):
    async def scenario():
        ready = await registry.start(_py("import time; time.sleep(0.5); print('ready', flush=True); time.sleep(30)"),
                                     "", "s1", background=True)
        res = await registry.wait_for(ready, "ready", 5)
        assert res.reason == "matched" and "ready" in res.output.stdout
        assert ready.status == "running"

        quits = await registry.start(_py("print('bye')"), "", "s1", background=True)
        res = await registry.wait_for(quits, "ready", 5)
        assert res.reason == "exited" and res.output.stdout == "bye\n"

        quiet = await registry.start(_py(_SLEEP), "", "s1", background=True)
        started = time.monotonic()
        res = await registry.wait_for(quiet, "ready", 1)
        assert res.reason == "timeout" and 0.9 <= time.monotonic() - started < 3
        await registry.kill_all()
    run(scenario())
    assert jobs_module.clamp_wait_timeout(0) == 1
    assert jobs_module.clamp_wait_timeout(500) == 110
    assert jobs_module.clamp_wait_timeout("x") == 30


def test_wait_for_cancel_keeps_job(registry, _py, _alive):
    cmd = _py("import time; print('noise', flush=True); time.sleep(30)")

    async def scenario():
        job = await registry.start(cmd, "", "s1", background=True)
        await until(lambda: job.out_path.stat().st_size > 0)
        task = asyncio.create_task(registry.wait_for(job, "ready", 10))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert job.status == "running" and _alive(job.pid)
        assert (job.out_offset, job.err_offset) == (0, 0)
        await registry.kill_all()
    run(scenario())


# --- T2.6 kill, kill_owner, kill_all -----------------------------------------------------------

def test_kill_tree(registry, tmp_path, _py, _wait_dead):
    pids_file = tmp_path / "pids"

    async def scenario():
        job = await registry.start(_py(_TREE.format(pids=pids_file)), "", "s1", background=True)
        pids = await asyncio.to_thread(_read_pids, pids_file, 2)
        assert await registry.kill(job, by="model") == "killed"
        assert job.killed_by == "model"
        assert await registry.kill(job, by="user") == "killed" and job.killed_by == "model"
        return pids
    pids = run(scenario())
    assert all(_wait_dead(pid, 5) for pid in pids)


def test_kill_owner_and_all(registry, jobs_home, _py, _alive, _wait_dead):
    async def scenario():
        cmd = _py(_SLEEP)
        a = await registry.start(cmd, "", "s1", background=True)
        sub = await registry.start(cmd, "", "s1/sub/r1", background=True)
        b = await registry.start(cmd, "", "s2", background=True)
        killed = await registry.kill_owner("s1/sub/r1", "owner_ended")
        assert killed == [sub] and sub.status == "killed" and sub.killed_by == "owner_ended"
        assert a.status == b.status == "running"
        await registry.kill_all()
        return [a, sub, b]
    jobs = run(scenario())
    assert all(j.status == "killed" for j in jobs)
    assert all(_wait_dead(j.pid, 5) for j in jobs)
    for session in ("s1", "s2"):
        index = json.loads((jobs_home / session / "jobs.json").read_text(encoding="utf-8"))
        assert all(e["ended"] is not None for e in index)


def test_transfer(registry, jobs_home, _py, _alive):
    """/clear and /resume hand running jobs and queued notices to the new session (D8)."""
    async def scenario():
        cmd = _py(_SLEEP)
        dev = await registry.start(cmd, "", "s1", background=True, job_id="dev")
        auto = await registry.start(cmd, "", "s1", background=True)
        done = await registry.start(_py("pass"), "", "s1", background=True, job_id="done")
        await registry.wait(done, 10)
        sub = await registry.start(cmd, "", "s1/sub/r1", background=True, job_id="sub")
        moved = registry.transfer("s1", "s2")
        assert moved == [dev, auto] and dev.owner == auto.owner == "s2"
        assert registry.get("s2", "dev") is dev and _alive(dev.pid)
        assert [j.id for j in registry.jobs("s1")] == ["done"] and sub.owner == "s1/sub/r1"
        assert [n.id for n in registry.drain_notices("s2")] == ["done"] and registry.drain_notices("s1") == []
        nxt = await registry.start(cmd, "", "s2", background=True)
        assert nxt.id == "bg-2"  # bg-1 travelled with the jobs
        index = {s: json.loads((jobs_home / s / "jobs.json").read_text(encoding="utf-8")) for s in ("s1", "s2")}
        assert {e["id"] for e in index["s1"]} == {"done", "sub"}  # the moved ones aren't orphans of s1
        assert {e["id"] for e in index["s2"]} == {"dev", "bg-1", "bg-2"}
        assert registry.transfer("s1", "s3") == []
        await registry.kill_all()
    run(scenario())


def test_kill_all_sync(registry, _py, _wait_dead):
    async def scenario():
        return await registry.start(_py(_SLEEP), "", "s1", background=True)
    job = run(scenario())
    registry.kill_all_sync()
    assert job.status == "killed"
    assert _wait_dead(job.pid, 5)


# --- T2.7 owner isolation ----------------------------------------------------------------------

def test_owner_isolation(registry, _py):
    async def scenario():
        cmd = _py(_SLEEP)
        await registry.start(cmd, "", "s1", background=True, job_id="dev")
        await registry.start(cmd, "", "s1/sub/r1", background=True, job_id="mine")
        with pytest.raises(JobError) as err:
            registry.get("s1/sub/r1", "dev")
        assert '"mine"' in str(err.value) and '"dev"' not in str(err.value).split("Your jobs:")[1]
        assert [j.id for j in registry.jobs("s1")] == ["dev"]
        assert {j.id for j in registry.jobs()} == {"dev", "mine"}
        await registry.kill_all()
    run(scenario())


# --- T2.8 notices ------------------------------------------------------------------------------

@pytest.mark.parametrize("how, expect", [
    ("exits", True),
    ("killed_by_user", True),
    ("killed_by_model", False),
    ("owner_ended", False),
    ("foreground", False),
])
def test_notices(registry, _py, how, expect):
    async def scenario():
        if how == "exits":
            job = await registry.start(_py("import sys; print('out'); sys.exit('boom')"), "", "s1", background=True)
            await registry.wait(job, 5)
        elif how == "foreground":
            job = await registry.start("exit 1", "", "s1", background=False)
            await registry.wait(job, 5)
        else:
            job = await registry.start(_py(_SLEEP), "", "s1", background=True)
            if how == "owner_ended":
                await registry.kill_owner("s1", "owner_ended")
            else:
                await registry.kill(job, by="user" if how == "killed_by_user" else "model")
        await asyncio.sleep(POLL_S)
        notices = registry.drain_notices("s1")
        assert registry.drain_notices("s1") == []
        return job, notices
    job, notices = run(scenario())
    assert len(notices) == (1 if expect else 0)
    if how == "exits":
        n = notices[0]
        assert (n.id, n.status, n.code) == (job.id, "exited", 1)
        assert n.stdout_tail == ["out"] and n.stderr_tail == ["boom"]
        text = format_notice(notices)
        assert text.startswith(f'<background-job id="{job.id}" status="exited" code="1" runtime="')
        assert "stderr (last 1 lines):\n  boom" in text and text.endswith("</background-job>")
        assert len(text) <= jobs_module.NOTICE_MAX_CHARS + 200
    if how == "killed_by_user":
        assert notices[0].status == "killed" and notices[0].killed_by == "user"


def test_notice_tail_caps(registry, _py):
    cmd = _py("import sys\nfor i in range(500): print('x' * 300)\nsys.exit(2)")

    async def scenario():
        job = await registry.start(cmd, "", "s1", background=True)
        await registry.wait(job, 5)
        return registry.drain_notices("s1")
    notices = run(scenario())
    assert len(notices[0].stdout_tail) == 20
    text = format_notice(notices)
    body = text.split("\n", 1)[1].rsplit("Full output", 1)[0]
    assert len(body) <= jobs_module.NOTICE_MAX_CHARS + 1
    two = format_notice(notices * 2)
    assert two.count("<background-job ") == 2


def test_notice_after_detach(registry, _py):
    async def scenario():
        job = await registry.start(_py("import time; time.sleep(0.5)"), "", "s1", background=False)
        registry.detach(job)
        await until(lambda: job.ended is not None)
        return registry.drain_notices("s1")
    assert len(run(scenario())) == 1


# --- T2.9 wait and detach ----------------------------------------------------------------------

def test_wait_exit_or_detach(registry, _py, _alive):
    async def scenario():
        quick = await registry.start("exit 0", "", "s1", background=False)
        assert await registry.wait(quick, 5) == "exited"

        sleeper = await registry.start(_py(_SLEEP), "", "s1", background=False)
        waiter = asyncio.create_task(registry.wait(sleeper, 5))
        await asyncio.sleep(0.2)
        started = time.monotonic()
        registry.detach(sleeper)
        assert await waiter == "detached" and time.monotonic() - started < 0.5
        assert sleeper.foreground is False and _alive(sleeper.pid) and sleeper.status == "running"

        other = await registry.start(_py(_SLEEP), "", "s1", background=False)
        assert await registry.wait(other, 0.3) == "timeout"
        assert other.status == "running"  # the caller kills on timeout, not the registry

        registry.detach(quick)  # ended: no-op
        assert quick.foreground is True
        await registry.kill_all()
    run(scenario())


def test_detach_foreground(registry, _py):
    async def scenario():
        cmd = _py(_SLEEP)
        main = await registry.start(cmd, "", "s1", background=False)
        sub = await registry.start(cmd, "", "s1/sub/r1", background=False)
        user = await registry.start(cmd, "", "user", background=False)
        other = await registry.start(cmd, "", "s2", background=False)
        bg = await registry.start(cmd, "", "s1", background=True)
        assert {j.uid for j in registry.detach_foreground("s1")} == {main.uid, sub.uid}
        assert user.foreground and other.foreground and not bg.foreground
        await registry.kill_all()
    run(scenario())


# --- T2.10 startup pruning and orphan detection ------------------------------------------------

def test_prune_old_logs(jobs_home):
    old, new = jobs_home / "old-session", jobs_home / "new-session"
    for d in (old, new):
        d.mkdir(parents=True)
        (d / "x.out").write_text("log")
    eight_days = time.time() - 8 * 86400
    os.utime(old, (eight_days, eight_days))
    held = open(new / "x.out", "rb")
    try:
        assert prune_logs(days=7) == [old]
    finally:
        held.close()
    assert not old.exists() and new.exists()


def test_orphan_detection(jobs_home):
    def index(session, entries):
        d = jobs_home / session
        d.mkdir(parents=True)
        (d / "jobs.json").write_text(json.dumps(entries), encoding="utf-8")

    index("earlier", [
        {"id": "dev", "pid": 4812, "command": "npm run dev", "started": 1.0, "ended": None},
        {"id": "bg-1", "pid": 4813, "command": "exit 0", "started": 1.0, "ended": 2.0},
    ])
    index("ended", [{"id": "bg-1", "pid": 1, "command": "x", "started": 1.0, "ended": 2.0}])
    index("current", [{"id": "dev", "pid": 7, "command": "mine", "started": 1.0, "ended": None}])
    (jobs_home / "corrupt").mkdir()
    (jobs_home / "corrupt" / "jobs.json").write_text("{not json", encoding="utf-8")
    assert find_orphans("current") == [("npm run dev", 4812)]
    assert set(find_orphans("nothing-here-yet")) == {("npm run dev", 4812), ("mine", 7)}


# --- Phase 5: loop and lifecycle ---------------------------------------------------------------

def _msg(content=None, tool_calls=None):
    from types import SimpleNamespace
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))], usage=None,
    )


def _tc(name, args, call_id):
    from types import SimpleNamespace
    args = json.dumps(args)
    tc = SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=args))
    tc.model_dump = lambda: {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}
    return tc


@pytest.fixture
def scripted(monkeypatch, registry):
    """`scripted(responses)`: loop()'s call_agent replaced by a list of responses (or callables
    getting the messages); returns the list of message lists the model was called with."""
    from dostuff.agent import loop as loop_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    monkeypatch.setattr(loop_module, "get_model_token_limit", lambda: 100_000)
    calls: list = []

    def install(responses):
        async def fake_call_agent(steps_history, system_instruction, tool_names=None, stream=False):
            calls.append([dict(s) for s in steps_history])
            item = responses.pop(0)
            if callable(item):
                item = await item(steps_history)
            return item, None
        monkeypatch.setattr(loop_module, "call_agent", fake_call_agent)
        return calls
    return install


def _main_loop(working=None, session_id="s1", **kw):
    from dostuff.agent.loop import loop
    from dostuff.agent.permissions import SessionPermissions
    working = working if working is not None else []
    result = asyncio.run(loop(
        session_id=session_id, turn_id="t", user_text="hi", dynamic_system_instruction="sys", mcp_client=None,
        working_history=working, turn_type="interactive_loop", stream=False, max_iterations=6,
        permissions=SessionPermissions(mode="auto"), **kw,
    ))
    return result, working


def _bg(cmd, job_id, call_id="c1"):
    return _tc("bash_command", {"command": cmd, "run_in_background": True, "background_id": job_id}, call_id)


# --- T5.1 owner token per loop() run -----------------------------------------------------------

def test_loop_owner_token(registry, scripted, _py):
    sleeper = _py(_SLEEP)
    scripted([_msg(tool_calls=[_bg(sleeper, "srv")]), _msg("done")])
    _main_loop()
    assert [(j.owner, j.id) for j in registry.jobs()] == [("s1", "srv")]

    from dostuff.agent.permissions import SessionPermissions
    from dostuff.tools.delegate_to_subagent import _impl_delegate_to_subagent
    scripted([_msg(tool_calls=[_bg(sleeper, "sub")]), _msg("sub done"),
              _msg(tool_calls=[_bg(sleeper, "sub")]), _msg("sub done")])

    async def delegate():
        return await _impl_delegate_to_subagent(
            task="t", role="r", mcp_client=None, session_id="s1", turn_id="t",  # type: ignore[arg-type]
            requested_tools=["bash_command"], permissions=SessionPermissions(mode="auto"),
        )
    for _ in range(2):
        assert "sub done" in asyncio.run(delegate())
    owners = [j.owner for j in registry.jobs() if j.id == "sub"]
    assert len(owners) == 2 and len(set(owners)) == 2
    assert all(o.startswith("s1/sub/") and len(o) > len("s1/sub/") for o in owners)


# --- T5.2 notices drained before every call_agent ----------------------------------------------

class _Adapter:
    def __init__(self):
        self.events: list = []
        self._cancelled = False
        self._cancel_event = asyncio.Event()

    def emit(self, event_type, data):
        self.events.append((event_type, data))

    async def ask(self, message, **kwargs):
        return "yes"


async def _turn(working, session_id="s1", **kw):
    from dostuff.agent.loop import loop
    from dostuff.agent.permissions import SessionPermissions
    kw.setdefault("turn_type", "interactive_loop")
    return await loop(
        session_id=session_id, turn_id="t", user_text="hi", dynamic_system_instruction="sys", mcp_client=None,
        working_history=working, stream=False, max_iterations=6, permissions=SessionPermissions(mode="auto"), **kw,
    )


# exits after 0.6 s: past bash_command's 200 ms "exited immediately" window, so it ends as a notice
_FAIL_LATER = "import sys, time; time.sleep(0.6); sys.stderr.write('boom' + chr(10)); sys.exit(1)"
_SLOW = "import time; time.sleep(1.5)"


def _notice_steps(messages):
    return [m for m in messages if m["role"] == "user" and "<background-job" in str(m["content"])]


def test_loop_notice_reaches_model(registry, scripted, _py):
    fail_a, fail_b, slow = _py(_FAIL_LATER), _py(_FAIL_LATER), _py(_SLOW)
    calls = scripted([
        _msg(tool_calls=[_bg(fail_a, "a", "c1"), _bg(fail_b, "b", "c2")]),
        _msg(tool_calls=[_tc("bash_command", {"command": slow}, "c3")]),
        _msg("done"),
    ])
    working: list = [{"role": "user", "content": "hi"}]
    asyncio.run(_turn(working))
    # iteration 3's call: both notices, batched into one user step, as its last message
    assert _notice_steps(calls[1]) == []
    [step] = _notice_steps(calls[2])
    assert calls[2][-1] is not None and calls[2][-1]["content"] == step["content"]
    assert step["content"].count("<background-job ") == 2
    assert '<background-job id="a" status="exited" code="1"' in step["content"]
    assert "stderr (last 1 lines):\n  boom" in step["content"]
    assert _notice_steps(working) == [step]  # stored in history


def test_loop_notice_while_idle(registry, scripted, _py):
    fail = _py(_FAIL_LATER)
    calls = scripted([_msg(tool_calls=[_bg(fail, "test")]), _msg("started"), _msg("seen")])
    working: list = [{"role": "user", "content": "hi"}]

    async def scenario():
        await _turn(working)
        job = registry.get("s1", "test")
        await registry.wait(job, 5)
        working.append({"role": "user", "content": "next"})
        await _turn(working)
    asyncio.run(scenario())
    # the next turn's first model call sees the notice, after the user's message
    assert [m["content"] for m in calls[2][-2:-1]] == ["next"]
    assert '<background-job id="test" status="exited" code="1"' in calls[2][-1]["content"]


def test_loop_notice_owner_scoped(registry, scripted, _py):
    fail = _py(_FAIL_LATER)

    async def scenario():
        sub = await registry.start(fail, "", "s1/sub/r1", background=True, job_id="sub")
        main = await registry.start(fail, "", "s1", background=True, job_id="main")
        await registry.wait(sub, 5)
        await registry.wait(main, 5)
        scripted([_msg("main done"), _msg("sub done")])
        main_history: list = []
        await _turn(main_history)
        sub_history: list = []
        await _turn(sub_history, turn_type="subagent_loop", owner="s1/sub/r2")
        return main_history, sub_history
    main_history, sub_history = asyncio.run(scenario())
    assert ['id="main"' in s["content"] for s in _notice_steps(main_history)] == [True]
    assert _notice_steps(sub_history) == []  # another run's queue
    assert [n.id for n in registry.drain_notices("s1/sub/r1")] == ["sub"]


def test_loop_job_event(registry, scripted, _py):
    sleeper, fail, slow = _py(_SLEEP), _py(_FAIL_LATER), _py(_SLOW)
    scripted([
        _msg(tool_calls=[_bg(sleeper, "a", "c1")]),
        _msg(tool_calls=[_tc("kill_shell", {"background_id": "a"}, "c2")]),
        _msg(tool_calls=[_bg(fail, "b", "c3")]),
        _msg(tool_calls=[_tc("bash_command", {"command": slow}, "c4")]),
        _msg("done"),
    ])
    adapter = _Adapter()
    asyncio.run(_turn([], adapter=adapter))
    jobs = [(d["id"], d["status"], d["code"]) for e, d in adapter.events if e == "job"]
    assert jobs[:2] == [("a", "running", None), ("a", "killed", jobs[1][2])]
    assert jobs[2:] == [("b", "running", None), ("b", "exited", 1)]  # foreground commands emit nothing


# --- T5.6 atexit fallback -----------------------------------------------------------------------

def test_atexit_kills_jobs(tmp_path, _py, _wait_dead):
    import subprocess
    import sys
    pid_file = tmp_path / "sleeper.pid"
    sleeper = _py(f"""
        import os, time
        open(r"{pid_file}", "w").write(str(os.getpid()))
        time.sleep(30)
    """)
    child = tmp_path / "child.py"
    child.write_text(
        "import asyncio, pathlib, time\n"
        "from dostuff.tools.bash.jobs import get_registry\n"
        "async def main():\n"
        f"    await get_registry().start({sleeper!r}, '', 'atexit-test', background=True)\n"
        f"    while not pathlib.Path({str(pid_file)!r}).exists():\n"
        "        await asyncio.sleep(0.05)\n"
        "asyncio.run(main())\n",  # then exits normally, without any cleanup of its own
        encoding="utf-8",
    )
    home = tmp_path / "home"
    home.mkdir()
    repo = Path(jobs_module.__file__).resolve().parents[3]  # this checkout's dostuff, not an installed one
    env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home),  # not the real ~/.dostuff
           "PYTHONPATH": os.pathsep.join(filter(None, [str(repo), os.environ.get("PYTHONPATH")]))}
    subprocess.run([sys.executable, str(child)], check=True, timeout=30, env=env)
    pid = int(pid_file.read_text())
    try:
        assert _wait_dead(pid, 5)
    finally:
        if not _wait_dead(pid, 0):
            os.kill(pid, 9 if os.name != "nt" else 15)


# --- T7.2: README, config example and the parity plan document background jobs ---------------------

def test_readme_documents_jobs():
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    assert "\n## Background jobs\n" in readme
    start = readme.index("\n## Background jobs\n")
    section = readme[start:readme.index("\n## ", start + 1)]
    for phrase in ("run_in_background", "background_id", "bash_output", "wait_for", "kill_shell", "/jobs",
                   "/jobs kill", "Ctrl+B", "<background-job", "~/.dostuff/jobs/", "jobs.json"):
        assert phrase in section, phrase
    for row in ("| Subagent run ends", "| `/compact`", "| `/clear`", "| `/resume`", "| Quit", "| Headless"):
        assert row in section, row
    assert "killed hard" in section and "still be running" in section  # the crash caveat (D11)
    assert "**Background jobs**" in readme[readme.index("## Features"):readme.index("## Install")]
    controls = readme[readme.index("## TUI controls"):readme.index("## Tools and permissions")]
    assert "| `Ctrl+B`" in controls and "| `/jobs" in controls and "**Shells**" in controls
    example = (repo / "dostuff" / "templates" / "config.example.yaml").read_text(encoding="utf-8")
    status_comment = example[example.index("# All items:"):example.index("status_bar:")]
    assert "background jobs" in status_comment
    assert "status_bar: [cwd, model, mode, context]" in example
    parity = repo / "plan" / "2026-10-28-claude-code-parity" / "plan.md"  # plan/ is gitignored: local only
    if parity.exists():
        text = parity.read_text(encoding="utf-8")
        phase2 = text[text.index("### Phase 2"):text.index("\n### ", text.index("### Phase 2") + 1)]
        assert "2026-10-31-background-shell/spec.md" in phase2
