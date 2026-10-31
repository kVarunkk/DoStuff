"""bash_command / run_shell (plan/2026-10-31-background-shell/tasks.md).

Phase 1 pins today's foreground behaviour (CON-005) before run_shell moves onto the job registry;
these tests must pass unchanged on both implementations.
"""
import asyncio
import os
import time
from pathlib import Path

import pytest

from dostuff.tools.bash import bash_command as bash_module
from dostuff.tools.bash import jobs as jobs_module
from dostuff.tools.bash.bash_command import bash_command, run_shell


@pytest.fixture(autouse=True)
def _own_registry(registry, monkeypatch):
    """Every shell command is a job (D1): give each test a fresh registry under a temp jobs root."""
    monkeypatch.setattr(jobs_module, "_registry", registry)
    return registry


def run(coro):
    return asyncio.run(coro)


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n")


def _read_pids(path: Path, n: int, timeout: float = 5.0) -> list[int]:
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


def _kill_pid(pid: int) -> None:
    try:
        os.kill(pid, 9 if os.name != "nt" else 15)
    except OSError:
        pass


# A script that starts a grandchild sleeper, writes "<own pid> <grandchild pid>" and sleeps
_TREE = """
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
open(r"{pids}", "w").write(f"{{os.getpid()}} {{child.pid}}")
print("started", flush=True)
time.sleep(30)
"""


# --- T1.1 result format, exit codes, stderr, cwd ----------------------------------------------

def test_fg_result_format(tmp_path, monkeypatch, _py):
    monkeypatch.chdir(tmp_path)
    assert _norm(run(bash_command("echo hi"))) == "$ echo hi\nexit: 0\nstdout:\nhi\n\nstderr:\n"
    assert "\nexit: 3\n" in run(bash_command("exit 3"))

    cmd = _py("import sys; sys.stderr.write('oops\\n')")
    out = _norm(run(bash_command(cmd)))
    assert out == f"$ {cmd}\nexit: 0\nstdout:\n\nstderr:\noops\n"

    (tmp_path / "sub").mkdir()
    where = _py("import os; print(os.getcwd())")
    assert str(tmp_path / "sub") in run(bash_command(where, cwd="sub"))
    other = tmp_path / "other"
    other.mkdir()
    assert str(other) in run(bash_command(where, cwd=str(other)))

    bad = run(bash_command("echo hi", cwd="missing-dir"))
    assert bad.startswith("$ echo hi\nerror: ")

    monkeypatch.setattr(bash_module, "select_shell", lambda: ("bash", [str(tmp_path / "no-shell.exe"), "-c"]))
    assert run(bash_command("echo hi")).startswith("$ echo hi\nerror: ")


def test_fg_utf8_output(_py):
    cmd = _py("import sys; sys.stdout.buffer.write('héllo ✓\\n'.encode('utf-8'))")
    assert "stdout:\nhéllo ✓\n" in _norm(run(bash_command(cmd)))
    bad = _py("import sys; sys.stdout.buffer.write(b'a\\xffb\\n')")
    assert "stdout:\na�b\n" in _norm(run(bash_command(bad)))


def test_fg_timeout_clamp(monkeypatch):
    seen = []

    async def spy(command, cwd="", timeout=60, owner=""):
        seen.append(timeout)
        return 0, "", "", ""
    monkeypatch.setattr(bash_module, "run_shell", spy)
    for value in (0, "abc", 9999):
        run(bash_command("echo hi", timeout=value))  # type: ignore[arg-type]
    assert seen == [60, 60, 110]


# --- T1.2 timeout and cancellation kill the whole tree ----------------------------------------

def test_fg_timeout_kills_tree(tmp_path, _py, _wait_dead):
    pids_file = tmp_path / "pids"
    cmd = _py(_TREE.format(pids=pids_file))
    code, _, _, note = run(run_shell(cmd, timeout=1.5))
    pids = _read_pids(pids_file, 2)
    try:
        assert code is None
        assert note.startswith("Timed out after")
        assert all(_wait_dead(pid, 3) for pid in pids)
    finally:
        for pid in pids:
            _kill_pid(pid)


def test_fg_cancel_kills_tree(tmp_path, _py, _wait_dead):
    pids_file = tmp_path / "pids"
    cmd = _py(_TREE.format(pids=pids_file))

    async def scenario():
        task = asyncio.create_task(run_shell(cmd, timeout=10))
        deadline = time.monotonic() + 5
        while not pids_file.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    run(scenario())
    pids = _read_pids(pids_file, 2)
    try:
        assert all(_wait_dead(pid, 3) for pid in pids)
    finally:
        for pid in pids:
            _kill_pid(pid)


# --- T1.3 `cmd &` returns promptly -------------------------------------------------------------

def test_fg_background_ampersand_returns(tmp_path, _py):
    pid_file = tmp_path / "sleeper"
    cmd = _py(f"""
        import subprocess, sys
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], stdout=sys.stdout, stderr=sys.stderr)
        open(r"{pid_file}", "w").write(str(p.pid))
    """)
    started = time.monotonic()
    try:
        out = run(bash_command(cmd))
        assert time.monotonic() - started < 5
        assert "\nexit: 0\n" in out
    finally:
        for pid in _read_pids(pid_file, 1):
            _kill_pid(pid)


# --- T3.1 run_shell = start + wait + read ------------------------------------------------------

def test_run_shell_uses_registry(registry, monkeypatch, tmp_path, _py):
    calls = []
    real_start, real_wait = registry.start, registry.wait

    async def start(*args, **kwargs):
        calls.append(("start", kwargs.get("background")))
        return await real_start(*args, **kwargs)

    async def wait(job, timeout):
        calls.append(("wait", timeout))
        return await real_wait(job, timeout)
    monkeypatch.setattr(registry, "start", start)
    monkeypatch.setattr(registry, "wait", wait)
    flag = tmp_path / "flag"
    cmd = _py(f"""
        import pathlib, time
        print("hi", flush=True)
        while not pathlib.Path(r"{flag}").exists():
            time.sleep(0.05)
    """)

    async def scenario():
        task = asyncio.create_task(run_shell(cmd, timeout=10))
        deadline = time.monotonic() + 5
        while not registry.jobs() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        [job] = registry.jobs()
        assert job.status == "running" and job.foreground
        assert job.out_path.exists() and job.err_path.exists()
        flag.write_text("go")
        return job, await task
    job, (code, stdout, stderr, note) = run(scenario())
    assert (code, _norm(stdout), stderr, note) == (0, "hi\n", "", "")
    assert calls == [("start", False), ("wait", 10)]
    assert job.status == "exited" and job.exit_code == 0


def test_run_shell_no_drain_grace(tmp_path, _py):
    pid_file = tmp_path / "sleeper"
    cmd = _py(f"""
        import subprocess, sys
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], stdout=sys.stdout, stderr=sys.stderr)
        open(r"{pid_file}", "w").write(str(p.pid))
    """)
    started = time.monotonic()
    try:
        code, _, _, note = run(run_shell(cmd))
        assert time.monotonic() - started < 1.5
        assert code == 0
        assert "Output capture stopped" not in note
    finally:
        for pid in _read_pids(pid_file, 1):
            _kill_pid(pid)


# --- T4.1 bash_command(run_in_background=True) -------------------------------------------------

def test_bg_start_result(registry, _py, _alive):
    sleeper = _py("import time; time.sleep(30)")

    async def scenario():
        started = time.monotonic()
        out = await bash_command(sleeper, timeout=1, run_in_background=True, background_id="Dev", owner="s1")
        assert time.monotonic() - started < 1
        job = registry.get("s1", "dev")
        assert out == (
            f"$ {sleeper}\n"
            f'started background job "dev" (pid {job.pid})\n'
            f"stdout: {job.out_path}\n"
            f"stderr: {job.err_path}\n"
            'Use bash_output("dev") to read its output and kill_shell("dev") to stop it.'
        )
        await asyncio.sleep(2)  # timeout=1 is ignored for background jobs
        assert job.status == "running" and _alive(job.pid)

        # Registry refusals are error results, not exceptions
        again = await bash_command(sleeper, run_in_background=True, background_id="dev", owner="s1")
        assert again.startswith(f"$ {sleeper}\nerror: ") and "already running" in again
        bad = await bash_command(sleeper, run_in_background=True, background_id="a b", owner="s1")
        assert bad.startswith(f"$ {sleeper}\nerror: invalid background_id")
        missing = await bash_command("echo hi", cwd="no-such-dir", run_in_background=True, owner="s1")
        assert missing.startswith("$ echo hi\nerror: ")
        assert [j.id for j in registry.jobs("s1")] == ["dev"]

        # background_id is ignored for a foreground call
        fg = await bash_command("echo hi", background_id="dev", owner="s1")
        assert "\nexit: 0\n" in fg
        await registry.kill_all()
    run(scenario())


def test_bg_start_immediate_exit(registry, monkeypatch):
    # A shell builtin fails well inside 200 ms locally, but a slow CI runner can take longer to start
    # cmd.exe, so widen the window; a job that exits inside it still returns as soon as it exits
    monkeypatch.setattr(bash_module, "IMMEDIATE_EXIT_S", 5.0)
    cmd = "cd no-such-dir-for-this-test"

    async def scenario():
        out = _norm(await bash_command(cmd, run_in_background=True, owner="s1"))
        head, stderr = out.split("\nstderr:\n")
        assert head == f'$ {cmd}\nbackground job "bg-1" exited immediately with code 1'
        assert "no-such-dir-for-this-test" in stderr
        assert registry.drain_notices("s1") == []  # the start result already said so
    run(scenario())


def test_bg_owner_injected(registry, monkeypatch, _py):
    from dostuff.agent import run_tool as run_tool_module
    from dostuff.agent.permissions import PermissionConfig
    from dostuff.tools.definitions import TOOL_MAP, TOOL_SCHEMAS
    assert "owner" in TOOL_MAP["bash_command"].injected
    params = TOOL_SCHEMAS["bash_command"]["function"]["parameters"]["properties"]
    assert "owner" not in params and {"run_in_background", "background_id"} <= set(params)

    monkeypatch.setattr(run_tool_module, "permission_config", lambda: PermissionConfig(mode="auto"))
    sleeper = _py("import time; time.sleep(30)")

    async def scenario():
        args = {"command": sleeper, "run_in_background": True, "background_id": "w", "owner": "sneaky"}
        await run_tool_module.run_tool("bash_command", args, session_id="s9", owner="s9/sub/r1")
        await run_tool_module.run_tool("bash_command", dict(args, background_id="v"), session_id="s9")
        assert {(j.owner, j.id) for j in registry.jobs()} == {("s9/sub/r1", "w"), ("s9", "v")}
        await registry.kill_all()
    run(scenario())


# --- T4.2 bash_output ---------------------------------------------------------------------------

def test_bash_output_tool(registry, monkeypatch, tmp_path, _py):
    from dostuff.tools.bash.bash_output import bash_output
    flag = tmp_path / "flag"
    cmd = _py(f"""
        import pathlib, sys, time
        print("booting", flush=True)
        print("ERROR one", flush=True)
        while not pathlib.Path(r"{flag}").exists():
            time.sleep(0.05)
        print("ready on :3000", flush=True)
        sys.stderr.write("warn\\n")
        time.sleep(30)
    """)

    async def scenario():
        await bash_command(cmd, run_in_background=True, background_id="dev", owner="s1")
        job = registry.get("s1", "dev")
        out = _norm(await bash_output("dev", wait_for="ERROR", timeout=5, owner="s1"))
        assert out.startswith('job "dev": running (pid ')
        assert "wait_for: matched\nstdout (new):\n  booting\n  ERROR one\nstderr (new):\n  (none)" in out

        out = _norm(await bash_output("dev", wait_for="ready", timeout=1, owner="s1"))
        assert "wait_for: timed out after 1s\nstdout (new):\n  (none)\n" in out

        flag.write_text("go")
        out = _norm(await bash_output("dev", filter="ready", wait_for="ready", timeout=5, owner="s1"))
        assert "stdout (new):\n  ready on :3000\n" in out

        # cap: the latest output wins behind a skip header
        monkeypatch.setattr(jobs_module.JobRegistry, "_cap", staticmethod(lambda text, cap, path: f"[capped {cap}]"))
        from dostuff.config import get_config
        out = _norm(await bash_output("dev", owner="s1"))
        assert f"[capped {get_config().max_tool_output_chars}]" in out

        await registry.kill(job, by="model")
        out = _norm(await bash_output("dev", owner="s1"))
        assert out.startswith('job "dev": killed (by model)\n')

        quick = await bash_command("exit 3", run_in_background=True, background_id="q", owner="s1")
        assert "exited immediately with code 3" in quick
        assert _norm(await bash_output("q", owner="s1")).startswith('job "q": exited (code 3)\n')

        unknown = await bash_output("nope", owner="s1")
        assert unknown.startswith('error: no job "nope". Your jobs: "dev" (killed), "q" (exited)')
        assert (await bash_output("dev", owner="s2")).startswith('error: no job "dev". Your jobs: none')
        assert (await bash_output("dev", filter="(", owner="s1")).startswith("error: invalid filter regex")
    run(scenario())


# --- T4.3 kill_shell ----------------------------------------------------------------------------

def test_kill_shell_tool(registry, _py, _wait_dead):
    from dostuff.tools.bash.kill_shell import kill_shell
    cmd = _py("""
        import sys, time
        for i in range(30):
            print(f"line {i}", flush=True)
        sys.stderr.write("oops\\n"); sys.stderr.flush()
        time.sleep(30)
    """)

    async def scenario():
        await bash_command(cmd, run_in_background=True, background_id="dev", owner="s1")
        job = registry.get("s1", "dev")
        await registry.wait_for(job, "oops", 5)
        out = _norm(await kill_shell("dev", owner="s1"))
        assert _wait_dead(job.pid, 5)
        lines = "\n".join(f"  line {i}" for i in range(10, 30))
        assert out == f'job "dev": killed (by model)\nstdout (last 20 lines):\n{lines}\nstderr (last 20 lines):\n  oops'
        assert _norm(await kill_shell("dev", owner="s1")) == out  # idempotent
        assert registry.drain_notices("s1") == []
        assert (await kill_shell("nope", owner="s1")).startswith('error: no job "nope". Your jobs: "dev" (killed)')
    run(scenario())


# --- T7.1: the prompt and the timeout note point at background jobs, not nohup ---------------------

def test_no_nohup_advice(tmp_path, _py):
    from dostuff.helpers.agent.constants import SYSTEM_INSTRUCTIONS
    root = Path(bash_module.__file__).parents[2]
    files = [p for p in root.rglob("*") if p.suffix in (".py", ".yaml", ".md", ".example") and p.is_file()
             and "__pycache__" not in p.parts]
    assert files and not [p for p in files if "nohup" in p.read_text(encoding="utf-8", errors="replace")]
    for name in ("run_in_background", "background_id", "bash_output", "wait_for", "kill_shell", "<background-job>"):
        assert name in SYSTEM_INSTRUCTIONS, name
    sleeper = _py("import time; time.sleep(30)")
    _, _, _, note = asyncio.run(run_shell(sleeper, "", 0.5))
    assert note.startswith("Timed out after") and "run_in_background=true" in note
