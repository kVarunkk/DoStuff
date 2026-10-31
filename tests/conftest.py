"""Shared fixtures."""
import itertools
import sys
import textwrap
from pathlib import Path

import pytest

_hook_ids = itertools.count()


@pytest.fixture
def hook_cmd(tmp_path):
    """`hook_cmd(body)`: write a hook script and return a shell command running it with this interpreter.

    Same as `_script` in tests/test_hooks.py, for the TUI and headless tests that configure hooks.
    """
    from dostuff.tools.bash import bash_command

    def make(body: str) -> str:
        path = tmp_path / f"hook_{next(_hook_ids)}.py"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        cmd = f'"{Path(sys.executable).as_posix()}" "{path.as_posix()}"'
        # PowerShell treats a leading quoted string as a value, not a command
        return f"& {cmd}" if bash_command.select_shell()[0] == "PowerShell" else cmd
    return make


# --- shell jobs (plan/2026-10-31-background-shell/tasks.md) ------------------------------------

_py_ids = itertools.count()


@pytest.fixture
def _py(tmp_path):
    """`_py(body)`: write `job<n>.py` and return a shell command running it with this interpreter."""
    from dostuff.tools.bash import bash_command

    def make(body: str) -> str:
        path = tmp_path / f"job{next(_py_ids)}.py"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        cmd = f'"{Path(sys.executable).as_posix()}" "{path.as_posix()}"'
        return f"& {cmd}" if bash_command.select_shell()[0] == "PowerShell" else cmd
    return make


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    import os
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie child still answers kill(0); reap it if it is ours
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def _pid_wait_dead(pid: int, timeout: float = 3.0) -> bool:
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


@pytest.fixture
def _alive():
    """`_alive(pid)`: is the process running? (no psutil)"""
    return _pid_alive


@pytest.fixture
def _wait_dead():
    """`_wait_dead(pid, timeout=3)`: poll until the process is gone; returns whether it is."""
    return _pid_wait_dead


@pytest.fixture
def jobs_home(tmp_path, monkeypatch):
    """Point the jobs root (`~/.dostuff/jobs`) at a temp dir; returns it."""
    from dostuff.tools.bash import jobs
    root = tmp_path / "jobs"
    monkeypatch.setattr(jobs, "jobs_root", lambda: root)
    return root


@pytest.fixture
def registry(jobs_home):
    """A fresh `JobRegistry`; teardown kills what is left and fails if any job process survives."""
    from dostuff.tools.bash.jobs import JobRegistry
    reg = JobRegistry()
    yield reg
    pids = [j.pid for j in reg.jobs()]
    reg.kill_all_sync()
    leaked = [pid for pid in pids if not _pid_wait_dead(pid, 5)]
    assert not leaked, f"job processes still alive after teardown: {leaked}"
