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


# --- subagents (plan/2026-11-01-subagents/tasks.md) ---------------------------------------------

def fake_response(text: str | None = None, tool_calls: list | None = None, usage=None):
    """A non-streamed litellm response with one message."""
    from types import SimpleNamespace
    message = SimpleNamespace(content=text, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def fake_tool_call(name: str, args: dict, call_id: str = "c1"):
    import json
    from types import SimpleNamespace
    arguments = json.dumps(args)
    tc = SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))
    tc.model_dump = lambda: {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
    return tc


class FakeStep:
    """One scripted reply: wait for `gate` (if any), sleep `delay`, then return `response` (or raise it)."""

    def __init__(self, response, delay: float = 0.0, gate=None):
        self.response = response
        self.delay = delay
        self.gate = gate


class FakeModel:
    """Stands in for `litellm.acompletion`. Replies are scripted per agent: `script("[A]", …)` serves calls
    whose first user message contains `[A]`, so parallel runs don't take each other's replies. Calls with no
    matching marker take from `script(None, …)`, then get `"done"`. Every call is recorded in `calls`."""

    def __init__(self):
        self.scripts: dict[str | None, list[FakeStep]] = {}
        self.calls: list[dict] = []

    def script(self, marker: str | None, *steps) -> None:
        self.scripts.setdefault(marker, []).extend(s if isinstance(s, FakeStep) else FakeStep(s) for s in steps)

    def gate(self, marker: str | None, response):
        """Script a reply held until the returned event is set."""
        import asyncio
        event = asyncio.Event()
        self.script(marker, FakeStep(response, gate=event))
        return event

    def _marker(self, messages: list[dict]) -> str | None:
        first_user = next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")
        return next((m for m in self.scripts if m is not None and m in first_user), None)

    def calls_for(self, marker: str) -> list[dict]:
        return [c for c in self.calls if self._marker(c["messages"]) == marker]

    async def acompletion(self, **kwargs):
        import asyncio
        self.calls.append(kwargs)
        queue = self.scripts.get(self._marker(kwargs["messages"])) or self.scripts.get(None) or []
        step = queue.pop(0) if queue else FakeStep(fake_response("done"))
        if step.gate is not None:
            await step.gate.wait()
        if step.delay:
            await asyncio.sleep(step.delay)
        if isinstance(step.response, BaseException):
            raise step.response
        return step.response


@pytest.fixture
def fake_model(monkeypatch):
    """Fake the model for every agent (main and subagents); returns the `FakeModel`."""
    from dostuff.agent import call_agent as call_agent_module
    from dostuff.agent import loop as loop_module
    model = FakeModel()
    monkeypatch.setattr(call_agent_module, "acompletion", model.acompletion)
    monkeypatch.setattr(loop_module, "get_model_token_limit", lambda: 100_000)
    return model


class FakeAdapter:
    """A parent adapter. `ask` waits on a future the test resolves (`answer(i, value)`), unless `auto` is set;
    prompts are recorded in order in `asks` as (title, future)."""

    def __init__(self, auto=None):
        import asyncio
        self.events: list = []
        self.asks: list = []
        self.auto = auto
        self._cancelled = False
        self._cancel_event = asyncio.Event()

    def emit(self, event_type, data):
        self.events.append((event_type, data))

    async def ask(self, message, **kwargs):
        import asyncio
        if self.auto is not None:
            self.asks.append((message, None))
            return self.auto
        future = asyncio.get_running_loop().create_future()
        self.asks.append((message, future))
        return await future

    async def ask_batch(self, messages, **kwargs):
        return "each"

    def answer(self, index: int, value) -> None:
        self.asks[index][1].set_result(value)

    def cancel(self):
        self._cancelled = True
        self._cancel_event.set()


@pytest.fixture
def agents(registry, monkeypatch):
    """A fresh `AgentRegistry` (on top of a fresh jobs `registry`); teardown fails if any run's task is left."""
    from dostuff.agent.subagents import registry as agents_module
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    reg = agents_module.AgentRegistry()
    monkeypatch.setattr(agents_module, "_agent_registry", reg)
    yield reg
    leaked = [r.id for r in reg.runs() if r.task is not None and not r.task.done()]
    assert not leaked, f"agent runs still running after the test: {leaked}"


@pytest.fixture
def agents_home(tmp_path, monkeypatch):
    """A fake `~` with empty `.dostuff/agents` and `.claude/agents`, so tests never read the real ones."""
    home = tmp_path / "home"
    for folder in (".dostuff", ".claude"):
        (home / folder / "agents").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


@pytest.fixture
def _start_subagent(monkeypatch):
    """`await _start_subagent(task, tools=…, isolation=…, adapter=…)`: start a subagent the way the main agent
    does and return the tool result. The Phase 1 pins go through this one helper (tasks.md T1.x, T2.6)."""
    from dostuff.tools.delegate_to_subagent import _impl_delegate_to_subagent

    async def start(task: str, *, tools=None, isolation: str = "", adapter=None, permissions=None) -> str:
        return await _impl_delegate_to_subagent(
            task=task, role="pin", mcp_client=object(),  # type: ignore[arg-type]
            session_id="s", turn_id="t", requested_tools=tools, adapter=adapter,
            permissions=permissions, isolation=isolation,
        )
    return start
