"""Shared fixtures."""
import itertools
import re
import sys
import textwrap
from pathlib import Path

import pytest

_hook_ids = itertools.count()


# --- optional extras (plan/2026-11-08-multimodal-input, CON-010) -------------------------------

EXTRA_MODULES = {
    "images": ["PIL"],
    "docs": ["pypdf", "docx", "openpyxl", "pptx"],
}


def requires_extra(name: str) -> None:
    """Skip the calling test when `dostuff[name]` isn't installed — or fail it, when DOSTUFF_REQUIRE_EXTRAS=1
    (CI), so a broken extra can't quietly turn into skipped tests."""
    import importlib
    import os
    missing = []
    for module in EXTRA_MODULES[name]:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if not missing:
        return
    reason = f"needs dostuff[{name}] (missing: {', '.join(missing)})"
    if os.environ.get("DOSTUFF_REQUIRE_EXTRAS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


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
def registry(jobs_home, monkeypatch):
    """A fresh `JobRegistry` on a fresh notice queue; teardown kills what is left and fails if any job process
    survives."""
    from dostuff.helpers.agent import notices
    from dostuff.tools.bash.jobs import JobRegistry
    monkeypatch.setattr(notices, "_queue", notices.NoticeQueue())
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


FORK_DIRECTIVE_START = "You are a fork of the main agent"  # dostuff.agent.subagents.fork.FORK_DIRECTIVE


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
        users = [str(m.get("content") or "") for m in messages if m.get("role") == "user"]
        # The turn context is the harness's, not the agent's task; it may be merged into the prompt's message
        users = [re.sub(r"<turn-context>.*?</turn-context>\s*", "", u, flags=re.DOTALL) for u in users]
        users = [u for u in users if u]
        # A fork's history starts with the main agent's; its own task starts at the fork directive
        forks = [u[u.index(FORK_DIRECTIVE_START):] for u in users if FORK_DIRECTIVE_START in u]
        first_user = forks[0] if forks else users[0] if users else ""
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
    prompts are recorded in order in `asks` as (title, future). `ask_user` returns `ask_answers` (scripted)."""

    def __init__(self, auto=None, ask_answers=None):
        import asyncio
        self.events: list = []
        self.asks: list = []
        self.auto = auto
        self.ask_answers = ask_answers  # scripted [[(question, answer), …]] for ask_user (ask-user T2.6)
        self.asked: list = []  # what ask_user was given, per call
        self._cancelled = False
        self._cancel_event = asyncio.Event()

    async def ask_user(self, questions, **kwargs):
        self.asked.append(questions)
        return self.ask_answers

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


@pytest.fixture(autouse=True)
def _restore_skill_roots():
    """Discovery publishes the sandbox's allowlist process-wide, and many tests discover without saying so."""
    from dostuff.helpers.tools.resolve_safe_path import set_skill_roots, skill_roots
    roots = skill_roots()
    yield
    set_skill_roots(roots)


@pytest.fixture
def home_skills(tmp_path, monkeypatch):
    """A fake `~` with empty skill folders. Three patches, because no one of them reaches every reader:
    `Path.home` for the sandbox, `SKILLS_DIRS` (computed at import) for discovery, the env for `expanduser`."""
    from dostuff.helpers.skills import discover_skills as ds
    home = tmp_path / "home"
    folders = [home / ".dostuff" / "skills", home / ".agents" / "skills"]
    for folder in folders:
        folder.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(ds, "SKILLS_DIRS", [str(folder) for folder in folders])
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


@pytest.fixture
def _start_subagent(monkeypatch):
    """`await _start_subagent(task, tools=…, isolation=…, adapter=…)`: start a subagent the way the main agent
    does and return the tool result, without the `agent_id` footer. The Phase 1 pins go through this one helper
    (tasks.md T1.x, T2.6); `tools` registers a `pin` type with that tool list."""
    from dostuff.agent.subagents.definitions import BUILTIN_TYPES, AgentDef
    from dostuff.tools.agent import FOOTER, agent

    async def start(task: str, *, tools=None, isolation: str = "", adapter=None, permissions=None) -> str:
        agent_type = "general-purpose"
        if tools is not None:
            monkeypatch.setitem(BUILTIN_TYPES, "pin", AgentDef(name="pin", description="pin", tools=tuple(tools)))
            agent_type = "pin"
        result = await agent(
            description="pin", prompt=task, mcp_client=object(),  # type: ignore[arg-type]
            session_id="s", turn_id="t", subagent_type=agent_type, isolation=isolation, adapter=adapter,
            permissions=permissions,
        )
        body, sep, footer = result.rpartition("\n\n" + FOOTER.split("{")[0])
        return body if sep else result  # errors have no footer
    return start


@pytest.fixture
def agent_type(monkeypatch):
    """`agent_type("rev", tools=["read_file"])`: register an agent type for this test only."""
    from dostuff.agent.subagents.definitions import BUILTIN_TYPES, AgentDef

    def register(name: str, *, tools=None, prompt: str = "", model: str = "inherit", description: str = "test type",
                 max_iterations=None):
        agent_def = AgentDef(name=name, description=description, prompt=prompt,
                             tools=tuple(tools) if tools is not None else None, model=model, source="test",
                             max_iterations=max_iterations)
        monkeypatch.setitem(BUILTIN_TYPES, name, agent_def)
        return agent_def
    return register


# --- ask_user (plan/2026-11-02-ask-user/tasks.md) -----------------------------------------------

class AskScript:
    """A scripted `ask_user` adapter: records the cleaned questions it was given and returns the next
    scripted answer list (`[("Which db?", "SQLite")]`), or `None` for a cancelled call.

    `raise_exc=` makes it raise, to prove the tool never lets an adapter exception escape (T1.5).
    """

    def __init__(self, answers=None, raise_exc=None):
        self.answers = answers  # a list of pairs, or None for a cancelled call
        self.raise_exc = raise_exc
        self.questions: list = []  # every call's questions, in order
        self.calls = 0

    async def ask_user(self, questions, **kwargs):
        self.calls += 1
        self.questions.append(questions)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.answers


@pytest.fixture
def ask_script():
    """`AskScript(answers)` for the tool-level tests (no widgets, no app)."""
    return AskScript


class FakeTui:
    """The app calls `TuiAdapter` makes for a question, for the no-widget adapter tests of Phase 3.

    A recording stub, not a Textual app: the real widget behaviour is the Phase 4 pilot tests' business.
    """

    def __init__(self):
        self.calls: list = []  # (method name, args) in order
        self._pending_ui: list = []
        self.turn_in_progress = True

    def ui(self, fn):
        self.calls.append(("ui", ()))
        self._pending_ui.append(fn)

    def flush(self) -> None:
        while self._pending_ui:
            self._pending_ui.pop(0)()

    def _mount_message(self, widget) -> None:
        self.calls.append(("_mount_message", (widget,)))

    def _show_ask_panel(self, panel_state) -> None:
        self.calls.append(("_show_ask_panel", (panel_state,)))

    def _hide_ask_panel(self) -> None:
        self.calls.append(("_hide_ask_panel", ()))

    def _show_permission_prompt(self, question: str, options: list) -> None:
        self.calls.append(("_show_permission_prompt", (question, options)))

    def _hide_permission_prompt(self, restore_input: bool = True) -> None:
        self.calls.append(("_hide_permission_prompt", (restore_input,)))

    def _flash_input_hint(self, text: str, seconds: float = 2) -> None:
        self.calls.append(("_flash_input_hint", (text,)))

    def _hand_input_over(self, placeholder: str = "") -> None:
        self.calls.append(("_hand_input_over", (placeholder,)))

    def _give_input_back(self) -> None:
        self.calls.append(("_give_input_back", ()))

    def _approval_waiting(self, question: str = "") -> None:
        self.calls.append(("_approval_waiting", (question,)))

    def _set_title(self, state: str) -> None:
        self.calls.append(("_set_title", (state,)))

    @property
    def _turn_in_progress(self) -> bool:
        return self.turn_in_progress

    def named(self, name: str) -> list:
        """Every call of one method, in order."""
        return [args for method, args in self.calls if method == name]


@pytest.fixture
def fake_tui():
    """A `FakeTui` recording stub for the adapter's no-widget tests."""
    return FakeTui


# --- /btw (plan/2026-11-03-btw/tasks.md) --------------------------------------------------------

def make_fork_ctx(**overrides):
    """A small `ForkContext`: a user step, an assistant tool call with its result, then an assistant answer."""
    from dostuff.agent.subagents.fork import ForkContext
    fields = dict(
        system_instruction="SYS", compaction_notes="", tool_names=["read_file", "bash"], session_hook_context=None,
        history=[
            {"role": "user", "content": "fix the retry bug"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]},
            {"role": "tool", "name": "read_file", "tool_call_id": "c1", "content": "def retry(): ..."},
            {"role": "assistant", "content": "Found it in a.py."},
        ],
    )
    fields.update(overrides)
    return ForkContext(**fields)


@pytest.fixture
def fork_ctx():
    """`fork_ctx(**overrides) -> ForkContext`."""
    return make_fork_ctx


class FakeSpan:
    def __init__(self, recording: bool = True):
        self.recording = recording
        self.ended = 0

    def is_recording(self) -> bool:
        return self.recording

    def end(self) -> None:
        self.ended += 1


class SideCall:
    """A scripted `call_agent` for the side call: records what it was given (deep-copied) and returns
    `(response, span)` (`responses` in order, then `fake_response("ok")`), or raises `raise_exc`.
    `gate` (an asyncio.Event) holds every call until it is set."""

    def __init__(self, *responses, raise_exc=None, recording: bool = True, gate=None):
        self.responses = list(responses)
        self.raise_exc = raise_exc
        self.recording = recording
        self.gate = gate
        self.calls: list[dict] = []
        self.spans: list[FakeSpan] = []

    async def __call__(self, steps_history, system_instruction, tool_names=None, stream=False):
        import copy
        self.calls.append({"steps_history": copy.deepcopy(steps_history), "system_instruction": system_instruction,
                           "tool_names": copy.deepcopy(tool_names), "stream": stream})
        if self.gate is not None:
            await self.gate.wait()
        if self.raise_exc is not None:
            raise self.raise_exc
        span = FakeSpan(self.recording)
        self.spans.append(span)
        return (self.responses.pop(0) if self.responses else fake_response("ok")), span


@pytest.fixture
def side_call(monkeypatch):
    """`side_call(*responses, raise_exc=…)`: install a `SideCall` where `answer_side_question` imports
    `call_agent` from, and return it. A stub module in `sys.modules`, so the unit tests never load litellm."""
    import types

    def install(*responses, **kwargs) -> SideCall:
        fake = SideCall(*responses, **kwargs)
        module = sys.modules.get("dostuff.agent.call_agent")
        if getattr(module, "_side_call_stub", False):  # this test's own stub, from an earlier install
            monkeypatch.setattr(module, "call_agent", fake)
        elif module is not None:
            # loop.py binds `call_agent` by name on import: imported after this patch, a subagent's loop would get
            # the fake. litellm is loaded already, so importing it now costs nothing.
            import dostuff.agent.loop  # noqa: F401
            monkeypatch.setattr(module, "call_agent", fake)
        else:
            stub = types.ModuleType("dostuff.agent.call_agent")
            stub.call_agent = fake  # type: ignore[attr-defined]
            stub._side_call_stub = True  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, "dostuff.agent.call_agent", stub)
        return fake
    return install


class FakeAgentRegistry:
    """`start(**kwargs)` records the kwargs and returns a run with id `r1`, or raises `raise_exc`. Everything
    else (listeners, `runs`) is the real registry's, so the app around it keeps working."""

    def __init__(self, raise_exc=None):
        self.raise_exc = raise_exc
        self.started: list[dict] = []

    def __getattr__(self, name):
        from dostuff.agent.subagents.registry import get_agent_registry
        return getattr(get_agent_registry(), name)

    async def start(self, **kwargs):
        from types import SimpleNamespace
        self.started.append(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc
        return SimpleNamespace(id="r1", run_id="r1")
