"""Lifecycle hooks (plan/2026-10-29-hooks/spec.md, tasks.md)."""
import asyncio
import dataclasses
import importlib
import itertools
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

from dostuff import config as config_module
from dostuff.config import get_config, reset_config
from dostuff.helpers import hooks
from dostuff.helpers.hooks import HookEntry, HookOutcome, load_hooks, run_hook, run_hooks
# the package re-exports the run_hook function, which shadows the submodule's name
dispatch_module = importlib.import_module("dostuff.helpers.hooks.dispatch")
run_hook_module = importlib.import_module("dostuff.helpers.hooks.run_hook")
trust_module = importlib.import_module("dostuff.helpers.hooks.trust")
from dostuff.helpers.hooks.load_hooks import load_merged_hooks
from dostuff.helpers.hooks.run_hook import payload_for
from dostuff.helpers.tools.project_root import project_root, use_project_root
from dostuff.tools.bash import bash_command as bash_module

D9 = r'''
hooks:
  project_trust: ask          # ask | allow | deny  (D4) — only read from ~/.dostuff/config.yaml
  stop_block_limit: 3         # Stop blocks honoured per turn before the loop is allowed to end (1–10)
  SessionStart:
    - matcher: "startup|resume|clear"   # regex, full match against the source; empty or "*" = all
      hooks:
        - type: command
          command: "git status --short"
  PreToolUse:
    - matcher: "bash_command"           # regex, full match against the tool name
      if: "(?i)^\\s*git\\s+push"        # optional regex, searched in tool_input.command (shell tool only)
      fail_closed: true
      timeout: 10
      hooks:
        - type: command
          name: block-main-push
          command: "python .dostuff/hooks/block_main_push.py"
  PostToolUse:
    - matcher: "write_file|edit_file"
      hooks:
        # the script reads tool_input.path from the stdin JSON and runs the formatter on it;
        # its stdout is fed back to the model (D5)
        - type: command
          name: format
          command: "python .dostuff/hooks/format_file.py"
  Stop:
    - hooks:
        - { type: command, name: tests, command: "python .dostuff/hooks/run_tests.py" }
'''


# --- fixtures and helpers ---------------------------------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake `~` (with `.dostuff/`) and a project cwd; returns the home path."""
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    (home / ".dostuff").mkdir(parents=True)
    (proj / ".dostuff").mkdir(parents=True)
    monkeypatch.setattr(config_module.Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(proj)
    for var in ("MODEL", "MODEL_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(dispatch_module, "_untrusted_warned", set())
    monkeypatch.setattr(trust_module, "_decisions", {})
    reset_config()
    yield home
    reset_config()


@pytest.fixture
def proj(home):
    return Path.cwd()


def write_user_config(home: Path, data: dict) -> Path:
    path = home / ".dostuff" / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    reset_config()
    return path


def write_project_config(proj: Path, data: dict) -> Path:
    path = proj / ".dostuff" / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    reset_config()
    return path


_script_ids = itertools.count()


def _python_cmd(path: Path) -> str:
    cmd = f'"{Path(sys.executable).as_posix()}" "{path.as_posix()}"'
    # PowerShell treats a leading quoted string as a value, not a command
    return f"& {cmd}" if bash_module.select_shell()[0] == "PowerShell" else cmd


def _script(tmp_path: Path, body: str) -> str:
    """Write a hook script and return a shell command that runs it with this interpreter."""
    path = tmp_path / f"hook{next(_script_ids)}.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return _python_cmd(path)


def _cmd(text: str) -> str:
    """A one-liner that behaves the same in bash, PowerShell and cmd (`echo hi`, `exit 2`)."""
    return text


class ScriptedAdapter:
    def __init__(self, answers=None):
        self.events: list = []
        self.answers = list(answers or [])
        self._cancelled = False
        self._cancel_event = asyncio.Event()

    def emit(self, event_type, data):
        self.events.append((event_type, data))

    async def ask(self, message, **kwargs):
        return self.answers.pop(0)

    def cancel(self):
        self._cancelled = True
        self._cancel_event.set()


def _hooks_block(event: str, *groups: dict) -> dict:
    return {"hooks": {event: list(groups)}}


def _group(command: str, **kw) -> dict:
    hook = {"type": "command", "command": command}
    for key in ("name", "timeout"):
        if key in kw:
            hook[key] = kw.pop(key)
    return {**kw, "hooks": [hook]}


def _pre(tool: str = "bash_command", tool_input: dict | None = None) -> dict:
    return payload_for(
        "PreToolUse", session_id="s1", turn_id="t1", turn_type="interactive_loop",
        tool_name=tool, tool_input=tool_input if tool_input is not None else {"command": "ls"}, tool_use_id="c1",
    )


def _entry(command: str, event: str = "PreToolUse", **kw) -> HookEntry:
    entries, warnings = load_hooks({event: [_group(command, **kw)]}, source="user", origin="test.yaml")
    assert not warnings and len(entries) == 1
    return entries[0]


def _run(coro):
    return asyncio.run(coro)


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_dead(pid: int, within: float = 3.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return not _alive(pid)


# --- T0.1 -------------------------------------------------------------------------------------

def test_imports():
    expected = {"load_hooks", "run_hook", "run_hooks", "HookOutcome", "HookResult", "HookEntry",
                "user_prompt_submit", "session_start"}
    assert set(hooks.__all__) == expected
    for name in expected:
        assert getattr(hooks, name) is not None


# --- T1.1 -------------------------------------------------------------------------------------

def test_load_hooks_parses_block():
    raw = yaml.safe_load(D9)["hooks"]
    entries, warnings = load_hooks(raw, source="user", origin="/home/u/.dostuff/config.yaml")
    assert warnings == []
    assert [e.event for e in entries] == ["SessionStart", "PreToolUse", "PostToolUse", "Stop"]
    start, pre, post, stop = entries
    assert start.matcher == "startup|resume|clear" and start.name == "SessionStart#1"
    assert start.command == "git status --short" and start.timeout == 30 and start.fail_closed is False
    assert pre.matcher == "bash_command" and pre.if_pattern == r"(?i)^\s*git\s+push"
    assert pre.fail_closed is True and pre.timeout == 10 and pre.name == "block-main-push"
    assert pre.command == "python .dostuff/hooks/block_main_push.py"
    assert post.matcher == "write_file|edit_file" and post.name == "format" and post.if_pattern is None
    assert stop.matcher is None and stop.name == "tests"
    for e in entries:
        assert e.source == "user" and e.origin == "/home/u/.dostuff/config.yaml" and e.broken is None

    # group-level fail_closed / timeout apply to every hook in the group; a hook-level value wins
    entries, warnings = load_hooks({"PreToolUse": [{
        "fail_closed": True, "timeout": 7,
        "hooks": [{"type": "command", "command": "echo a"},
                  {"type": "command", "command": "echo b", "timeout": 3, "fail_closed": False}],
    }]}, source="user", origin="x")
    assert warnings == []
    assert [(e.timeout, e.fail_closed, e.name) for e in entries] == [
        (7, True, "PreToolUse#1"), (3, False, "PreToolUse#2")]


# --- T1.2 -------------------------------------------------------------------------------------

def test_hooks_provenance(home, proj):
    user_path = write_user_config(home, _hooks_block("PreToolUse", _group("echo user", name="u")))

    # (a) a project `PreToolUse: []` does not remove the user's hooks
    write_project_config(proj, {"hooks": {"PreToolUse": []}})
    cfg = get_config()
    assert [e.name for e in cfg.hooks["PreToolUse"]] == ["u"]

    # (b) project entries are appended after the user's, tagged
    proj_path = write_project_config(proj, _hooks_block("PreToolUse", _group("echo proj", name="p")))
    cfg = get_config()
    pre = cfg.hooks["PreToolUse"]
    assert [(e.name, e.source, e.origin) for e in pre] == [
        ("u", "user", str(user_path)), ("p", "project", str(proj_path))]

    # (c) a project file cannot trust itself
    write_project_config(proj, {"hooks": {"project_trust": "allow"}})
    cfg = get_config()
    assert cfg.hooks_project_trust == "ask"
    assert len([w for w in cfg.hooks_warnings if "project_trust" in w]) == 1

    # (d) stop_block_limit is the minimum of the two files
    write_user_config(home, {"hooks": {"stop_block_limit": 5}})
    write_project_config(proj, {"hooks": {"stop_block_limit": 2}})
    assert get_config().hooks_stop_block_limit == 2
    write_user_config(home, {"hooks": {"stop_block_limit": 2}})
    write_project_config(proj, {"hooks": {"stop_block_limit": 5}})
    assert get_config().hooks_stop_block_limit == 2

    # the engine does not read the generically merged Config.raw["hooks"]
    write_user_config(home, _hooks_block("PreToolUse", _group("echo user", name="u")))
    cfg = get_config()
    cfg.raw["hooks"] = {"PreToolUse": [_group("echo other", name="other")]}
    assert [e.name for e in get_config().hooks["PreToolUse"]] == ["u"]


def test_project_hooks_skipped_until_trusted(home, proj, tmp_path):
    log = tmp_path / "ran.log"
    cmd = _script(tmp_path, f"open({str(log)!r}, 'a').write('ran')")
    write_project_config(proj, _hooks_block("PreToolUse", _group(cmd)))
    adapter = ScriptedAdapter(answers=["no"])  # asked once; the answer holds for the session
    for _ in range(2):
        outcome = _run(run_hooks("PreToolUse", _pre(), adapter=adapter))
        assert outcome.decision == "none"
    assert not log.exists()
    assert len([e for e in adapter.events if "project hooks not trusted" in str(e[1])]) == 1


# --- T1.3 -------------------------------------------------------------------------------------

GOOD = {"type": "command", "command": "echo ok", "name": "good"}


@pytest.mark.parametrize("block, kept_names, check", [
    ({"PreTool": [{"hooks": [GOOD]}], "PreToolUse": [{"hooks": [GOOD]}]}, ["good"], None),
    ({"PreToolUse": {"hooks": [GOOD]}, "Stop": [{"hooks": [GOOD]}]}, ["good"], None),
    ({"PreToolUse": [{"hooks": [{"type": "http", "url": "x"}, GOOD]}]}, ["good"], None),
    ({"PreToolUse": [{"hooks": [{"command": "echo x"}, GOOD]}]}, ["good"], None),
    ({"PreToolUse": [{"matcher": "[", "hooks": [{"type": "command", "command": "echo x"}]},
                     {"hooks": [GOOD]}]}, ["good"], None),
    ({"PreToolUse": [{"if": "(", "hooks": [{"type": "command", "command": "echo x"}]},
                     {"hooks": [GOOD]}]}, ["good"], None),
    ({"PreToolUse": [{"if": "^(?i)git", "hooks": [{"type": "command", "command": "echo x"}]},
                     {"hooks": [GOOD]}]}, ["good"], None),
    ({"PreToolUse": [{"hooks": [{**GOOD, "timeout": "abc"}]}]}, ["good"], lambda e: e.timeout == 30),
    ({"PreToolUse": [{"hooks": [{**GOOD, "timeout": 9999}]}]}, ["good"], lambda e: e.timeout == 300),
    ({"PreToolUse": [{"hooks": "nope"}, {"hooks": [GOOD]}]}, ["good"], None),
    ({"PostToolUse": [{"fail_closed": True, "hooks": [GOOD]}]}, ["good"], lambda e: e.fail_closed is False),
])
def test_load_hooks_drops_invalid_entries_with_warnings(block, kept_names, check):
    entries, warnings = load_hooks(block, source="user", origin="x")
    assert warnings, block
    assert [e.name for e in entries] == kept_names
    assert all(e.broken is None for e in entries)
    if check:
        assert check(entries[0])


def test_load_hooks_block_not_a_mapping():
    entries, warnings = load_hooks("nope", source="user", origin="x")
    assert entries == [] and len(warnings) == 1


@pytest.mark.parametrize("value, expected", [(0, 1), (50, 10), ("x", 3)])
def test_stop_block_limit_clamped(tmp_path, value, expected):
    user = tmp_path / "user.yaml"
    user.write_text(yaml.safe_dump({"hooks": {"stop_block_limit": value}}), encoding="utf-8")
    loaded = load_merged_hooks(user, tmp_path / "missing.yaml")
    assert loaded.stop_block_limit == expected
    assert any("stop_block_limit" in w for w in loaded.warnings)


def test_broken_fail_closed_entry_is_kept(home, monkeypatch):
    def no_subprocess(*a, **kw):
        raise AssertionError("a broken entry must not start a process")

    write_user_config(home, _hooks_block(
        "PreToolUse", _group("echo x", name="guard", matcher="bash_command", fail_closed=True, **{"if": "^(?i)git"})))
    cfg = get_config()
    (entry,) = cfg.hooks["PreToolUse"]
    assert entry.broken and "if" in entry.broken
    assert any("guard" in w or "PreToolUse" in w for w in cfg.hooks_warnings)
    monkeypatch.setattr(run_hook_module.asyncio, "create_subprocess_exec", no_subprocess)
    outcome = _run(run_hooks("PreToolUse", _pre("bash_command", {"command": "ls"})))
    assert outcome.decision == "deny"
    assert outcome.reason.startswith("hook 'guard' misconfigured: ")
    # a bad `if` is ignored, so it still only matches its matcher
    assert _run(run_hooks("PreToolUse", _pre("read_file", {"path": "a"}))).decision == "none"

    # a bad matcher matches every tool
    write_user_config(home, _hooks_block("PreToolUse", _group("echo x", name="guard", matcher="[", fail_closed=True)))
    outcome = _run(run_hooks("PreToolUse", _pre("read_file", {"path": "a"})))
    assert outcome.decision == "deny" and "misconfigured" in outcome.reason


# --- T1.4 -------------------------------------------------------------------------------------

def test_run_hook_payload_and_env(home, tmp_path):
    out = tmp_path / "seen.json"
    cmd = _script(tmp_path, f"""
        import json, os, sys
        payload = json.load(sys.stdin)
        json.dump({{"env": dict(os.environ), "stdin": payload, "tty": sys.stdin.isatty()}}, open({str(out)!r}, "w"))
    """)
    base = {"session_id": "s1", "turn_id": "t1", "turn_type": "interactive_loop"}
    cases = {
        "PreToolUse": {"tool_name": "bash_command", "tool_input": {"command": "ls"}, "tool_use_id": "c1"},
        "UserPromptSubmit": {"prompt": "hello"},
        "SessionStart": {"source": "startup"},
        "Stop": {"last_assistant_message": "done", "stop_hook_active": False},
    }
    for event, extra in cases.items():
        payload = payload_for(event, **base, **extra)
        result = _run(run_hook(_entry(cmd, event=event), payload))
        assert result.error is None, result
        seen = json.loads(out.read_text())
        stdin = seen["stdin"]
        assert stdin == payload
        assert stdin["hook_event_name"] == event
        assert {k: stdin[k] for k in base} == base
        assert stdin["cwd"] == str(project_root())
        assert set(stdin) == {"hook_event_name", "cwd", *base, *extra}
        env = seen["env"]
        assert env["DOSTUFF_HOOK_EVENT"] == event
        assert env["DOSTUFF_SESSION_ID"] == "s1" and env["DOSTUFF_TURN_ID"] == "t1"
        assert env["DOSTUFF_TURN_TYPE"] == "interactive_loop"
        assert env["DOSTUFF_PROJECT_ROOT"] == str(project_root())
        assert seen["tty"] is False
    with pytest.raises(TypeError):
        payload_for("PreToolUse", **base, tool_name="x")  # missing tool_input / tool_use_id


# --- T1.5 -------------------------------------------------------------------------------------

def test_run_hook_stdout_forms(home, tmp_path):
    def run(body):
        return _run(run_hook(_entry(_script(tmp_path, body)), _pre()))

    plain = _run(run_hook(_entry(_cmd("echo notes")), _pre()))
    assert plain.decision == "none" and plain.context == "notes" and plain.error is None

    block = run('print(\'{"decision": "block", "reason": "no"}\')')
    assert block.decision == "deny" and block.reason == "no" and block.context == ""

    full = run("""
        import json
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "allow",
            "permissionDecisionReason": "fine", "updatedInput": {"command": "ls -la"},
            "additionalContext": "extra"}}))
    """)
    assert full.decision == "allow" and full.reason == "fine"
    assert full.updated_input == {"command": "ls -la"} and full.context == "extra"

    deny = run("""
        import json
        print(json.dumps({"hookSpecificOutput": {"permissionDecision": "deny", "permissionDecisionReason": "nope"}}))
    """)
    assert deny.decision == "deny" and deny.reason == "nope"

    odd = run('print(\'{"decision": "maybe"}\')')
    assert odd.decision == "none" and odd.warning and "maybe" in odd.warning
    with pytest.raises(dataclasses.FrozenInstanceError):
        odd.decision = "deny"  # type: ignore[misc]


# --- T1.6 -------------------------------------------------------------------------------------

def test_run_hook_exit_codes(home, tmp_path):
    def run(body):
        return _run(run_hook(_entry(_script(tmp_path, body), name="chk"), _pre()))

    r = run("import sys; sys.stderr.write('no pushes'); sys.exit(2)")
    assert r.decision == "deny" and r.reason == "no pushes" and r.error is None

    r = run("import sys; print('from stdout'); sys.exit(2)")
    assert r.decision == "deny" and r.reason == "from stdout"

    r = run("import sys; sys.stderr.write('boom'); sys.exit(1)")
    assert r.decision == "none" and r.exit_code == 1
    assert r.error and r.error.startswith("hook 'chk'") and "1" in r.error and "boom" in r.error

    r = run("""
        import json, sys
        print(json.dumps({"decision": "block", "reason": "x"}))
        sys.exit(3)
    """)
    assert r.decision == "none" and r.error and "3" in r.error


# --- T1.7 -------------------------------------------------------------------------------------

_SPAWNER = """
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    open({pid_file!r}, "w").write(str(child.pid))
    time.sleep(30)
"""


def _read_pid(path: Path) -> int:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError("hook never wrote its child pid")


def test_run_hook_timeout_kills_and_returns(home, tmp_path):
    pid_file = tmp_path / "child.pid"
    entry = dataclasses.replace(_entry(_script(tmp_path, _SPAWNER.format(pid_file=str(pid_file)))), timeout=1.0)
    started = time.monotonic()
    result = _run(run_hook(entry, _pre()))
    assert time.monotonic() - started < 4
    assert result.error and "timed out" in result.error and not result.cancelled
    assert _wait_dead(_read_pid(pid_file))


def test_run_hook_cancel_kills(home, tmp_path):
    pid_file = tmp_path / "child.pid"
    entry = dataclasses.replace(_entry(_script(tmp_path, _SPAWNER.format(pid_file=str(pid_file)))), timeout=10.0)
    adapter = ScriptedAdapter()

    async def main():
        task = asyncio.create_task(run_hook(entry, _pre(), adapter=adapter))
        await asyncio.to_thread(_read_pid, pid_file)
        adapter.cancel()
        cancelled_at = time.monotonic()
        result = await task
        return result, time.monotonic() - cancelled_at

    result, elapsed = _run(main())
    assert result.cancelled and result.decision == "none"
    assert elapsed < 2.0
    assert _wait_dead(_read_pid(pid_file))


def test_run_hook_output_cap(home, tmp_path):
    entry = _entry(_script(tmp_path, "print('x' * 10000)"))
    write_user_config(home, {"loop": {"max_tool_output_chars": 1000}, "hooks": {"PostToolUse": [
        {"hooks": [{"type": "command", "command": entry.command}]}]}})
    payload = payload_for("PostToolUse", session_id="s", turn_id="t", turn_type="interactive_loop",
                          tool_name="read_file", tool_input={}, tool_use_id="c", tool_response="r")
    outcome = _run(run_hooks("PostToolUse", payload))
    assert len(outcome.context) < 1200 and "truncated" in outcome.context


# --- T1.8 -------------------------------------------------------------------------------------

def _logger(tmp_path: Path, log: Path, tag: str) -> str:
    return _script(tmp_path, f"open({str(log)!r}, 'a').write({tag!r} + '\\n')")


def _log_lines(log: Path) -> list[str]:
    return log.read_text().split() if log.exists() else []


def test_run_hooks_matching(home, tmp_path):
    log = tmp_path / "log.txt"
    write_user_config(home, {"hooks": {
        "PreToolUse": [
            _group(_logger(tmp_path, log, "bash"), matcher="bash_command"),
            _group(_logger(tmp_path, log, "files"), matcher="write_file|edit_file"),
            _group(_logger(tmp_path, log, "all"), matcher="*"),
            _group(_logger(tmp_path, log, "partial"), matcher="write"),
            _group(_logger(tmp_path, log, "push"), matcher="*", **{"if": r"(?i)^\s*git\s+push"}),
        ],
        "SessionStart": [_group(_logger(tmp_path, log, "start"), matcher="startup|resume|clear")],
    }})

    def fired(payload, event="PreToolUse"):
        log.unlink(missing_ok=True)
        _run(run_hooks(event, payload))
        return _log_lines(log)

    assert fired(_pre("bash_command", {"command": "git status"})) == ["bash", "all"]
    assert fired(_pre("bash_command", {"command": "GIT push origin main"})) == ["bash", "all", "push"]
    # `if` only filters the shell tool; full match means "write" does not fire for write_file
    assert fired(_pre("write_file", {"path": "a", "content": ""})) == ["files", "all", "push"]
    assert fired(_pre("read_file", {"path": "a"})) == ["all", "push"]

    def start(source):
        return payload_for("SessionStart", session_id="s", turn_id="t", turn_type="interactive_loop", source=source)

    assert fired(start("resume"), "SessionStart") == ["start"]
    assert fired(start("compact"), "SessionStart") == []


def test_run_hooks_no_hooks_is_a_noop(home, tmp_path, monkeypatch):
    import builtins

    def boom(*a, **kw):
        raise AssertionError("no-op path touched the subprocess API or the filesystem")

    for config in ({}, {"hooks": {}}, {"hooks": {"Stop": [_group("echo x")]}}):
        write_user_config(home, config)
        get_config()  # load outside the patched section
        with monkeypatch.context() as m:
            m.setattr(run_hook_module.asyncio, "create_subprocess_exec", boom)
            m.setattr(builtins, "open", boom)
            assert _run(run_hooks("PreToolUse", _pre())) == HookOutcome()


# --- T1.9 -------------------------------------------------------------------------------------

def _decide(tmp_path, decision=None, *, updated=None, context=None, name=None):
    spec: dict = {}
    if decision:
        spec["permissionDecision"] = decision
        spec["permissionDecisionReason"] = f"{decision}-reason"
    if updated is not None:
        spec["updatedInput"] = updated
    if context is not None:
        spec["additionalContext"] = context
    body = f"import json; print(json.dumps({{'hookSpecificOutput': {spec!r}}}))"
    return _group(_script(tmp_path, body), **({"name": name} if name else {}))


def test_run_hooks_merge(home, tmp_path):
    seen = tmp_path / "seen.json"
    recorder = _group(_script(tmp_path, f"""
        import json, sys
        json.dump(json.load(sys.stdin)["tool_input"], open({str(seen)!r}, "w"))
    """))
    write_user_config(home, {"hooks": {"PreToolUse": [
        _decide(tmp_path, "allow", context="one"),
        _decide(tmp_path, "ask", updated={"command": "first"}, context="two"),
        _decide(tmp_path, "deny", updated={"command": "second"}, context="three"),
        recorder,
        _decide(tmp_path, "allow"),
    ]}})
    outcome = _run(run_hooks("PreToolUse", _pre("bash_command", {"command": "original"})))
    assert outcome.decision == "deny" and outcome.reason == "deny-reason"
    assert outcome.context == "one\ntwo\nthree"
    assert [name for name, _ in outcome.contexts] == ["PreToolUse#1", "PreToolUse#2", "PreToolUse#3"]
    assert outcome.updated_input == {"command": "second"}
    assert json.loads(seen.read_text()) == {"command": "original"}

    write_user_config(home, {"hooks": {"PreToolUse": [
        _group(_script(tmp_path, "import sys; sys.exit(1)"), name="crash"),
        _decide(tmp_path, "deny"),
    ]}})
    outcome = _run(run_hooks("PreToolUse", _pre()))
    assert outcome.decision == "deny"
    assert len(outcome.errors) == 1 and "crash" in outcome.errors[0]


def test_allow_with_rewrite_is_none(home, tmp_path):
    write_user_config(home, {"hooks": {"PreToolUse": [
        _decide(tmp_path, "allow"),
        _decide(tmp_path, updated={"command": "ls -la"}),
    ]}})
    outcome = _run(run_hooks("PreToolUse", _pre()))
    assert outcome.decision == "none" and outcome.updated_input == {"command": "ls -la"}


# --- T1.10 ------------------------------------------------------------------------------------

def test_fail_closed(home, tmp_path):
    crash = _script(tmp_path, "import sys; sys.exit(1)")
    slow = _script(tmp_path, "import time; time.sleep(5)")
    prompt = payload_for("UserPromptSubmit", session_id="s", turn_id="t", turn_type="interactive_loop", prompt="hi")
    for event, payload, command in (("PreToolUse", _pre(), crash), ("PreToolUse", _pre(), slow),
                                    ("UserPromptSubmit", prompt, crash)):
        write_user_config(home, _hooks_block(event, _group(command, name="g", timeout=1, fail_closed=False)))
        outcome = _run(run_hooks(event, payload))
        assert outcome.decision == "none" and len(outcome.errors) == 1
        write_user_config(home, _hooks_block(event, _group(command, name="g", timeout=1, fail_closed=True)))
        outcome = _run(run_hooks(event, payload))
        assert outcome.decision == "deny" and outcome.reason == "hook 'g' failed"
        assert len(outcome.errors) == 1


def test_cancelled_hook_is_not_promoted_to_deny(home, tmp_path):
    write_user_config(home, _hooks_block("PreToolUse", _group(
        _script(tmp_path, "import time; time.sleep(5)"), name="g", fail_closed=True)))
    adapter = ScriptedAdapter()

    async def main():
        task = asyncio.create_task(run_hooks("PreToolUse", _pre(), adapter=adapter))
        await asyncio.sleep(0.3)
        adapter.cancel()
        return await task

    outcome = _run(main())
    assert outcome.cancelled and outcome.decision == "none"


# --- T1.11 ------------------------------------------------------------------------------------

def test_run_hook_cwd_and_shell(home, tmp_path, monkeypatch):
    out = tmp_path / "cwd.txt"
    entry = _entry(_script(tmp_path, f"import os; open({str(out)!r}, 'w').write(os.getcwd())"))
    assert _run(run_hook(entry, _pre())).error is None
    assert Path(out.read_text()) == project_root()

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    with use_project_root(worktree):
        assert _run(run_hook(entry, _pre())).error is None
    assert Path(out.read_text()) == worktree

    monkeypatch.setattr(bash_module, "select_shell", lambda: ("bash", ["definitely-not-a-real-shell", "-c"]))
    result = _run(run_hook(entry, _pre()))
    assert result.error and result.decision == "none"


# --- T1.12 ------------------------------------------------------------------------------------

CLAUDE_CODE_SCRIPT = """
    import json, sys
    data = json.load(sys.stdin)
    command = data["tool_input"].get("command", "")
    decision = "deny" if "rm -rf" in command else "allow"
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": f"policy: {decision}",
    }}))
"""


def test_claude_code_script_contract(home, tmp_path):
    # the block as it appears in a Claude Code settings.json, with only the matcher renamed (D11)
    settings_json = {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [
        {"type": "command", "command": _script(tmp_path, CLAUDE_CODE_SCRIPT)}]}]}}
    settings_json["hooks"]["PreToolUse"][0]["matcher"] = "bash_command"
    write_user_config(home, settings_json)
    assert _run(run_hooks("PreToolUse", _pre("bash_command", {"command": "rm -rf /"}))).decision == "deny"
    outcome = _run(run_hooks("PreToolUse", _pre("bash_command", {"command": "ls"})))
    assert outcome.decision == "allow" and outcome.reason == "policy: allow"

    # edit on disk + reload: the next dispatch uses the new command, no restart
    path = home / ".dostuff" / "config.yaml"
    path.write_text(yaml.safe_dump(_hooks_block("PreToolUse", _group(_cmd("exit 2")))), encoding="utf-8")
    get_config().reload()
    assert _run(run_hooks("PreToolUse", _pre("bash_command", {"command": "ls"}))).decision == "deny"


def test_config_file_edit_applies_without_reload(home, proj):
    """A config.yaml written by edit_file or a shell (no edit_config, no reload()) applies on the next dispatch."""
    path = write_project_config(proj, _hooks_block("PreToolUse", _group(_cmd("exit 2"))))
    write_user_config(home, {"hooks": {"project_trust": "allow"}})
    assert _run(run_hooks("PreToolUse", _pre("bash_command", {"command": "ls"}))).decision == "deny"

    path.write_text(yaml.safe_dump({"hooks": {}}) + "# hook removed\n", encoding="utf-8")
    assert _run(run_hooks("PreToolUse", _pre("bash_command", {"command": "ls"}))).decision == "none"


def test_config_file_broken_edit_keeps_loaded_hooks(home):
    """YAML that doesn't parse keeps the loaded config, and the next good write applies."""
    path = write_user_config(home, _hooks_block("PreToolUse", _group(_cmd("exit 2"))))
    assert _run(run_hooks("PreToolUse", _pre("bash_command", {"command": "ls"}))).decision == "deny"

    path.write_text("hooks: [unclosed\n", encoding="utf-8")
    assert get_config().reload_if_changed() is False
    assert _run(run_hooks("PreToolUse", _pre("bash_command", {"command": "ls"}))).decision == "deny"

    path.write_text(yaml.safe_dump({"hooks": {}}), encoding="utf-8")
    assert get_config().reload_if_changed() is True
    assert _run(run_hooks("PreToolUse", _pre("bash_command", {"command": "ls"}))).decision == "none"
    assert get_config().load_error == ""


def test_config_file_broken_edit_records_error(home):
    """reload_if_changed() says why it kept the loaded config, so the TUI can tell the user."""
    path = write_user_config(home, {"permissions": {"mode": "ask"}})
    get_config()
    path.write_text("permissions: [unclosed\n", encoding="utf-8")
    assert get_config().reload_if_changed() is False
    assert str(path) in get_config().load_error


def test_permissions_edited_by_hand_apply_to_next_tool_call(home):
    """permission_config() reads a config.yaml changed mid-session, not the one loaded at startup."""
    from dostuff.agent.run_tool import permission_config
    path = write_user_config(home, {"permissions": {"mode": "ask"}})
    assert permission_config().mode == "ask"
    path.write_text(yaml.safe_dump({"permissions": {"mode": "plan", "allow": ["bash_command(ls)"]}}), encoding="utf-8")
    cfg = permission_config()
    assert (cfg.mode, cfg.allow) == ("plan", ["bash_command(ls)"])


# === Phase 2 — tool events ===================================================================

from types import SimpleNamespace

from dostuff.agent import loop as loop_module
from dostuff.agent import run_tool as run_tool_module
from dostuff.agent.permissions import PermissionConfig, SessionPermissions
from dostuff.agent.run_tool import (
    CANCELLED_RESULT, authorize_batch, authorize_tool, post_tool_use, pre_tool_use, run_tool,
)


def _stderr_exit2(tmp_path, reason: str) -> str:
    return _script(tmp_path, f"import sys; sys.stderr.write({reason!r}); sys.exit(2)")


def _mode(monkeypatch, mode: str) -> None:
    monkeypatch.setattr(run_tool_module, "permission_config", lambda: PermissionConfig(mode=mode))


class DispatchSpy:
    def __init__(self, result="tool output", delay: float = 0.0):
        self.calls: list[tuple[str, dict]] = []
        self.result = result
        self.delay = delay

    async def __call__(self, name, args, **kwargs):
        self.calls.append((name, dict(args)))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.result


class AskAdapter(ScriptedAdapter):
    """Records prompts; with no answers queued, an ask fails the test."""
    def __init__(self, answers=None):
        super().__init__(answers)
        self.prompts: list[str] = []
        self.batches: list[list[str]] = []

    async def ask(self, message, **kwargs):
        self.prompts.append(message)
        assert self.answers, f"unexpected approval prompt: {message}"
        return self.answers.pop(0)


class BatchAskAdapter(AskAdapter):
    async def ask_batch(self, messages, always_label=""):
        self.batches.append(list(messages))
        return self.answers.pop(0)


def _tool_kwargs(**kw):
    return {"session_id": "s1", "turn_id": "t1", **kw}


# --- T2.1 -------------------------------------------------------------------------------------

def test_pre_tool_use_deny(home, tmp_path, monkeypatch):
    write_user_config(home, _hooks_block("PreToolUse", _group(_stderr_exit2(tmp_path, "no pushes to main"),
                                                              matcher="bash_command")))
    spy = DispatchSpy()
    monkeypatch.setattr(run_tool_module, "dispatch_tool", spy)
    adapter = AskAdapter()
    result = _run(run_tool("bash_command", {"command": "git push origin main"}, adapter=adapter, **_tool_kwargs()))
    assert result == "no pushes to main"
    assert spy.calls == [] and adapter.prompts == []

    args, outcome = _run(pre_tool_use("bash_command", {"command": "ls"}, session_id="s1", turn_id="t1",
                                      turn_type="interactive_loop", adapter=adapter))
    assert outcome.decision == "deny" and args == {"command": "ls"}


# --- T2.2 -------------------------------------------------------------------------------------

@pytest.mark.parametrize("case, mode, tool, args, hook, asks, expect", [
    ("plan beats hook allow", "plan", "write_file", {"path": "a.txt", "content": "x"},
     HookOutcome(decision="allow"), 0, "Plan mode"),
    ("hook deny in auto", "auto", "read_file", {"path": "a.txt"},
     HookOutcome(decision="deny", reason="nope"), 0, "nope"),
    ("settings still asks", "auto", "edit_config", {"old_str": "", "new_str": "x: 1\n"},
     HookOutcome(decision="allow"), 1, "User denied"),
    ("hook ask in auto", "auto", "read_file", {"path": "a.txt"},
     HookOutcome(decision="ask", reason="check this"), 1, "User denied"),
    ("hook allow skips prompt", "ask", "bash_command", {"command": "rm -rf build"},
     HookOutcome(decision="allow"), 0, None),
    ("rewrite drops allow", "ask", "write_file", {"path": "a.txt", "content": "x"},
     HookOutcome(decision="allow", updated_input={"content": "y"}), 1, "User denied"),
    ("no decision", "ask", "write_file", {"path": "a.txt", "content": "x"},
     HookOutcome(), 1, "User denied"),
])
def test_hook_permission_precedence(home, monkeypatch, case, mode, tool, args, hook, asks, expect):
    _mode(monkeypatch, mode)
    adapter = AskAdapter(["no"] * asks)
    result = _run(authorize_tool(tool, args, adapter=adapter, hook=hook))
    assert len(adapter.prompts) == asks, case
    if expect is None:
        assert result is None, case
    else:
        assert result is not None and result.startswith(expect), (case, result)
    if hook.decision == "ask":
        assert "check this" in adapter.prompts[0]
    # with no hook decision the permission layer behaves exactly as without hooks
    if hook.decision == "none" and hook.updated_input is None:
        bare = AskAdapter(["no"] * asks)
        assert _run(authorize_tool(tool, args, adapter=bare, hook=None)) == result
        assert bare.prompts == adapter.prompts


def test_hook_precedence_through_run_tool(home, tmp_path, monkeypatch):
    # plan mode: a real hook that allows everything cannot loosen it
    write_user_config(home, _hooks_block("PreToolUse", _decide(tmp_path, "allow")))
    spy = DispatchSpy()
    monkeypatch.setattr(run_tool_module, "dispatch_tool", spy)
    result = _run(run_tool("write_file", {"path": "a.txt", "content": "x"}, adapter=AskAdapter(),
                           permissions=SessionPermissions(mode="plan"), **_tool_kwargs()))
    assert result.startswith("Plan mode") and spy.calls == []
    # ask mode: the same allow runs a mutating command without a prompt
    _mode(monkeypatch, "ask")
    assert _run(run_tool("bash_command", {"command": "rm -rf build"}, adapter=AskAdapter(), **_tool_kwargs())) == "tool output"
    assert spy.calls == [("bash_command", {"command": "rm -rf build"})]


def test_authorize_batch_honours_hook_ask(home, monkeypatch):
    _mode(monkeypatch, "auto")
    calls = [("read_file", {"path": "a"}), ("read_file", {"path": "b"})]
    ask, deny = HookOutcome(decision="ask", reason="look"), HookOutcome(decision="deny", reason="blocked")

    # one ask + one deny: nothing to batch; per call the deny never prompts, the ask does
    adapter = BatchAskAdapter(["no"])
    assert _run(authorize_batch(calls, adapter=adapter, hooks=[ask, deny])) is None
    assert _run(authorize_tool(*calls[1], adapter=adapter, hook=deny)) == "blocked"
    assert adapter.prompts == []
    assert _run(authorize_tool(*calls[0], adapter=adapter, hook=ask)).startswith("User denied")
    assert len(adapter.prompts) == 1 and adapter.batches == []

    # two asks share one batch prompt
    adapter = BatchAskAdapter(["yes"])
    assert _run(authorize_batch(calls, adapter=adapter, hooks=[ask, ask])) == [None, None]
    assert len(adapter.batches) == 1 and len(adapter.batches[0]) == 2 and adapter.prompts == []

    # without hooks, auto mode never prompts
    assert _run(authorize_batch(calls, adapter=BatchAskAdapter())) is None


# --- T2.3 -------------------------------------------------------------------------------------

def test_pre_tool_use_rewrite(home, tmp_path, monkeypatch):
    rewrite = {"_confirmed": True, "overwrite": True, "adapter": "x", "command": "git push --force"}
    write_user_config(home, _hooks_block("PreToolUse", _decide(tmp_path, updated=rewrite)))
    spy = DispatchSpy()
    monkeypatch.setattr(run_tool_module, "dispatch_tool", spy)
    _mode(monkeypatch, "ask")

    args, outcome = _run(pre_tool_use("bash_command", {"command": "git status"}, session_id="s1", turn_id="t1",
                                      turn_type="interactive_loop", adapter=None))
    assert args == {"command": "git push --force"} and outcome.decision == "none"

    # `git status` would be auto-allowed; the rewritten command faces the prompt, and "no" denies it
    adapter = AskAdapter(["no"])
    result = _run(run_tool("bash_command", {"command": "git status"}, adapter=adapter, **_tool_kwargs()))
    assert result.startswith("User denied") and "git push --force" in adapter.prompts[0]
    assert spy.calls == []
    # "yes" runs the rewritten call, without the smuggled keys
    adapter = AskAdapter(["yes"])
    _run(run_tool("bash_command", {"command": "git status"}, adapter=adapter, **_tool_kwargs()))
    assert spy.calls == [("bash_command", {"command": "git push --force"})]
    # plan mode refuses the rewrite like any model call
    result = _run(run_tool("bash_command", {"command": "git status"}, adapter=AskAdapter(),
                           permissions=SessionPermissions(mode="plan"), **_tool_kwargs()))
    assert result.startswith("Plan mode") and len(spy.calls) == 1


# --- T2.4 -------------------------------------------------------------------------------------

def test_pre_tool_use_failure_paths(home, tmp_path, monkeypatch):
    crash = _script(tmp_path, "import sys; sys.exit(1)")
    spy = DispatchSpy()
    monkeypatch.setattr(run_tool_module, "dispatch_tool", spy)
    _mode(monkeypatch, "auto")

    write_user_config(home, _hooks_block("PreToolUse", _group(crash, name="g")))
    adapter = AskAdapter()
    assert _run(run_tool("read_file", {"path": "a"}, adapter=adapter, **_tool_kwargs())) == "tool output"
    assert len(spy.calls) == 1
    assert any(t == "error" and "hook 'g'" in str(d) for t, d in adapter.events)

    write_user_config(home, _hooks_block("PreToolUse", _group(crash, name="g", fail_closed=True)))
    assert _run(run_tool("read_file", {"path": "a"}, adapter=AskAdapter(), **_tool_kwargs())) == "hook 'g' failed"
    assert len(spy.calls) == 1

    write_user_config(home, _hooks_block("PreToolUse", _group(_script(tmp_path, "import time; time.sleep(5)"))))
    adapter = AskAdapter()

    async def main():
        task = asyncio.create_task(run_tool("read_file", {"path": "a"}, adapter=adapter, **_tool_kwargs()))
        await asyncio.sleep(0.3)
        adapter.cancel()
        return await task

    assert _run(main()) == CANCELLED_RESULT
    assert len(spy.calls) == 1


# --- T2.5 -------------------------------------------------------------------------------------

def _post(tool, args, result, adapter=None):
    return _run(post_tool_use(tool, args, result, session_id="s1", turn_id="t1", turn_type="interactive_loop",
                              adapter=adapter))


def test_post_tool_use_context(home, tmp_path):
    echo_diff = _script(tmp_path, "print('--- a\\n+++ b')")
    write_user_config(home, _hooks_block("PostToolUse", _group(echo_diff, name="format")))
    result, outcome = _post("write_file", {"path": "a.py"}, "File written")
    assert result == 'File written\n<hook name="format">--- a\n+++ b</hook>'
    assert outcome.context == "--- a\n+++ b"

    write_user_config(home, _hooks_block("PostToolUse", _group(echo_diff)))
    assert _post("write_file", {"path": "a.py"}, "ok")[0].endswith('<hook name="PostToolUse#1">--- a\n+++ b</hook>')

    # truncated after concatenation: the hook's block survives at the tail
    cap = get_config().max_tool_output_chars
    big, _ = _post("write_file", {"path": "a.py"}, "x" * 100_000)
    assert len(big) < cap + 500 and big.endswith("+++ b</hook>")

    write_user_config(home, _hooks_block("PostToolUse", _group(_stderr_exit2(tmp_path, "lint failed"), name="lint")))
    result, outcome = _post("write_file", {"path": "a.py"}, "ok")
    assert result == 'ok\n<hook name="lint" status="blocked">lint failed</hook>'

    write_user_config(home, _hooks_block("PostToolUse", _group(_script(tmp_path, "import time; time.sleep(5)"))))
    adapter = ScriptedAdapter()

    async def main():
        task = asyncio.create_task(post_tool_use("write_file", {"path": "a.py"}, "ok", session_id="s1",
                                                 turn_id="t1", turn_type="interactive_loop", adapter=adapter))
        await asyncio.sleep(0.3)
        adapter.cancel()
        return await task

    result, outcome = _run(main())
    assert result == "ok" and outcome.cancelled


def test_post_tool_use_payload_has_tool_response(home, tmp_path):
    seen = tmp_path / "seen.json"
    write_user_config(home, _hooks_block("PostToolUse", _group(_script(tmp_path, f"""
        import json, sys
        json.dump(json.load(sys.stdin), open({str(seen)!r}, "w"))
    """))))
    result, _ = _post("read_file", {"path": "a"}, "contents")
    assert result == "contents"
    payload = json.loads(seen.read_text())
    assert payload["tool_name"] == "read_file" and payload["tool_input"] == {"path": "a"}
    assert payload["tool_response"] == "contents" and payload["hook_event_name"] == "PostToolUse"


# --- T2.6 / T2.7 / T2.9 — loop wiring ---------------------------------------------------------

def _msg(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))], usage=None,
    )


def _tc(name, args, call_id):
    raw = args if isinstance(args, str) else json.dumps(args)
    tc = SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=raw))
    tc.model_dump = lambda: {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}
    return tc


def _loop(monkeypatch, responses, *, dispatch=None, adapter=None, permissions=None, turn_type="interactive_loop",
          max_iterations=5, on_call=None, instructions=None, working=None, **loop_kw):
    """Run loop() against scripted model responses; `on_call(n)` runs before the n-th model call returns,
    `instructions` collects each call's system instruction."""
    model_calls: list[list[dict]] = []

    async def fake_call_agent(steps_history, system_instruction, tool_names=None, stream=False):
        model_calls.append([dict(s) for s in steps_history])
        if instructions is not None:
            instructions.append(system_instruction)
        if on_call is not None:
            on_call(len(model_calls))
        return responses.pop(0), None

    monkeypatch.setattr(loop_module, "call_agent", fake_call_agent)
    monkeypatch.setattr(loop_module, "get_model_token_limit", lambda: 100_000)
    if dispatch is not None:
        monkeypatch.setattr(loop_module, "dispatch_tool", dispatch)
    working = working if working is not None else []
    result = _run(loop_module.loop(
        session_id="s1", turn_id="t1", user_text="hi", dynamic_system_instruction="sys", mcp_client=None,
        working_history=working, turn_type=turn_type, adapter=adapter or ScriptedAdapter(), stream=False,
        max_iterations=max_iterations, permissions=permissions, **loop_kw,
    ))
    return result, working, model_calls


def _tool_steps(working):
    return {s["tool_call_id"]: s["content"] for s in working if s.get("role") == "tool"}


def test_loop_denied_call_never_dispatches(home, tmp_path, monkeypatch):
    write_user_config(home, _hooks_block("PreToolUse", _group(
        _stderr_exit2(tmp_path, "no pushes"), matcher="bash_command", **{"if": "git\\s+push"})))
    order: list[str] = []
    batches: list[list[str]] = []
    real_pre, real_batch = loop_module.pre_tool_use, loop_module.authorize_batch

    async def pre_spy(name, args, **kw):
        order.append(f"pre:{name}")
        return await real_pre(name, args, **kw)

    async def batch_spy(calls, **kw):
        order.append("batch")
        batches.append([name for name, _ in calls])
        return await real_batch(calls, **kw)

    monkeypatch.setattr(loop_module, "pre_tool_use", pre_spy)
    monkeypatch.setattr(loop_module, "authorize_batch", batch_spy)
    spy = DispatchSpy()
    responses = [_msg(tool_calls=[_tc("bash_command", {"command": "git push"}, "c1"),
                                  _tc("read_file", {"path": "a.txt"}, "c2")]), _msg(content="done")]
    result, working, _ = _loop(monkeypatch, responses, dispatch=spy)
    assert result == "done"
    assert spy.calls == [("read_file", {"path": "a.txt"})]
    assert batches == [["read_file"]]
    assert _tool_steps(working) == {"c1": "no pushes", "c2": "tool output"}
    assert order == ["pre:bash_command", "pre:read_file", "batch"]


def test_loop_post_hook_context_reaches_model(home, tmp_path, monkeypatch):
    log = tmp_path / "post.log"
    logger = _script(tmp_path, f"""
        import json, sys
        data = json.load(sys.stdin)
        open({str(log)!r}, "a").write(data["tool_name"] + "|" + data["tool_response"] + "\\n")
        print("checked " + data["tool_name"])
    """)
    write_user_config(home, {"permission_mode": "ask", "hooks": {
        "PreToolUse": [_group(_stderr_exit2(tmp_path, "hook says no"), matcher="delete_file")],
        "PostToolUse": [_group(logger, name="check")],
    }})
    responses = [_msg(tool_calls=[_tc("read_file", {"path": "a.txt"}, "c1")]), _msg(content="done")]
    _, working, model_calls = _loop(monkeypatch, responses, dispatch=DispatchSpy("file body"))
    expected = 'file body\n<hook name="check">checked read_file</hook>'
    assert _tool_steps(working) == {"c1": expected}
    assert any(s.get("content") == expected for s in model_calls[1])

    # parse error, plan-mode denial, hook denial: PostToolUse never fires
    log.unlink()
    responses = [_msg(tool_calls=[_tc("read_file", "{bad json", "c1"), _tc("write_file", {"path": "a", "content": "x"}, "c2"),
                                  _tc("delete_file", {"path": "a"}, "c3")]), _msg(content="done")]
    spy = DispatchSpy()
    _, working, _ = _loop(monkeypatch, responses, dispatch=spy, permissions=SessionPermissions(mode="plan"))
    assert spy.calls == [] and not log.exists()
    steps = _tool_steps(working)
    assert steps["c2"].startswith("Plan mode") and steps["c3"] == "hook says no"

    # timed out: fires with the timeout text
    monkeypatch.setattr(loop_module, "TOOL_TIMEOUT_S", 0.2)
    responses = [_msg(tool_calls=[_tc("read_file", {"path": "a"}, "c1")]), _msg(content="done")]
    _loop(monkeypatch, responses, dispatch=DispatchSpy(delay=2))
    assert log.read_text().splitlines() == ["read_file|Error: tool 'read_file' timed out after 0s."]

    # cancelled while running: no PostToolUse
    log.unlink()
    monkeypatch.setattr(loop_module, "TOOL_TIMEOUT_S", 120.0)
    adapter = ScriptedAdapter()

    class CancellingSpy(DispatchSpy):
        async def __call__(self, name, args, **kwargs):
            adapter.cancel()
            await asyncio.sleep(5)

    responses = [_msg(tool_calls=[_tc("read_file", {"path": "a"}, "c1")])]
    result, working, _ = _loop(monkeypatch, responses, dispatch=CancellingSpy(), adapter=adapter)
    assert result == "Cancelled by user." and not log.exists()
    assert _tool_steps(working)["c1"].startswith(CANCELLED_RESULT)


def test_hooks_outside_tool_timeout(home, tmp_path, monkeypatch):
    slow = _script(tmp_path, "import time; time.sleep(0.4); print('slow ok')")
    write_user_config(home, {"hooks": {"PreToolUse": [_group(slow)], "PostToolUse": [_group(slow, name="slow")]}})
    monkeypatch.setattr(loop_module, "TOOL_TIMEOUT_S", 0.3)
    responses = [_msg(tool_calls=[_tc("read_file", {"path": "a"}, "c1")]), _msg(content="done")]
    _, working, _ = _loop(monkeypatch, responses, dispatch=DispatchSpy("fast"))
    assert _tool_steps(working) == {"c1": 'fast\n<hook name="slow">slow ok</hook>'}


def test_plan_mode_interaction(home, tmp_path, monkeypatch):
    log = tmp_path / "post.log"
    write_user_config(home, {"hooks": {
        "PreToolUse": [_decide(tmp_path, "allow")],
        "PostToolUse": [_group(_logger(tmp_path, log, "post"))],
    }})
    spy = DispatchSpy()
    responses = [_msg(tool_calls=[_tc("write_file", {"path": "a", "content": "x"}, "c1"),
                                  _tc("read_file", {"path": "a"}, "c2")]), _msg(content="done")]
    _, working, _ = _loop(monkeypatch, responses, dispatch=spy, permissions=SessionPermissions(mode="plan"))
    steps = _tool_steps(working)
    assert steps["c1"].startswith("Plan mode") and steps["c2"].startswith("tool output")
    assert spy.calls == [("read_file", {"path": "a"})]
    assert _log_lines(log) == ["post"]


# --- T2.8 -------------------------------------------------------------------------------------

def test_hook_event_rendering(home, tmp_path):
    write_user_config(home, _hooks_block("PreToolUse", _group(_stderr_exit2(tmp_path, "no pushes"))))
    adapter = ScriptedAdapter()
    _run(pre_tool_use("bash_command", {"command": "git push"}, session_id="s1", turn_id="t1",
                      turn_type="interactive_loop", adapter=adapter))
    hook_events = [d for t, d in adapter.events if t == "hook"]
    assert hook_events == [{"event": "PreToolUse", "tool": "bash_command", "decision": "deny", "reason": "no pushes"}]

    from dostuff.cli_tui import TuiAdapter
    appended: list[tuple[str, str]] = []
    app = SimpleNamespace(_append=lambda text, msg_type="agent": appended.append((text, msg_type)),
                          ui=lambda fn: fn())
    tui = TuiAdapter(app)
    tui.emit("hook", hook_events[0])
    tui.emit("error", "hook 'g' exited with code 1")
    assert appended == [("⚓ PreToolUse bash_command → deny (no pushes)", "system"),
                        ("hook 'g' exited with code 1", "error")]

    from dostuff.headless import HeadlessAdapter
    lines: list[dict] = []
    headless = HeadlessAdapter(interactive=False, quiet=True, stream=lines.append)
    headless.emit("hook", hook_events[0])
    assert lines == [{"type": "hook", "event": "PreToolUse", "tool": "bash_command", "decision": "deny",
                      "reason": "no pushes"}]


# === Phase 3 — turn events ===================================================================

from dostuff.helpers.hooks import events as events_module
from dostuff.helpers.hooks.dispatch import hook_line


def _ups(prompt, adapter=None):
    return _run(events_module.user_prompt_submit(prompt, session_id="s1", turn_id="t1", adapter=adapter))


def _event_logger(tmp_path, log, *fields):
    """A hook that appends `field|field|…` from its stdin payload to `log` and exits 0 silently."""
    keys = list(fields)
    return _script(tmp_path, f"""
        import json, sys
        data = json.load(sys.stdin)
        open({str(log)!r}, "a").write("|".join(str(data.get(k)) for k in {keys!r}) + "\\n")
    """)


# --- T3.1 -------------------------------------------------------------------------------------

def test_user_prompt_submit_helper(home, tmp_path):
    assert _ups("hi") == (None, "")  # no hooks: nothing happens

    write_user_config(home, _hooks_block("UserPromptSubmit", _group(_stderr_exit2(tmp_path, "not in this repo please"))))
    assert _ups("hi") == ("not in this repo please", "")

    write_user_config(home, _hooks_block("UserPromptSubmit", _decide(tmp_path, context="team rules: be brief")))
    assert _ups("hi") == (None, "team rules: be brief")
    write_user_config(home, _hooks_block("UserPromptSubmit", _group(_script(tmp_path, "print('plain notes')"))))
    assert _ups("hi") == (None, "plain notes")  # plain stdout is context for this event (D3)

    crash = _script(tmp_path, "import sys; sys.exit(1)")
    write_user_config(home, _hooks_block("UserPromptSubmit", _group(crash, name="guard", fail_closed=True)))
    assert _ups("hi") == ("hook 'guard' failed", "")
    write_user_config(home, _hooks_block("UserPromptSubmit", _group(crash, name="guard")))
    adapter = ScriptedAdapter()
    assert _ups("hi", adapter) == (None, "")
    assert any(t == "error" and "hook 'guard'" in str(d) for t, d in adapter.events)

    log = tmp_path / "ups.log"
    write_user_config(home, _hooks_block("UserPromptSubmit", _group(_event_logger(tmp_path, log, "hook_event_name", "prompt"))))
    _ups("fix the tests")
    assert log.read_text().splitlines() == ["UserPromptSubmit|fix the tests"]

    assert events_module.blocked_prompt_message("nope") == "Prompt blocked by a UserPromptSubmit hook: nope"


# --- T3.2 -------------------------------------------------------------------------------------

def _always_blocking_stop(tmp_path, log):
    return _script(tmp_path, f"""
        import json, sys
        data = json.load(sys.stdin)
        open({str(log)!r}, "a").write(json.dumps(data["stop_hook_active"]) + "|" + data["last_assistant_message"] + "\\n")
        sys.stderr.write("tests are red")
        sys.exit(2)
    """)


def test_stop_block_continues_then_stops(home, tmp_path, monkeypatch):
    log = tmp_path / "stop.log"
    write_user_config(home, {"hooks": {"stop_block_limit": 3,
                                       "Stop": [_group(_always_blocking_stop(tmp_path, log), name="tests")]}})
    adapter = ScriptedAdapter()
    responses = [_msg(content=f"answer {i}") for i in range(4)]
    result, working, model_calls = _loop(monkeypatch, responses, adapter=adapter, max_iterations=10)
    assert result == "answer 3" and len(model_calls) == 4
    assert [s["content"] for s in working if s["role"] == "user"] == ["[Stop hook tests] tests are red"] * 3
    # no extra keys on the step: steps go to the provider verbatim
    assert all(set(s) == {"role", "content"} for s in working if s["role"] == "user")
    assert log.read_text().splitlines() == ["false|answer 0", "true|answer 1", "true|answer 2", "true|answer 3"]
    assert any(t == "system" and "stop_block_limit" in str(d) for t, d in adapter.events)
    assert ("hook", {"event": "Stop", "tool": "", "decision": "block", "reason": "tests are red"}) in adapter.events

    # a Stop hook that lets the turn end: one call, no note
    write_user_config(home, _hooks_block("Stop", _group(_cmd("exit 0"))))
    result, working, model_calls = _loop(monkeypatch, [_msg(content="done")])
    assert result == "done" and len(model_calls) == 1 and [s["role"] for s in working] == ["assistant"]


def test_stop_not_fired_on_abnormal_end(home, tmp_path, monkeypatch):
    log = tmp_path / "stop.log"
    write_user_config(home, _hooks_block("Stop", _group(_always_blocking_stop(tmp_path, log), name="tests")))

    # cancelled during the model call
    adapter = ScriptedAdapter()
    result, _, _ = _loop(monkeypatch, [_msg(content="x")], adapter=adapter, on_call=lambda n: adapter.cancel())
    assert result == "Cancelled by user." and not log.exists()

    # model error
    def boom(n):
        raise ValueError("boom")
    result, _, _ = _loop(monkeypatch, [_msg(content="x")], on_call=boom)
    assert result.startswith("Model call failed") and not log.exists()

    # the empty-response give-up
    result, _, _ = _loop(monkeypatch, [_msg(), _msg()])
    assert "empty response" in result and not log.exists()

    # max_iterations still applies across continuations: the turn ends at the cap
    result, _, model_calls = _loop(monkeypatch, [_msg(content="a"), _msg(content="b")], max_iterations=2)
    assert result.startswith("Reached maximum iterations (2)") and len(model_calls) == 2
    assert len(log.read_text().splitlines()) == 2  # one per final answer, none for hitting the cap


# --- T3.3 -------------------------------------------------------------------------------------

def test_subagent_events(home, tmp_path, monkeypatch):
    log = tmp_path / "events.log"
    logger = _event_logger(tmp_path, log, "hook_event_name", "turn_type")
    write_user_config(home, {"hooks": {event: [_group(logger)] for event in (
        "PreToolUse", "PostToolUse", "Stop", "SubagentStop", "SessionStart", "UserPromptSubmit")}})
    responses = [_msg(tool_calls=[_tc("read_file", {"path": "a"}, "c1")]), _msg(content="done")]
    result, _, _ = _loop(monkeypatch, responses, dispatch=DispatchSpy(), turn_type="subagent_loop")
    assert result == "done"
    assert log.read_text().splitlines() == ["PreToolUse|subagent_loop", "PostToolUse|subagent_loop",
                                            "SubagentStop|subagent_loop"]

    log.unlink()
    _loop(monkeypatch, [_msg(content="done")])
    assert log.read_text().splitlines() == ["Stop|interactive_loop"]
    assert events_module.stop_event_for("subagent_loop") == "SubagentStop"
    assert events_module.stop_event_for("interactive_loop") == "Stop"

    # hook lines from a subagent carry its role, like the rest of its output
    from dostuff.tools.delegate_to_subagent import _SubagentAdapter
    parent = ScriptedAdapter()
    _SubagentAdapter(parent, "reviewer").emit("hook", {"event": "PreToolUse", "tool": "bash_command",
                                                       "decision": "deny", "reason": "no"})
    assert hook_line(parent.events[0][1]) == "[subagent: reviewer] ⚓ PreToolUse bash_command → deny (no)"


# --- T3.4 -------------------------------------------------------------------------------------

def test_session_start_context_is_session_state(home, tmp_path, monkeypatch):
    holder = {"startup": "branch: x"}
    instructions: list[str] = []
    working: list = []
    for _ in range(2):  # two consecutive turns
        _loop(monkeypatch, [_msg(tool_calls=[_tc("read_file", {"path": "a"}, "c1")]), _msg(content="done")],
              dispatch=DispatchSpy(), instructions=instructions, working=working, session_hook_context=holder)
    assert len(instructions) == 4 and all("branch: x" in i for i in instructions)
    assert not any("branch: x" in str(s.get("content")) for s in working)

    # a compaction fires SessionStart(compact) and stores its context under its own key
    write_user_config(home, _hooks_block("SessionStart", _group(_script(tmp_path, "print('recompacted ctx')"),
                                                                matcher="compact")))

    async def fake_compact(history, keep_budget, notes, token_limit=None):
        return history[-1:], "the summary"

    monkeypatch.setattr(loop_module, "compact_context", fake_compact)
    instructions.clear()
    _loop(monkeypatch, [_msg(content="done")], instructions=instructions, working=working,
          session_hook_context=holder, last_input_tokens=90_000)
    assert holder == {"startup": "branch: x", "compact": "recompacted ctx"}
    assert "[Summary of earlier conversation]: the summary" in instructions[0]
    assert "recompacted ctx" in instructions[0] and "branch: x" in instructions[0]
    assert "recompacted ctx" not in instructions[0].split("[Summary of earlier conversation]: ")[1].split("\n\n")[0]

    # startup/resume/clear sources are fired by the callers through the helper
    log = tmp_path / "ss.log"
    write_user_config(home, _hooks_block("SessionStart", _group(_script(tmp_path, f"""
        import json, sys
        source = json.load(sys.stdin)["source"]
        open({str(log)!r}, "a").write(source + "\\n")
        print("ctx-" + source)
    """), matcher="startup|resume|clear")))
    holder = {}
    _run(events_module.session_start("startup", holder, session_id="s1"))
    _run(events_module.session_start("compact", holder, session_id="s1"))  # matcher doesn't cover it
    assert holder == {"startup": "ctx-startup"} and log.read_text().split() == ["startup"]
    assert events_module.SESSION_START_SOURCES == ("startup", "resume", "clear", "compact")


# --- T4.1 -------------------------------------------------------------------------------------

@pytest.mark.parametrize("block", [None, "absent", "text", {"PreToolUse": 5}])
@pytest.mark.parametrize("which", ["user", "project"])
def test_config_hooks_parse(home, proj, block, which):
    write = write_user_config if which == "user" else write_project_config
    target = home if which == "user" else proj
    write(target, {"model": "openai/x"} if block == "absent" else {"hooks": block})
    cfg = get_config()
    assert cfg.hooks == {}
    assert cfg.hooks_project_trust == "ask" and cfg.hooks_stop_block_limit == 3
    if block in ("text", {"PreToolUse": 5}):
        assert len(cfg.hooks_warnings) == 1 and f"{which} config" in cfg.hooks_warnings[0]
    else:
        assert cfg.hooks_warnings == []


@pytest.mark.parametrize("value, expected", [("allow", "allow"), ("DENY", "deny"), ("ask", "ask"), ("sure", "ask")])
def test_config_hooks_project_trust(home, value, expected):
    write_user_config(home, {"hooks": {"project_trust": value}})
    cfg = get_config()
    assert cfg.hooks_project_trust == expected
    assert bool(cfg.hooks_warnings) == (value == "sure")
    write_user_config(home, {"hooks": {"project_trust": value, "PreToolUse": [_group("echo x")]}})
    assert get_config().hooks_project_trust == expected
    # reload() re-parses the block
    write_user_config(home, {"hooks": {"project_trust": "deny"}})
    get_config().reload()
    assert get_config().hooks_project_trust == "deny"


# --- T4.2 -------------------------------------------------------------------------------------

def test_hooks_config_key(home, tmp_path):
    from dostuff.tools import settings
    first, second = _script(tmp_path, "print('one')"), _script(tmp_path, "print('two')")
    block = yaml.safe_dump(_hooks_block("PostToolUse", _group(first, name="fmt")), width=10_000)
    result = settings.edit_config("", block)
    assert "unknown top-level key 'hooks'" not in result and "Warnings" not in result
    assert [e.command for e in get_config().hooks["PostToolUse"]] == [first]

    # replacing a hook command applies without a restart (edit_config reloads)
    script = lambda cmd: cmd.rsplit("/", 1)[1].rstrip('"')  # noqa: E731  (hookN.py)
    settings.edit_config(script(first), script(second))
    assert [e.command for e in get_config().hooks["PostToolUse"]] == [second]

    # an invalid hook edit is reported with the loader's own warning
    result = settings.edit_config("PostToolUse", "PostTool")
    assert "hooks.PostTool" in result and "unknown event" in result

    text = settings.read_settings()
    assert "unknown top-level key" not in text
    reference = text.split("## Reference")[1]
    for phrase in ("hooks", "not permission-checked", "trust", ".dostuff/hooks/"):
        assert phrase in reference, phrase


# --- T4.3 -------------------------------------------------------------------------------------

class RecordingAdapter(ScriptedAdapter):
    def __init__(self, answers=None):
        super().__init__(answers)
        self.asked: list[str] = []

    async def ask(self, message, **kwargs):
        self.asked.append(message)
        return await super().ask(message, **kwargs)


def test_project_hook_trust(home, proj, tmp_path):
    log = tmp_path / "ran.log"
    script = proj / ".dostuff" / "hooks" / "check.py"
    script.parent.mkdir()
    script.write_text(f"open({str(log)!r}, 'a').write('ran' + chr(10))\n", encoding="utf-8")
    cmd = _python_cmd(script)
    proj_path = write_project_config(proj, _hooks_block("PreToolUse", _group(cmd, name="check")))
    trust_file = home / ".dostuff" / "trusted_hooks.json"

    def runs() -> int:
        return len(log.read_text().splitlines()) if log.exists() else 0

    def dispatch(answers=(), **kw):
        adapter = RecordingAdapter(answers=list(answers))
        outcome = _run(run_hooks("PreToolUse", _pre(), adapter=adapter))
        assert adapter.answers == [], "every scripted answer is used"
        assert not outcome.errors, outcome.errors
        return adapter

    def new_session():
        trust_module._decisions.clear()
        dispatch_module._untrusted_warned.clear()
        reset_config()

    # first project hook: one prompt listing the command and the file it came from
    adapter = dispatch(["yes"])
    (message,) = adapter.asked
    assert "check" in message and cmd in message and str(proj_path) in message
    assert runs() == 1
    record = json.loads(trust_file.read_text(encoding="utf-8"))
    (root, digest), = record.items()
    assert Path(root) == proj.resolve() and len(digest) == 64

    # the same session does not ask again, nor does a later one (the record is on disk)
    assert dispatch().asked == [] and runs() == 2
    new_session()
    assert dispatch().asked == [] and runs() == 3

    # a comment-only edit of the YAML is not a change
    proj_path.write_text("# a comment\n" + proj_path.read_text(encoding="utf-8"), encoding="utf-8")
    new_session()
    assert dispatch().asked == [] and runs() == 4

    # editing the block re-asks
    write_project_config(proj, _hooks_block("PreToolUse", _group(cmd, name="check", timeout=5)))
    new_session()
    assert len(dispatch(["yes"]).asked) == 1 and runs() == 5

    # editing a script in .dostuff/hooks/ re-asks; a denial skips the hook with one notice, and the turn goes on
    script.write_text(script.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    new_session()
    adapter = RecordingAdapter(answers=["no"])
    for _ in range(2):
        outcome = _run(run_hooks("PreToolUse", _pre(), adapter=adapter))
        assert outcome.decision == "none" and not outcome.errors
    assert len(adapter.asked) == 1 and runs() == 5
    assert len([e for e in adapter.events if "project hooks not trusted" in str(e[1])]) == 1

    # a subagent in a worktree reuses the session's answer
    new_session()
    dispatch(["yes"])
    (tmp_path / "worktree").mkdir()
    with use_project_root(tmp_path / "worktree"):
        assert dispatch().asked == []
    assert runs() == 7

    # the user's project_trust: deny → never runs, never asks; allow → runs without asking
    trust_file.unlink()
    write_user_config(home, {"hooks": {"project_trust": "deny"}})
    new_session()
    assert dispatch().asked == [] and runs() == 7
    write_user_config(home, {"hooks": {"project_trust": "allow"}})
    new_session()
    assert dispatch().asked == [] and runs() == 8
    assert not trust_file.exists()

    # no one to ask (adapter=None) → skipped, nothing remembered
    write_user_config(home, {})
    new_session()
    _run(run_hooks("PreToolUse", _pre()))
    assert runs() == 8 and not trust_file.exists()


def test_trust_digest(tmp_path, monkeypatch):
    entries, _ = load_hooks({"Stop": [_group("echo a")]}, source="project", origin="o")
    hooks_dir = tmp_path / "hooks"
    first = trust_module.digest(entries, hooks_dir)
    hooks_dir.mkdir()
    (hooks_dir / "a.py").write_text("x", encoding="utf-8")
    second = trust_module.digest(entries, hooks_dir)
    assert first and second and first != second
    monkeypatch.setattr(trust_module, "HOOKS_DIR_CAP", 0)
    assert trust_module.digest(entries, hooks_dir) is None  # over the cap → always ask


# --- T4.5 -------------------------------------------------------------------------------------

def test_doctor_hooks_check(home, proj, monkeypatch):
    from dostuff.helpers.ui import doctor

    def checks():
        reset_config()
        return [c for c in doctor.hooks_checks()]

    def which(name):
        return None if name == "missingtool" else f"/usr/bin/{name}"
    monkeypatch.setattr(doctor.shutil, "which", which)

    # no hooks: one info line
    (none,) = checks()
    assert none.ok is None and "none" in none.label

    # valid hooks: ok; builtins (`cd x && …`) aren't looked up on PATH
    write_user_config(home, {"hooks": {"PreToolUse": [_group("python check.py", name="a"),
                                                      _group("cd sub && make", name="b")]}})
    (summary,) = checks()
    assert summary.ok is True and "2" in summary.label

    # a bad regex on a normal entry: not ok, with a fix
    write_user_config(home, _hooks_block("PreToolUse", _group("echo x", name="bad", **{"if": "^(?i)git"})))
    bad = [c for c in checks() if c.ok is False]
    assert len(bad) == 1 and "bad" in bad[0].label and bad[0].fix

    # a broken fail_closed entry: not ok, says calls are being denied
    write_user_config(home, _hooks_block("PreToolUse", _group("echo x", name="guard", matcher="[", fail_closed=True)))
    bad = [c for c in checks() if c.ok is False]
    assert len(bad) == 1 and "guard" in bad[0].label and "denies" in bad[0].label

    # a command whose first token isn't on PATH
    write_user_config(home, _hooks_block("Stop", _group("missingtool --check", name="lint")))
    bad = [c for c in checks() if c.ok is False]
    assert len(bad) == 1 and "missingtool" in bad[0].label and "lint" in bad[0].label

    # untrusted project hooks: info; a project script outside .dostuff/hooks/: warning
    write_user_config(home, {})
    write_project_config(proj, {"hooks": {"Stop": [_group("python scripts/check.py", name="outside"),
                                                   _group("python .dostuff/hooks/ok.py", name="inside")]}})
    result = checks()
    info = [c for c in result if c.ok is None]
    assert len(info) == 1 and "not trusted" in info[0].label
    warn = [c for c in result if c.ok is False]
    assert len(warn) == 1 and "outside" in warn[0].label and ".dostuff/hooks/" in warn[0].fix

    # once trusted, no info line
    trust_module.remember(trust_module.trust_key(trust_module.project_entries(get_config().hooks)))
    assert not [c for c in checks() if c.ok is None]

    # doctor_checks includes them
    labels = [c.label for c in doctor.doctor_checks()]
    assert any("outside" in label for label in labels)


# --- T4.6 -------------------------------------------------------------------------------------

def test_example_config_hooks():
    from dostuff.helpers.hooks.load_hooks import EVENTS
    from dostuff.setup_wizard import read_template
    text = read_template("config.example.yaml")
    lines = text.splitlines()
    assert "# hooks:" in lines, "a commented-out hooks block"
    block = lines[lines.index("# hooks:"):]
    assert all(line.startswith("#") for line in block)
    uncommented = yaml.safe_load("\n".join(line[2:] if line.startswith("# ") else "" for line in block))
    raw = uncommented["hooks"]
    entries, warnings = load_hooks(raw, source="user", origin="config.example.yaml")
    assert warnings == []
    assert {e.event for e in entries} == set(EVENTS)  # one example per event
    assert raw["project_trust"] == "ask" and raw["stop_block_limit"] == 3
    for phrase in ("unsandboxed", "project_trust", ".dostuff/hooks/", "--trust-project-hooks"):
        assert phrase in text, phrase
    for e in entries:
        for bad in ("jq", "$(", "^(?i)"):
            assert bad not in e.command and bad not in (e.if_pattern or ""), (e.name, bad)
    # the live template still loads without hook warnings
    assert yaml.safe_load(text).get("hooks") is None


# --- T5.3 — documentation ---------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:end if end != -1 else len(text)]


def test_readme_documents_hooks():
    from dostuff.helpers.hooks.load_hooks import EVENTS
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "\n## Hooks\n" in readme
    hooks = _section(readme, "\n## Hooks\n")
    for event in EVENTS:
        assert f"`{event}`" in hooks, event
    # the exit-code contract: 0 / 2 / anything else
    for row in ("| `0` |", "| `2` |", "| other |"):
        assert row in hooks, row
    # the D11 name table
    for claude, ours in (("Bash", "bash_command"), ("Write", "write_file"), ("Edit", "edit_file"),
                         ("Read", "read_file"), ("mcp__<server>__<tool>", "call_mcp_tool")):
        assert f"| `{claude}` | `{ours}` |" in hooks, claude
    for phrase in ("project_trust", "trusted_hooks.json", "--trust-project-hooks", ".dostuff/hooks/",
                   "unsandboxed", "/hooks", "fail_closed"):
        assert phrase in hooks, phrase
    for script in ("block_main_push.py", "format_file.py"):
        assert f"# .dostuff/hooks/{script}" in hooks, script
    assert "**Hooks**" in _section(readme, "## Features")
    # the config pasteable byte-for-byte from spec D9 (plan/ is gitignored, so this part is local only)
    spec_path = REPO / "plan" / "2026-10-29-hooks" / "spec.md"
    if not spec_path.exists():
        return
    d9 = _section(spec_path.read_text(encoding="utf-8"), "### D9. Config shape")
    d9_yaml = d9[d9.index("```yaml\n"):d9.index("```\n", d9.index("```yaml\n") + 8) + 4]
    assert d9_yaml in hooks
    inner = d9_yaml.removeprefix("```yaml\n").removesuffix("```\n")
    entries, warnings = load_hooks(yaml.safe_load(inner)["hooks"], source="user", origin="README.md")
    assert warnings == [] and entries


@pytest.mark.skipif(not (REPO / "plan").is_dir(), reason="plan/ is gitignored, so local only")
def test_plans_point_at_the_hooks_spec():
    # CHANGELOG.md is written when NOTES.md is folded in at push time, so it isn't checked here
    parity = (REPO / "plan" / "2026-10-28-claude-code-parity" / "plan.md").read_text(encoding="utf-8")
    phase1 = _section(parity.replace("\n### ", "\n## "), "## Phase 1")
    assert "2026-10-29-hooks/spec.md" in phase1
    assert "matching Claude Code's shape" not in phase1  # links instead of restating the design
    roadmap = (REPO / "plan" / "2026-10-26-v0.2.1-roadmap.md").read_text(encoding="utf-8")
    future = _section(roadmap, "## Future")
    assert not any(line.startswith("- **Hooks**") for line in future.splitlines())
    assert "✅ **Hooks**" in roadmap


# --- background shell T4.5: hooks see and can stop the job tools ------------------------------

def test_hooks_bg_tools(home, tmp_path, monkeypatch, registry, _py, _alive):
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    _mode(monkeypatch, "auto")
    assert dispatch_module.SHELL_TOOLS == ("bash_command",)
    seen = tmp_path / "seen.json"
    record = _script(tmp_path, f"import sys; open(r'{seen}', 'w').write(sys.stdin.read())")
    deny_kill = _stderr_exit2(tmp_path, "keep the server up")
    write_user_config(home, {"hooks": {"PreToolUse": [
        _group(record, matcher="bash_command"),
        _group(deny_kill, matcher="kill_shell"),
    ]}})
    sleeper = _py("import time; time.sleep(30)")

    async def scenario():
        args = {"command": sleeper, "run_in_background": True, "background_id": "dev"}
        started = await run_tool("bash_command", args, **_tool_kwargs())
        assert started.startswith(f'$ {sleeper}\nstarted background job "dev"')
        tool_input = json.loads(seen.read_text(encoding="utf-8"))["tool_input"]
        assert tool_input["run_in_background"] is True and tool_input["background_id"] == "dev"

        result = await run_tool("kill_shell", {"background_id": "dev"}, **_tool_kwargs())
        job = registry.get("s1", "dev")
        assert result == "keep the server up" and job.status == "running" and _alive(job.pid)
        await registry.kill_all()
    _run(scenario())

    # A rewrite that turns a read-only call into a background server is re-authorized: plan mode denies it
    write_user_config(home, _hooks_block("PreToolUse", _decide(
        tmp_path, updated={"command": "npm run dev", "run_in_background": True, "background_id": "dev"})))
    result = _run(run_tool("bash_command", {"command": "git status"}, adapter=AskAdapter(),
                           permissions=SessionPermissions(mode="plan"), **_tool_kwargs()))
    assert result.startswith('Plan mode: not run (Run shell command (background job "dev"):)')
    assert [j.id for j in registry.jobs()] == ["dev"]  # only the job from the first part
