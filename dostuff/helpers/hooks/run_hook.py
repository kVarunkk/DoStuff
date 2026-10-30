"""Run one hook command: JSON on stdin, exit code + stdout parsed into a HookResult (spec D3, D7)."""
import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Literal

from dostuff.helpers.agent.loop_check_cancelled import is_cancelled
from dostuff.helpers.agent.tool_output import truncate_tool_output
from dostuff.helpers.hooks.load_hooks import HookEntry
from dostuff.helpers.tools.project_root import project_root
# Same shell, detach and kill-tree discipline as bash_command (CON-005, D7)
from dostuff.tools.bash import bash_command as _bash

Decision = Literal["allow", "deny", "ask", "none"]

FAILED_REASON = "hook '{name}' failed"
MISCONFIGURED_REASON = "hook '{name}' misconfigured: {problem}"
POLL_INTERVAL_S = 0.2  # same as the loop's _wait_or_cancel
_DRAIN_GRACE_S = 1.0  # keep reading this long after the hook exits (a background child may hold the pipes)

# Per-event payload keys on top of the envelope (D3)
EVENT_KEYS: dict[str, tuple[str, ...]] = {
    "PreToolUse": ("tool_name", "tool_input", "tool_use_id"),
    "PostToolUse": ("tool_name", "tool_input", "tool_use_id", "tool_response"),
    "UserPromptSubmit": ("prompt",),
    "SessionStart": ("source",),
    "Stop": ("last_assistant_message", "stop_hook_active"),
    "SubagentStop": ("last_assistant_message", "stop_hook_active"),
}
# Claude Code's top-level `decision` spellings
_DECISIONS: dict[str, Decision] = {"block": "deny", "approve": "allow", "allow": "allow", "deny": "deny", "ask": "ask"}
_PERMISSION_DECISIONS: dict[str, Decision] = {"allow": "allow", "deny": "deny", "ask": "ask"}


@dataclass(frozen=True)
class HookResult:
    name: str
    decision: Decision = "none"
    reason: str = ""
    updated_input: dict[str, Any] | None = None
    context: str = ""
    error: str | None = None  # crash, timeout, bad exit code: non-blocking unless the entry is fail_closed
    warning: str | None = None  # the hook ran, but part of its output was ignored
    cancelled: bool = False
    exit_code: int | None = None


def payload_for(event: str, *, session_id: str, turn_id: str, turn_type: str, **extra: Any) -> dict[str, Any]:
    """The stdin JSON for `event`: the common envelope plus exactly that event's keys (D3)."""
    expected = set(EVENT_KEYS[event])
    if set(extra) != expected:
        raise TypeError(f"{event} payload needs {sorted(expected)}, got {sorted(extra)}")
    return {
        "hook_event_name": event, "session_id": session_id, "turn_id": turn_id,
        "cwd": str(project_root()), "turn_type": turn_type, **extra,
    }


def _env(payload: dict[str, Any]) -> dict[str, str]:
    return {
        **os.environ,
        "DOSTUFF_HOOK_EVENT": str(payload.get("hook_event_name", "")),
        "DOSTUFF_SESSION_ID": str(payload.get("session_id", "")),
        "DOSTUFF_TURN_ID": str(payload.get("turn_id", "")),
        "DOSTUFF_PROJECT_ROOT": str(project_root()),
        "DOSTUFF_TURN_TYPE": str(payload.get("turn_type", "")),
    }


def _parse_stdout(name: str, stdout: str, cap: int) -> HookResult:
    """Exit 0: JSON in either Claude Code spelling, else plain text as context."""
    text = stdout.strip()
    try:
        data = json.loads(text) if text.startswith("{") else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return HookResult(name=name, context=truncate_tool_output(text, cap), exit_code=0)
    decision: Decision = "none"
    reason = str(data.get("reason") or "")
    warning = None
    if "decision" in data and data["decision"] is not None:
        mapped = _DECISIONS.get(str(data["decision"]).lower())
        if mapped is None:
            warning = f"hook '{name}': unknown decision {data['decision']!r} ignored"
        else:
            decision = mapped
    specific = data.get("hookSpecificOutput")
    updated_input = None
    context = ""
    if isinstance(specific, dict):
        if "permissionDecision" in specific:
            mapped = _PERMISSION_DECISIONS.get(str(specific["permissionDecision"]).lower())
            if mapped is None:
                warning = f"hook '{name}': unknown permissionDecision {specific['permissionDecision']!r} ignored"
            else:
                decision = mapped
                reason = str(specific.get("permissionDecisionReason") or reason)
        if isinstance(specific.get("updatedInput"), dict):
            updated_input = specific["updatedInput"]
        context = str(specific.get("additionalContext") or "")
    return HookResult(
        name=name, decision=decision, reason=truncate_tool_output(reason, cap), updated_input=updated_input,
        context=truncate_tool_output(context, cap), warning=warning, exit_code=0,
    )


async def _wait(proc: asyncio.subprocess.Process, timeout: float, adapter: Any) -> str:
    """Wait for the hook to exit: "exited", "timeout" or "cancelled" (polling the ESC flag)."""
    deadline = time.monotonic() + timeout
    waiter = asyncio.ensure_future(proc.wait())
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            done, _ = await asyncio.wait({waiter}, timeout=min(POLL_INTERVAL_S, remaining))
            if done:
                return "exited"
            if is_cancelled(adapter):
                return "cancelled"
    finally:
        waiter.cancel()


async def run_hook(
    entry: HookEntry, payload: dict[str, Any], *, adapter: Any = None, max_output_chars: int | None = None,
) -> HookResult:
    """Run `entry.command` through select_shell() in project_root(); never raises (except CancelledError)."""
    name = entry.name
    if entry.broken:
        return HookResult(name=name, decision="deny",
                          reason=MISCONFIGURED_REASON.format(name=name, problem=entry.broken))
    if max_output_chars is None:
        from dostuff.config import get_config  # lazy: config imports this package
        max_output_chars = get_config().max_tool_output_chars
    _, argv = _bash.select_shell()
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, entry.command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(project_root()), env=_env(payload), **_bash._detach_kwargs(),
        )
    except OSError as e:
        return HookResult(name=name, error=f"hook '{name}' could not start: {e}")

    out, err = bytearray(), bytearray()
    readers = asyncio.gather(_bash._read_into(proc.stdout, out), _bash._read_into(proc.stderr, err))
    readers.add_done_callback(lambda f: None if f.cancelled() else f.exception())
    try:
        if proc.stdin is not None:
            try:
                proc.stdin.write(json.dumps(payload).encode("utf-8"))
                await proc.stdin.drain()
                proc.stdin.close()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the hook exited without reading stdin
        state = await _wait(proc, entry.timeout, adapter)
        if state != "exited":
            await _bash._kill_tree(proc)
        done, _ = await asyncio.wait({readers}, timeout=_DRAIN_GRACE_S)
        if not done:
            _bash._stop_reading(proc, readers)
    except asyncio.CancelledError:
        _bash._stop_reading(proc, readers)
        await _bash._kill_tree(proc)
        raise

    if state == "cancelled":
        return HookResult(name=name, cancelled=True)
    if state == "timeout":
        return HookResult(name=name, error=f"hook '{name}' timed out after {entry.timeout:g}s and was killed")
    # Normalise Windows line endings: this text lands in the model's context verbatim
    stdout = out.decode("utf-8", errors="replace").replace("\r\n", "\n")
    stderr = err.decode("utf-8", errors="replace").replace("\r\n", "\n").strip()
    code = proc.returncode
    if code == 0:
        return _parse_stdout(name, stdout, max_output_chars)
    if code == 2:
        reason = stderr or stdout.strip()
        return HookResult(name=name, decision="deny", reason=truncate_tool_output(reason, max_output_chars), exit_code=2)
    detail = f": {truncate_tool_output(stderr, max_output_chars)}" if stderr else ""
    return HookResult(name=name, error=f"hook '{name}' exited with code {code}{detail}", exit_code=code)
