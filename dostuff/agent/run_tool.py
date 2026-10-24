import asyncio
import difflib
import inspect
from pathlib import Path
from typing import Any, cast

from opentelemetry.trace import get_current_span

from dostuff.agent.permissions import (
    ApprovalReply, PermissionConfig, PermissionDecision, PermissionMode, SessionPermissions, check,
    describe_approval_keys, to_reply,
)
from dostuff.config import get_config
from dostuff.helpers.tools.resolve_safe_path import resolve_safe_path
from dostuff.lib.checkpoints import CHECKPOINT_TOOLS, get_checkpoint_store
from dostuff.lib.mcp.mcp_client import MCPClient
from dostuff.lib.tracing import content_attr, traced
from dostuff.tools.definitions import HARNESS_PARAMS, TOOL_MAP, ToolSpec
from dostuff.tools.settings import SETTINGS_TOOLS, settings_preview

# Legacy approval flags a model may still try to pass; always dropped before dispatch.
APPROVAL_KEYS = frozenset({"_confirmed", "overwrite"})
_PREVIEW_MAX_LINES = 40


def _sanitize_args(spec: ToolSpec, fn_args: dict[str, Any]) -> dict[str, Any]:
    """Drop approval flags, `_`-prefixed keys and harness-injected params supplied by the model."""
    blocked = APPROVAL_KEYS | spec.injected | HARNESS_PARAMS
    return {k: v for k, v in fn_args.items() if not k.startswith("_") and k not in blocked}


def _approval_preview(tool: str, args: dict[str, Any]) -> str:
    """A plain diff for file writes/edits so the user sees what they approve."""
    if tool in SETTINGS_TOOLS:
        return settings_preview(tool, args)
    if tool not in ("write_file", "edit_file") or not isinstance(args.get("path"), str):
        return ""
    try:
        path = resolve_safe_path(args["path"])
        old = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    except Exception:
        return ""
    if tool == "edit_file":
        old_str, new_str = str(args.get("old_str", "")), str(args.get("new_str", ""))
        if not old_str or old_str not in old:
            return "  (old_str not found; the edit will fail)"
        new = old.replace(old_str, new_str) if args.get("replace_all") else old.replace(old_str, new_str, 1)
    else:
        new = str(args.get("content", ""))
        if not old:
            return f"  (new file, {len(new.splitlines())} lines)"
    diff = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=args["path"], tofile=args["path"], lineterm="",
    ))
    if len(diff) > _PREVIEW_MAX_LINES:
        diff = diff[:_PREVIEW_MAX_LINES] + [f"… {len(diff) - _PREVIEW_MAX_LINES} more diff lines"]
    return "\n".join(diff)


def permission_config() -> PermissionConfig:
    cfg = get_config()
    return PermissionConfig(mode=cast(PermissionMode, cfg.permission_mode), allow=list(cfg.permission_allow))


def _decide(fn_name: str, fn_args: dict[str, Any], permissions: SessionPermissions | None) -> PermissionDecision:
    spec = TOOL_MAP.get(fn_name)
    if spec is None:
        return PermissionDecision("allow")  # unknown tools fail in dispatch with a clear error
    args = _sanitize_args(spec, fn_args)
    return check(fn_name, spec.kind, args, permissions, permission_config(), _approval_preview(fn_name, args))


def _denial(decision: PermissionDecision, reply: ApprovalReply | None = None) -> str:
    text = f"User denied: {decision.reason}"
    if reply is not None and reply.feedback.strip():
        text += f"\nThe user said: {reply.feedback.strip()}"
    return text


async def authorize_tool(
    fn_name: str,
    fn_args: dict[str, Any],
    adapter: Any = None,
    permissions: SessionPermissions | None = None,
) -> str | None:
    """Run the permission check (asking the user if needed).

    Returns None if the call may proceed, else the denial text to use as the tool result.
    """
    decision = _decide(fn_name, fn_args, permissions)
    if decision.action == "allow":
        return None
    if decision.action == "deny":
        return _denial(decision)
    if adapter is None or not hasattr(adapter, "ask"):
        return f"User denied: no interactive approval available for {fn_name}. ({decision.reason})"
    reply = to_reply(await adapter.ask(decision.prompt, always_label=describe_approval_keys(decision.approval_keys)))
    if reply.answer == "always":
        if permissions is not None:
            permissions.approve(decision.approval_keys)
        return None
    if reply.answer == "yes":
        return None
    return _denial(decision, reply)


async def authorize_batch(
    calls: list[tuple[str, dict[str, Any]]],
    adapter: Any = None,
    permissions: SessionPermissions | None = None,
) -> list[str | None] | None:
    """Ask once for every call of a model response that needs approval.

    Returns one result per call (None = may run, str = denial), or None when there's nothing
    to batch (fewer than two prompts, or an adapter without `ask_batch`) or the user chose
    to review the calls one by one; callers then use `authorize_tool` per call.
    """
    if adapter is None or not hasattr(adapter, "ask_batch") or len(calls) < 2:
        return None
    decisions = [_decide(name, args, permissions) for name, args in calls]
    asking = [d for d in decisions if d.action == "ask"]
    if len(asking) < 2:
        return None
    keys = [k for d in asking for k in d.approval_keys]
    # "always" is offered only if it would cover every call (shell commands with unsafe syntax have no key)
    always_label = describe_approval_keys(keys) if all(d.approval_keys for d in asking) else ""
    reply = to_reply(await adapter.ask_batch([d.prompt for d in asking], always_label=always_label))
    if reply.answer == "each":
        return None
    if reply.answer == "always" and permissions is not None:
        permissions.approve(keys)
    allowed = reply.answer in ("yes", "always")
    return [
        _denial(d) if d.action == "deny"
        else _denial(d, reply) if d.action == "ask" and not allowed
        else None
        for d in decisions
    ]


@traced("tool_call")
async def dispatch_tool(
    fn_name: str,
    fn_args: dict[str, Any],
    mcp_client: MCPClient | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    adapter: Any = None,
    permissions: SessionPermissions | None = None,
) -> Any:
    """Execute an already-authorized tool call, injecting harness params the tool declares."""
    spec = TOOL_MAP.get(fn_name)
    if spec is None:
        raise KeyError(f"Tool '{fn_name}' not found.")
    args = _sanitize_args(spec, fn_args)
    available: dict[str, Any] = {
        "mcp_client": mcp_client, "session_id": session_id, "turn_id": turn_id,
        "adapter": adapter, "permissions": permissions,
    }
    for param in spec.injected:
        if available[param] is None and param in ("mcp_client", "session_id", "turn_id"):
            raise ValueError(f"Cannot execute tool '{fn_name}' without '{param}'.")
        args[param] = available[param]

    fn = spec.impl
    span = get_current_span()
    span.set_attribute("tool.input.name", fn_name)
    span.set_attribute("tool.input.args", content_attr(_sanitize_args(spec, fn_args)))
    checkpoint_path = await _checkpoint_before(fn_name, args, session_id, turn_id)
    try:
        if inspect.iscoroutinefunction(fn):
            result = await fn(**args)
        else:
            result = await asyncio.to_thread(fn, **args)
    finally:
        if checkpoint_path is not None:
            await asyncio.to_thread(get_checkpoint_store().after_change, cast(str, session_id), cast(str, turn_id), checkpoint_path)
    span.set_attribute("tool.output.result", content_attr(result))
    return result


async def _checkpoint_before(fn_name: str, args: dict[str, Any], session_id: str | None, turn_id: str | None) -> Path | None:
    """Snapshot the file a file tool is about to change, for /undo. Returns the path, or None if not tracked."""
    if fn_name not in CHECKPOINT_TOOLS or not session_id or not turn_id or not isinstance(args.get("path"), str):
        return None
    try:
        path = resolve_safe_path(args["path"])
    except Exception:
        return None  # the tool itself reports the bad path
    await asyncio.to_thread(get_checkpoint_store().before_change, session_id, turn_id, path)
    return path


async def run_tool(
    fn_name: str | None,
    fn_args: dict[str, Any],
    mcp_client: MCPClient | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    adapter: Any = None,
    permissions: SessionPermissions | None = None,
) -> Any:
    """Authorize then dispatch one tool call. A denial is returned as the result, not raised."""
    if not isinstance(fn_name, str):
        raise ValueError("Tool name must be a string.")
    denial = await authorize_tool(fn_name, fn_args, adapter=adapter, permissions=permissions)
    if denial is not None:
        return denial
    return await dispatch_tool(
        fn_name, fn_args, mcp_client=mcp_client, session_id=session_id, turn_id=turn_id,
        adapter=adapter, permissions=permissions,
    )
