import asyncio
import difflib
import inspect
from pathlib import Path
from typing import Any, cast

from opentelemetry.trace import get_current_span

from dostuff.agent.permissions import (
    ApprovalReply, PermissionConfig, PermissionDecision, PermissionMode, SessionPermissions, check,
    describe_approval_keys, effective_kind, forced_ask, to_reply,
)
from dostuff.config import get_config
from dostuff.helpers.agent.tool_output import truncate_tool_output
from dostuff.helpers.hooks import HookOutcome, run_hooks
from dostuff.helpers.hooks.dispatch import report_outcome
from dostuff.helpers.hooks.run_hook import payload_for
from dostuff.helpers.tools.resolve_safe_path import resolve_safe_path
from dostuff.lib.checkpoints import CHECKPOINT_TOOLS, get_checkpoint_store
from dostuff.lib.mcp.mcp_client import MCPClient
from dostuff.lib.tracing import content_attr, traced
from dostuff.tools.definitions import HARNESS_PARAMS, TOOL_MAP, ToolSpec
from dostuff.tools.settings import SETTINGS_TOOLS, settings_preview

# Legacy approval flags a model may still try to pass; always dropped before dispatch.
APPROVAL_KEYS = frozenset({"_confirmed", "overwrite"})
CANCELLED_RESULT = "Cancelled by user"
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
    cfg.reload_if_changed()  # permissions edited in config.yaml by hand apply from the next tool call
    return PermissionConfig(mode=cast(PermissionMode, cfg.permission_mode), allow=list(cfg.permission_allow))


def _hook_decision(
    hook: HookOutcome | None, spec: ToolSpec, args: dict[str, Any], base: PermissionDecision, preview: str,
) -> PermissionDecision:
    """Combine a PreToolUse outcome with the permission layer's decision; first match wins (hooks spec D2)."""
    if hook is None or base.action == "deny":
        return base  # 1. plan mode: a hook can only tighten it
    if hook.decision == "deny":
        return PermissionDecision("deny", reason=hook.reason or f"Blocked by hook '{hook.decided_by}'.")  # 2.
    if effective_kind(spec.kind, args) == "settings":
        return base  # 3. settings changes ask in every mode; a hook allow cannot silence them
    if hook.decision == "ask":  # 4. even in auto mode and for read-only / allowlisted calls
        note = f"Hook '{hook.decided_by}' asks: {hook.reason}" if hook.reason else f"Hook '{hook.decided_by}' asks"
        return forced_ask(spec.name, args, preview, note)
    if hook.decision == "allow" and hook.updated_input is None:
        return PermissionDecision("allow")  # 5. like permissions.allow; a rewrite voids it (D6)
    return base  # 6.


def _decide(
    fn_name: str, fn_args: dict[str, Any], permissions: SessionPermissions | None, hook: HookOutcome | None = None,
) -> PermissionDecision:
    spec = TOOL_MAP.get(fn_name)
    if spec is None:
        return PermissionDecision("allow")  # unknown tools fail in dispatch with a clear error
    args = _sanitize_args(spec, fn_args)
    preview = _approval_preview(fn_name, args)
    base = check(fn_name, spec.kind, args, permissions, permission_config(), preview)
    return _hook_decision(hook, spec, args, base, preview)


def _denial(decision: PermissionDecision, reply: ApprovalReply | None = None) -> str:
    if decision.action == "deny":  # a rule (plan mode), not the user: the reason is the whole message
        return decision.reason
    text = f"User denied: {decision.reason}"
    if reply is not None and reply.feedback.strip():
        text += f"\nThe user said: {reply.feedback.strip()}"
    return text


async def authorize_tool(
    fn_name: str,
    fn_args: dict[str, Any],
    adapter: Any = None,
    permissions: SessionPermissions | None = None,
    hook: HookOutcome | None = None,
) -> str | None:
    """Run the permission check (asking the user if needed), with the call's PreToolUse outcome as an input.

    Returns None if the call may proceed, else the denial text to use as the tool result.
    """
    decision = _decide(fn_name, fn_args, permissions, hook)
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
    hooks: list[HookOutcome | None] | None = None,
) -> list[str | None] | None:
    """Ask once for every call of a model response that needs approval (`hooks`: one PreToolUse outcome per call).

    Returns one result per call (None = may run, str = denial), or None when there's nothing
    to batch (fewer than two prompts, or an adapter without `ask_batch`) or the user chose
    to review the calls one by one; callers then use `authorize_tool` per call.
    """
    if adapter is None or not hasattr(adapter, "ask_batch") or len(calls) < 2:
        return None
    outcomes = hooks if hooks is not None else [None] * len(calls)
    decisions = [_decide(name, args, permissions, hook) for (name, args), hook in zip(calls, outcomes)]
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


async def pre_tool_use(
    fn_name: str, args: dict[str, Any], *, session_id: str | None, turn_id: str | None, turn_type: str,
    adapter: Any = None, tool_use_id: str = "",
) -> tuple[dict[str, Any], HookOutcome]:
    """Run PreToolUse hooks. Returns the (possibly rewritten, re-sanitized) args and the merged outcome.

    The caller turns `deny` into the tool result and `cancelled` into CANCELLED_RESULT, and passes the
    outcome to authorize_tool / authorize_batch, which apply the rest of the D2 precedence.
    """
    payload = payload_for(
        "PreToolUse", session_id=session_id or "", turn_id=turn_id or "", turn_type=turn_type,
        tool_name=fn_name, tool_input=args, tool_use_id=tool_use_id,
    )
    outcome = await run_hooks("PreToolUse", payload, adapter=adapter)
    if outcome.updated_input is not None:
        # Re-sanitize before the permission layer sees it: a rewrite can't smuggle in _confirmed,
        # overwrite or harness params, and faces the same gate as a model call (spec D6)
        spec = TOOL_MAP.get(fn_name)
        merged = {**args, **outcome.updated_input}
        args = _sanitize_args(spec, merged) if spec is not None else merged
    await report_outcome("PreToolUse", outcome, adapter=adapter, target=fn_name)
    return args, outcome


def _assemble(result: str, outcome: HookOutcome, cap: int) -> str:
    """Append each PostToolUse context (or a block note) as `<hook name="…">…</hook>`, then cap (spec D5)."""
    parts = [result]
    parts += [f'<hook name="{name}">{text}</hook>' for name, text in outcome.contexts]
    if outcome.decision == "deny":
        parts.append(f'<hook name="{outcome.decided_by}" status="blocked">{outcome.reason}</hook>')
    return truncate_tool_output("\n".join(parts), cap)


async def post_tool_use(
    fn_name: str, args: dict[str, Any], result: Any, *, session_id: str | None, turn_id: str | None,
    turn_type: str, adapter: Any = None, tool_use_id: str = "",
) -> tuple[Any, HookOutcome]:
    """Run PostToolUse hooks; their context lands in the result the model sees. Cancelled → result unchanged."""
    cap = get_config().max_tool_output_chars
    payload = payload_for(
        "PostToolUse", session_id=session_id or "", turn_id=turn_id or "", turn_type=turn_type,
        tool_name=fn_name, tool_input=args, tool_use_id=tool_use_id,
        tool_response=truncate_tool_output(str(result), cap),
    )
    outcome = await run_hooks("PostToolUse", payload, adapter=adapter)
    await report_outcome("PostToolUse", outcome, adapter=adapter, target=fn_name)
    if outcome.cancelled or (not outcome.contexts and outcome.decision != "deny"):
        return result, outcome
    return _assemble(str(result), outcome, cap), outcome


@traced("tool_call")
async def dispatch_tool(
    fn_name: str,
    fn_args: dict[str, Any],
    mcp_client: MCPClient | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    adapter: Any = None,
    permissions: SessionPermissions | None = None,
    owner: str | None = None,
) -> Any:
    """Execute an already-authorized tool call, injecting harness params the tool declares.

    `owner` is the agent shell jobs belong to: the session, or `<session>/sub/<run>` for a subagent.
    """
    spec = TOOL_MAP.get(fn_name)
    if spec is None:
        raise KeyError(f"Tool '{fn_name}' not found.")
    args = _sanitize_args(spec, fn_args)
    available: dict[str, Any] = {
        "mcp_client": mcp_client, "session_id": session_id, "turn_id": turn_id,
        "adapter": adapter, "permissions": permissions, "owner": owner or session_id or "",
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
    turn_type: str = "interactive_loop",
    owner: str | None = None,
) -> Any:
    """Hooks, authorize, dispatch, hooks: the same order as loop(). A denial is returned as the result, not raised."""
    if not isinstance(fn_name, str):
        raise ValueError("Tool name must be a string.")
    ids: dict[str, Any] = {"session_id": session_id, "turn_id": turn_id, "turn_type": turn_type, "adapter": adapter}
    fn_args, hook = await pre_tool_use(fn_name, fn_args, **ids)
    if hook.cancelled:
        return CANCELLED_RESULT
    if hook.decision == "deny":
        return hook.reason
    denial = await authorize_tool(fn_name, fn_args, adapter=adapter, permissions=permissions, hook=hook)
    if denial is not None:
        return denial
    result = await dispatch_tool(
        fn_name, fn_args, mcp_client=mcp_client, session_id=session_id, turn_id=turn_id,
        adapter=adapter, permissions=permissions, owner=owner,
    )
    result, _ = await post_tool_use(fn_name, fn_args, result, **ids)
    return result
