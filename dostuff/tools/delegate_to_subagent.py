from typing import Any
from dostuff.lib.mcp.mcp_client import MCPClient

BASE_SUBAGENT_INSTRUCTIONS = """You are a focused subagent operating in an isolated sub-loop.

Core Directives:
1. Complete the specified task accurately and concisely.
2. Rely strictly on the tools provided to gather facts. Do NOT fabricate information.
3. You do NOT have access to the parent conversation history. Use only the task description provided.
4. When finished, provide a clear, structured final report.
"""

RESTRICTED_SUBAGENT_TOOLS = {
    "delegate_to_subagent",
    "delete_file",
    # Subagents never see the user conversation, so they have nothing to remember
    "save_memory",
    "save_episode",
    # The todo list belongs to the parent; a subagent would overwrite it
    "todo_write",
}

DEFAULT_SUBAGENT_TOOLS = ["read_file", "list_files", "grep", "glob", "get_current_datetime", "search_mcp_tools", "get_mcp_tool_details", "call_mcp_tool"]

WRAP_UP_PROMPT = (
    "You have run out of steps. Do not call any more tools. Report what you found so far, "
    "what is still unknown, and what the next steps would be."
)
DIGEST_ITEM_CHARS = 300
DIGEST_MAX_CHARS = 4000


class _SubagentAdapter:
    """Wraps the parent adapter so subagent output is prefixed with its role.

    Approval prompts and cancellation state (`_cancel_event`, `_cancelled`) go
    straight through to the parent, so ESC and HITL work inside the subagent.
    """

    def __init__(self, parent: Any, role: str):
        self._parent = parent
        self._prefix = f"[subagent: {role}]"

    def emit(self, event_type: str, data: Any) -> None:
        if event_type == "status":  # context-window % of the subagent would overwrite the parent's
            return
        if event_type in ("usage",) or not isinstance(data, str):
            self._parent.emit(event_type, data)
        else:
            self._parent.emit(event_type, f"{self._prefix} {data}")

    async def ask(self, message: str, **kwargs: Any) -> Any:
        return await self._parent.ask(f"{self._prefix} {message}", **kwargs)

    async def ask_batch(self, messages: list[str], **kwargs: Any) -> Any:
        if not hasattr(self._parent, "ask_batch"):
            return "each"
        return await self._parent.ask_batch([f"{self._prefix} {m}" for m in messages], **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parent, name)


def _finished(working_history: list[dict[str, Any]]) -> bool:
    """True if the subagent's last step is a final answer (text, no tool calls)."""
    last = working_history[-1] if working_history else None
    return bool(last and last.get("role") == "assistant" and last.get("content") and not last.get("tool_calls"))


def _digest(working_history: list[dict[str, Any]]) -> str:
    """Mechanical summary of what the subagent did, newest last, capped to DIGEST_MAX_CHARS."""
    lines: list[str] = []
    for step in working_history[1:]:  # skip the task itself
        text = str(step.get("content") or "").strip().replace("\n", " ")
        if step.get("role") == "tool":
            lines.append(f"- {step.get('name')}: {text[:DIGEST_ITEM_CHARS]}")
        elif step.get("role") == "assistant" and text:
            lines.append(f"- note: {text[:DIGEST_ITEM_CHARS]}")
    digest = "\n".join(lines)
    if len(digest) > DIGEST_MAX_CHARS:
        digest = "[… earlier steps omitted]\n" + digest[-DIGEST_MAX_CHARS:]
    return digest or "(no steps completed)"


async def _partial_report(working_history: list[dict[str, Any]], system_instruction: str, tool_names: list[str]) -> str:
    """Ask the subagent for a wrap-up without tools; fall back to a digest of its steps."""
    from dostuff.agent.call_agent import call_agent
    try:
        interaction, span = await call_agent(
            steps_history=working_history + [{"role": "user", "content": WRAP_UP_PROMPT}],
            system_instruction=system_instruction,
            tool_names=tool_names,
            stream=False,
        )
        if span.is_recording():
            span.end()
        message = interaction.choices[0].message
        if message.content and not getattr(message, "tool_calls", None):
            return message.content
    except Exception:
        pass
    return f"Steps taken:\n{_digest(working_history)}"


def resolve_subagent_tools(requested_tools: list[str] | None) -> tuple[list[str], list[str]]:
    """Split requested tool names into (allowed, dropped). Restricted and unknown names are dropped."""
    from dostuff.tools.definitions import TOOL_MAP
    allowed: list[str] = []
    dropped: list[str] = []
    for name in dict.fromkeys(requested_tools or DEFAULT_SUBAGENT_TOOLS):
        if name in RESTRICTED_SUBAGENT_TOOLS or name not in TOOL_MAP:
            dropped.append(name)
        else:
            allowed.append(name)
    return allowed, dropped


async def _impl_delegate_to_subagent(
    task: str,
    role: str,
    mcp_client: MCPClient,
    session_id: str,
    turn_id: str,
    requested_tools: list[str] | None = None,
    adapter: Any = None,
    permissions: Any = None,
) -> str:
    """Executes a subagent task in an isolated context loop.

    Args:
        task: Self-contained instructions detailing what the subagent must accomplish.
        role: Role framing descriptor used to specialize the subagent's system prompt.
        requested_tools: List of tool names the subagent may use. If omitted, defaults to
            read-only tools. delegate_to_subagent and delete_file are never granted.

    Returns:
        The subagent's final report. If it stops early (iteration cap, model error), a partial
        report of what it found so far.
    """
    from dostuff.agent.loop import loop
    from dostuff.config import get_config
    from dostuff.helpers.agent.loop_check_cancelled import is_cancelled

    allowed_tool_names, dropped = resolve_subagent_tools(requested_tools)

    subagent_system_instruction = f"""{BASE_SUBAGENT_INSTRUCTIONS}

Your Specific Role: {role}
"""
    working_history: list[dict[str, Any]] = [{"role": "user", "content": task}]

    subagent_result = await loop(
        session_id=session_id,
        turn_id=turn_id,
        user_text=task,
        dynamic_system_instruction=subagent_system_instruction,
        mcp_client=mcp_client,
        working_history=working_history,
        turn_type="subagent_loop",
        adapter=_SubagentAdapter(adapter, role) if adapter is not None else None,
        permissions=permissions,
        tool_names=allowed_tool_names,
        stream=False,
        max_iterations=get_config().subagent_max_iterations,
    )

    if not _finished(working_history) and not is_cancelled(adapter):
        # Max iterations, model error or empty replies: keep the work instead of only the error
        report = await _partial_report(working_history, subagent_system_instruction, allowed_tool_names)
        subagent_result = f"[subagent did not finish: {subagent_result}]\n\nPartial report:\n{report}"

    if dropped:
        warning = f"[warning] Tools not granted to the subagent (restricted or unknown): {', '.join(dropped)}"
        return f"{warning}\n\n{subagent_result}"
    return subagent_result
