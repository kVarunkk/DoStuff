"""One subagent run: its loop, the partial report when it stops early, and the worktree close-out (spec D1)."""
from typing import Any, Literal

from dostuff.agent.subagents import RESTRICTED_SUBAGENT_TOOLS
from dostuff.agent.subagents.adapter import SubagentAdapter
from dostuff.helpers.tools.project_root import project_root
from dostuff.lib.mcp.mcp_client import MCPClient
from dostuff.lib.worktree import Worktree, WorktreeError, remove_worktree, worktree_changes

WRAP_UP_PROMPT = (
    "You have run out of steps. Do not call any more tools. Report what you found so far, "
    "what is still unknown, and what the next steps would be."
)
DIGEST_ITEM_CHARS = 300
DIGEST_MAX_CHARS = 4000

RunOutcome = Literal["completed", "failed", "stopped"]


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


def _close_worktree(wt: Worktree) -> str:
    """Remove the subagent's worktree if it changed nothing; otherwise keep it and tell the parent where."""
    try:
        dirty, commits = worktree_changes(wt)
        if not dirty and not commits:
            remove_worktree(wt)
            return f"[worktree] The subagent changed no files; worktree {wt.name} was removed."
    except WorktreeError as e:
        return f"[worktree] Kept {wt.path} (branch {wt.branch}); could not check or remove it: {e}"
    try:
        rel = wt.path.relative_to(project_root().resolve()).as_posix()
    except ValueError:
        rel = str(wt.path)
    changed = ", ".join(dirty[:20]) + (f" (+{len(dirty) - 20} more)" if len(dirty) > 20 else "")
    return (
        f"[worktree] The subagent's changes are kept in {rel} on branch {wt.branch}"
        + (f"; uncommitted: {changed}" if dirty else "")
        + (f"; {commits} commit(s)" if commits else "")
        + f". Review them with `git -C {rel} status` / `git -C {rel} diff`, then bring them over (copy the files, "
        f"or commit there and `git merge {wt.branch}`). Remove it afterwards with "
        f"`git worktree remove --force {rel}` and `git branch -D {wt.branch}`."
    )


def resolve_subagent_tools(requested_tools: list[str] | None) -> tuple[list[str], list[str]]:
    """Split requested tool names into (allowed, dropped). Restricted and unknown names are dropped.
    None means every tool a subagent may have (D2: `tools:` omitted)."""
    from dostuff.tools.definitions import TOOL_MAP, with_shell_job_tools
    if requested_tools is None:
        requested_tools = [name for name in TOOL_MAP if name not in RESTRICTED_SUBAGENT_TOOLS]
    allowed: list[str] = []
    dropped: list[str] = []
    for name in dict.fromkeys(requested_tools):
        if name in RESTRICTED_SUBAGENT_TOOLS or name not in TOOL_MAP:
            dropped.append(name)
        else:
            allowed.append(name)
    return with_shell_job_tools(allowed), dropped


async def run_agent_loop(
    task: str,
    mcp_client: MCPClient | None,
    session_id: str,
    turn_id: str,
    adapter: SubagentAdapter,
    permissions: Any,
    subagent_system_instruction: str,
    working_history: list[dict[str, Any]],
    allowed_tool_names: list[str],
    owner: str,
    max_iterations: int,
    store: Any = None,
    compaction_notes: str = "",
    inbox: list[str] | None = None,
    session_hook_context: dict[str, str] | None = None,
) -> tuple[str, RunOutcome]:
    """The subagent's loop, plus a partial report if it stopped before a final answer. `store` is the run's
    transcript (`AgentTranscript`), or None to keep its steps in memory only."""
    from dostuff.agent.loop import CANCELLED_TURN, MODEL_ERROR_PREFIX, loop

    subagent_result = await loop(
        session_id=session_id,
        turn_id=turn_id,
        user_text=task,
        dynamic_system_instruction=subagent_system_instruction,
        mcp_client=mcp_client,  # type: ignore[arg-type]  # only the MCP tools need it, and they say so
        working_history=working_history,
        turn_type="subagent_loop",
        adapter=adapter,
        permissions=permissions,
        tool_names=allowed_tool_names,
        stream=False,
        max_iterations=max_iterations,
        owner=owner,
        store=store,
        compaction_notes=compaction_notes,
        inbox=inbox,
        session_hook_context=session_hook_context,  # a fork's: the main agent's SessionStart context (D11)
    )
    if subagent_result == CANCELLED_TURN:
        return subagent_result, "stopped"
    outcome: RunOutcome = "failed" if subagent_result.startswith(MODEL_ERROR_PREFIX) else "completed"
    if not _finished(working_history):
        # Max iterations, model error or empty replies: keep the work instead of only the error
        report = await _partial_report(working_history, subagent_system_instruction, allowed_tool_names)
        subagent_result = f"[subagent did not finish: {subagent_result}]\n\nPartial report:\n{report}"
    return subagent_result, outcome
