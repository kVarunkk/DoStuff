"""The `agent` tool: start a subagent (subagents spec REQ-1, D1)."""
import asyncio
import uuid
from typing import Any

from dostuff.agent.subagents.definitions import UnknownAgentType, refresh_definitions
from dostuff.agent.subagents.fork import FORK_TYPE, NO_CONVERSATION, ForkContext
from dostuff.agent.subagents.registry import AgentRegistry, AgentRun, get_agent_registry
from dostuff.helpers.agent.loop_check_cancelled import is_cancelled
from dostuff.helpers.tools.project_root import project_root
from dostuff.lib.mcp.mcp_client import MCPClient
from dostuff.lib.worktree import Worktree, WorktreeError, create_worktree

FOOTER = "agent_id: {id} (send_message to continue)"
BACKGROUND_STARTED = ('Started agent "{id}" ({type}) in the background{queued}. '
                      "You'll get its report as a notice; don't poll.")


def background_started(run: AgentRun, registry: AgentRegistry) -> str:
    """The result of a background start (REQ-1.3)."""
    queued = f" (queued: {registry.max_concurrent} agents are already running)" if run.status == "queued" else ""
    return BACKGROUND_STARTED.format(id=run.id, type=run.agent_type, queued=queued)


async def foreground_reply(run: AgentRun, registry: AgentRegistry, adapter: Any) -> str:
    """Wait for a foreground run: its report and the agent_id footer (REQ-1.2). ESC stops the run; Ctrl+B
    returns at once and its report comes as a notice (D9)."""
    assert run.reply is not None
    try:
        report = await asyncio.shield(run.reply)
    except asyncio.CancelledError:
        await registry.stop(run, "user")  # ESC: this call was the run's only reader
        raise
    if run.background:
        return report  # Ctrl+B moved it (D9): its report comes as a notice
    if run.status == "stopped" and is_cancelled(adapter):
        from dostuff.agent.run_tool import CANCELLED_RESULT  # run_tool imports the tool registry
        report = CANCELLED_RESULT  # it saw the ESC first; same result as a call cancelled mid-flight
    return f"{report}\n\n{FOOTER.format(id=run.id)}"


async def agent(
    description: str,
    prompt: str,
    mcp_client: MCPClient,
    session_id: str,
    turn_id: str,
    subagent_type: str = "general-purpose",
    model: str = "",
    run_in_background: bool = False,
    isolation: str = "",
    adapter: Any = None,
    permissions: Any = None,
    fork_context: ForkContext | None = None,
) -> str:
    """Start a subagent: a separate agent loop that works on one task with its own tools and returns a report.

    Args:
        description: 3-5 words saying what the agent does, shown to the user (e.g. "find auth code").
        prompt: The task. A typed agent cannot see this conversation, so include every path, constraint and
            goal it needs, and say what its report should contain. For a fork, just the task.
        subagent_type: Which agent type to start; the system prompt lists them. "fork" starts a copy of you
            that inherits this conversation.
        model: A model for this agent: an alias from agents.models or a full model name. Empty uses the
            type's model, which by default is the session's. A fork always uses the session's model.
        run_in_background: True starts the agent and returns its agent_id at once, so you can keep working; its
            report arrives later in a <background-agent> message. Use it for independent, long work whose
            result you don't need yet. Don't poll or wait for it.
        isolation: "worktree" runs the agent in a new git worktree (a separate checkout of HEAD on its own
            branch), so its file changes don't touch the working tree. Use it for agents that write files. It
            does not see uncommitted changes. If it changes nothing the worktree is removed; otherwise the result
            says where the changes are. Leave empty to work in the project directory.

    Returns:
        The agent's final report (a partial report if it stopped early), then its agent_id; in the background,
        only its agent_id.
    """
    if subagent_type == FORK_TYPE:
        if fork_context is None:  # not called from the main agent's loop: nothing to inherit (D11)
            return NO_CONVERSATION
    else:
        definitions = (await refresh_definitions()).types
        if subagent_type not in definitions:
            return f"[error] {UnknownAgentType(subagent_type, sorted([*definitions, FORK_TYPE]))}"
    worktree: Worktree | None = None
    if isolation:
        if isolation != "worktree":
            return f"[error] Unknown isolation '{isolation}'; the only option is 'worktree'."
        try:
            worktree = await asyncio.to_thread(create_worktree, f"agent-{uuid.uuid4().hex[:6]}", project_root())
        except WorktreeError as e:
            return f"[error] Could not create a worktree for the subagent: {e}"
    registry = get_agent_registry()
    run = await registry.start(
        owner=session_id, agent_type=subagent_type, description=description, prompt=prompt,
        mcp_client=mcp_client, adapter=adapter, permissions=permissions, turn_id=turn_id, worktree=worktree,
        background=run_in_background, model=model,
        fork=fork_context if subagent_type == FORK_TYPE else None,
    )
    if run_in_background:
        return background_started(run, registry)
    return await foreground_reply(run, registry, adapter)
