"""The `agent` tool: start a subagent (subagents spec REQ-1, D1)."""
import asyncio
import uuid
from typing import Any

from dostuff.agent.subagents.definitions import UnknownAgentType, get_definitions
from dostuff.agent.subagents.registry import get_agent_registry
from dostuff.helpers.agent.loop_check_cancelled import is_cancelled
from dostuff.helpers.tools.project_root import project_root
from dostuff.lib.mcp.mcp_client import MCPClient
from dostuff.lib.worktree import Worktree, WorktreeError, create_worktree

FOOTER = "agent_id: {id} (send_message to continue)"


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
) -> str:
    """Start a subagent: a separate agent loop that works on one task with its own tools and returns a report.

    Args:
        description: 3-5 words saying what the agent does, shown to the user (e.g. "find auth code").
        prompt: The task. The agent cannot see this conversation, so include every path, constraint and goal
            it needs, and say what its report should contain.
        subagent_type: Which agent type to start; the system prompt lists them.
        model: A model for this agent: an alias from agents.models or a full model name. Empty uses the
            type's model, which by default is the session's.
        run_in_background: Not available yet; leave false.
        isolation: "worktree" runs the agent in a new git worktree (a separate checkout of HEAD on its own
            branch), so its file changes don't touch the working tree. Use it for agents that write files. It
            does not see uncommitted changes. If it changes nothing the worktree is removed; otherwise the result
            says where the changes are. Leave empty to work in the project directory.

    Returns:
        The agent's final report (a partial report if it stopped early), then its agent_id.
    """
    if run_in_background:
        return "[error] run_in_background is not available yet; start the agent in the foreground."
    definitions = get_definitions()
    if subagent_type not in definitions:
        return f"[error] {UnknownAgentType(subagent_type, sorted(definitions))}"
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
    )
    assert run.done is not None
    try:
        report = await asyncio.shield(run.done)
    except asyncio.CancelledError:
        await registry.stop(run, "user")  # ESC: this call was the run's only reader
        raise
    if run.status == "stopped" and is_cancelled(adapter):
        from dostuff.agent.run_tool import CANCELLED_RESULT  # run_tool imports the tool registry
        report = CANCELLED_RESULT  # it saw the ESC first; same result as a call cancelled mid-flight
    return f"{report}\n\n{FOOTER.format(id=run.id)}"
