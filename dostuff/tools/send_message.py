"""The `send_message` tool: a follow-up to a subagent you started (subagents spec D10, REQ-6)."""
from typing import Any

from dostuff.agent.subagents.fork import FORK_TYPE, NO_CONVERSATION, ForkContext
from dostuff.agent.subagents.registry import ENDED_STATUSES, get_agent_registry, owner_agents_error
from dostuff.lib.mcp.mcp_client import MCPClient
from dostuff.lib.worktree import WorktreeError
from dostuff.tools.agent import background_started, foreground_reply

QUEUED = 'Queued for "{id}".'


async def send_message(
    to: str,
    message: str,
    mcp_client: MCPClient,
    session_id: str,
    turn_id: str,
    run_in_background: bool = False,
    adapter: Any = None,
    permissions: Any = None,
    fork_context: ForkContext | None = None,
) -> str:
    """Send a follow-up to an agent you started. It keeps everything from its earlier work, so ask for more
    instead of starting a new agent on the same task.

    Args:
        to: The agent_id an `agent` call returned.
        message: What to tell or ask it, self-contained like a prompt.
        run_in_background: For an agent that has finished: True continues it in the background and returns at
            once; its report arrives later in a <background-agent> message. Ignored while the agent is running.

    Returns:
        For a finished agent, its new report and agent_id (in the background, only that it started). For a running
        one, that the message is queued: it reads it before its next step and you get its report as before.
    """
    registry = get_agent_registry()
    run = registry.get(to)
    if run is None or run.owner != session_id:
        return owner_agents_error(to, session_id)
    if run.status not in ENDED_STATUSES:
        run.inbox.append(message)  # read before its next model call (D10)
        return QUEUED.format(id=run.id)
    if run.agent_type == FORK_TYPE and run.fork is None and fork_context is None:
        return NO_CONVERSATION  # a fork loaded after a restart takes its prompt from the main agent's loop
    try:
        await registry.resume(run, message, background=run_in_background, mcp_client=mcp_client, adapter=adapter,
                              permissions=permissions, turn_id=turn_id, fork=fork_context)
    except WorktreeError as e:
        return f"[error] Could not create a new worktree for agent \"{run.id}\": {e}"
    if run_in_background:
        return background_started(run, registry)
    return await foreground_reply(run, registry, adapter)
