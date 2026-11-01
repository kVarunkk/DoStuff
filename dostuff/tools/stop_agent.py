"""The `stop_agent` tool: stop a running subagent (subagents spec REQ-1.4, §Stop)."""
from dostuff.agent.subagents.registry import get_agent_registry, owner_agents_error


async def stop_agent(agent_id: str, session_id: str) -> str:
    """Stop a running subagent that you started. Its shell jobs are killed and an unchanged worktree is removed.

    Args:
        agent_id: The agent_id the `agent` call returned.

    Returns:
        Whether it was stopped. A background agent's partial work still arrives as a <background-agent> message.
    """
    registry = get_agent_registry()
    run = registry.get(agent_id)
    if run is None or run.owner != session_id:
        return owner_agents_error(agent_id, session_id)
    if run.task is None or run.task.done():
        return f'Agent "{run.id}" already {run.status}; nothing to stop.'
    await registry.stop(run, "model")
    return f'Stopped agent "{run.id}" ({run.agent_type}).'
