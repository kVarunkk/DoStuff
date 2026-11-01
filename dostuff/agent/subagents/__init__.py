"""Subagents: agent types, runs and their registry (plan/2026-11-01-subagents/spec.md)."""
from contextvars import ContextVar
from typing import Any

# Set inside a run of a read-only type (explore, plan): bash_command may only run commands that
# permissions.is_read_only_shell accepts; anything else is denied without a prompt (spec D3)
read_only_shell: ContextVar[bool] = ContextVar("read_only_shell", default=False)

# The AgentRun whose task is running, None in the main agent: the TUI reads it to know which agent an
# approval prompt comes from (/agents → Answer approval, spec D14)
current_agent: ContextVar[Any] = ContextVar("current_agent", default=None)

# Never granted to a subagent, whatever its type or `tools:` list asks for (spec D12)
RESTRICTED_SUBAGENT_TOOLS = frozenset({
    # No nesting: a subagent can't start, message or stop agents (OUT-001)
    "agent",
    "send_message",
    "stop_agent",
    # Defining agent types is the main agent's, with the user's approval
    "create_agent_type",
    "delete_file",
    # Subagents never see the user conversation, so they have nothing to remember
    "save_memory",
    "save_episode",
    # The todo list belongs to the parent; a subagent would overwrite it
    "todo_write",
    # Settings are the user's to change, from the main conversation
    "read_settings",
    "edit_config",
    "set_mcp_server",
    "remove_mcp_server",
})
