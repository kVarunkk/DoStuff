"""Subagents: agent types, runs and their registry (plan/2026-11-01-subagents/spec.md)."""

# Never granted to a subagent, whatever its type or `tools:` list asks for (spec D12)
RESTRICTED_SUBAGENT_TOOLS = frozenset({
    # No nesting: a subagent can't start, message or stop agents (OUT-001)
    "agent",
    "send_message",
    "stop_agent",
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
