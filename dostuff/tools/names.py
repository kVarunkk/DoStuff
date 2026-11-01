"""The built-in tool names, without importing the tools.

dostuff.tools.definitions imports every tool module, and through the memory tools litellm, which
takes seconds. Code that runs before the TUI's first frame (agent-file parsing, via
build_base_instructions) only needs the names, so it reads them from here.
tests/test_subagents_types.py keeps this in step with TOOL_MAP.
"""

BUILTIN_TOOL_NAMES = frozenset({
    "read_file", "write_file", "edit_file", "delete_file", "list_files", "grep", "glob",
    "get_current_datetime", "web_fetch", "web_search", "todo_write", "save_memory", "save_episode",
    "search_mcp_tools", "get_mcp_tool_details", "call_mcp_tool",
    "agent", "send_message", "stop_agent", "create_agent_type",
    "read_settings", "edit_config", "set_mcp_server", "remove_mcp_server",
    "bash_command", "bash_output", "kill_shell",
})
