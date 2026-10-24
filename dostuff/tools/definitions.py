from dataclasses import dataclass, field
from typing import Any, Callable

from dostuff.agent.permissions import ToolKind
from dostuff.helpers.tools.generate_tool_schema import generate_litellm_tool_schema
from dostuff.tools.bash.bash_command import bash_command, select_shell
from dostuff.tools.delegate_to_subagent import _impl_delegate_to_subagent
from dostuff.tools.files.delete_file import delete_file
from dostuff.tools.files.edit_file import edit_file
from dostuff.tools.files.list_files import list_files
from dostuff.tools.files.read_file import read_file
from dostuff.tools.files.write_file import write_file
from dostuff.tools.get_current_datetime import get_current_datetime
from dostuff.tools.mcp.call_mcp_tool import _impl_call_mcp_tool
from dostuff.tools.mcp.get_mcp_tool_details import _impl_get_mcp_tool_details
from dostuff.tools.mcp.search_mcp_tools import _impl_search_mcp_tools
from dostuff.tools.memory.save_episode import save_episode
from dostuff.tools.memory.save_memory import save_memory
from dostuff.tools.search.glob import glob
from dostuff.tools.settings import edit_config, read_settings, remove_mcp_server, set_mcp_server
from dostuff.tools.search.grep import grep
from dostuff.tools.todo_write import todo_write
from dostuff.tools.web_fetch import web_fetch
from dostuff.tools.web_search import web_search

# Parameters the harness supplies; never exposed to (or accepted from) the model.
HARNESS_PARAMS = frozenset({"mcp_client", "session_id", "turn_id", "adapter", "permissions"})


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool: implementation, permission kind, and harness-injected params."""
    name: str
    impl: Callable[..., Any]
    kind: ToolKind
    injected: frozenset[str] = frozenset()
    description_suffix: str = ""
    schema: dict[str, Any] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        schema = generate_litellm_tool_schema(
            self.impl,
            name=self.name,
            exclude=self.injected | HARNESS_PARAMS,
            description_suffix=self.description_suffix,
        )
        object.__setattr__(self, "schema", schema)


_TOOLS = [
    ToolSpec("read_file", read_file, "read"),
    ToolSpec("write_file", write_file, "write"),
    ToolSpec("edit_file", edit_file, "write"),
    ToolSpec("delete_file", delete_file, "write"),
    ToolSpec("list_files", list_files, "read"),
    ToolSpec("grep", grep, "read"),
    ToolSpec("glob", glob, "read"),
    ToolSpec("get_current_datetime", get_current_datetime, "read"),
    # Asks per domain: a URL can carry data out, so reaching the network is not read-only
    ToolSpec("web_fetch", web_fetch, "network"),
    # The query leaves the machine, so it asks too; "always" covers every search
    ToolSpec("web_search", web_search, "network"),
    # Planning only: never asks, and the TUI shows the list in full
    ToolSpec("todo_write", todo_write, "read", frozenset({"session_id"})),
    # Memory saves never ask; each one is shown in the TUI and listed by /memories.
    ToolSpec("save_memory", save_memory, "read"),
    ToolSpec("save_episode", save_episode, "read", frozenset({"session_id"})),
    ToolSpec("search_mcp_tools", _impl_search_mcp_tools, "read", frozenset({"mcp_client"})),
    ToolSpec("get_mcp_tool_details", _impl_get_mcp_tool_details, "read", frozenset({"mcp_client"})),
    ToolSpec("call_mcp_tool", _impl_call_mcp_tool, "mcp", frozenset({"mcp_client"})),
    # Delegation itself never asks: the subagent's own tool calls go through the permission layer.
    ToolSpec(
        "delegate_to_subagent",
        _impl_delegate_to_subagent,
        "read",
        frozenset({"mcp_client", "session_id", "turn_id", "adapter", "permissions"}),
    ),
    # dostuff's own config.yaml and MCP servers; changes ask in every mode (permissions.py)
    ToolSpec("read_settings", read_settings, "read"),
    ToolSpec("edit_config", edit_config, "settings", frozenset({"adapter"})),
    ToolSpec("set_mcp_server", set_mcp_server, "settings", frozenset({"adapter"})),
    ToolSpec("remove_mcp_server", remove_mcp_server, "settings", frozenset({"adapter"})),
    ToolSpec(
        "bash_command",
        bash_command,
        "shell",
        description_suffix=f"Commands run in {select_shell()[0]}.",
    ),
]

TOOL_MAP: dict[str, ToolSpec] = {spec.name: spec for spec in _TOOLS}

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {name: spec.schema for name, spec in TOOL_MAP.items()}

# Need chromadb (the `memory` extra); not offered to the model without it
MEMORY_TOOLS = frozenset({"save_memory", "save_episode"})


def default_tool_names() -> list[str]:
    """Tools offered to the main agent: all of them, minus the memory tools when memory is off."""
    from dostuff.lib.memory.chroma import chroma_available
    if chroma_available():
        return list(TOOL_SCHEMAS)
    return [name for name in TOOL_SCHEMAS if name not in MEMORY_TOOLS]
