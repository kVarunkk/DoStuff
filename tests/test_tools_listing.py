"""/tools: the listing of built-in and MCP tools."""
from dostuff.agent.subagents import RESTRICTED_SUBAGENT_TOOLS
from dostuff.helpers.ui.tools_listing import tools_listing
from dostuff.tools.definitions import default_tool_names


def _row(text: str, name: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"  {name} "))


def test_every_offered_tool_is_listed_with_how_it_asks_and_subagent_access():
    text = tools_listing()
    for name in default_tool_names():
        row = _row(text, name)
        assert ("  no " in row) == (name in RESTRICTED_SUBAGENT_TOOLS), row
    assert "never" in _row(text, "read_file")
    assert "unless read-only" in _row(text, "bash_command")
    assert "always" in _row(text, "edit_config")
    assert "Read a text file" in _row(text, "read_file")  # first sentence of the description


def test_mcp_tools_are_grouped_per_server():
    mcp = {
        "gh__list_issues": {"server_name": "gh", "tool_name": "list_issues",
                            "schema": {"description": "List issues in a repo. Paginated."}},
        "db__query": {"server_name": "db", "tool_name": "query", "schema": {"description": None}},
    }
    lines = tools_listing(mcp).splitlines()
    assert "MCP tools (2)" in "\n".join(lines)
    assert lines.index("  db") < lines.index("    query") < lines.index("  gh")
    assert "    list_issues  List issues in a repo." in lines


def test_no_mcp_servers_says_how_to_add_one():
    assert "MCP tools: none" in tools_listing({})
