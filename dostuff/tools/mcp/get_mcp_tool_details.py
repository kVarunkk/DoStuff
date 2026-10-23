from dostuff.lib.mcp.mcp_client import MCPClient
import json


def _impl_get_mcp_tool_details(name: str, mcp_client: MCPClient) -> str:
    """Inspect and return the full JSON input schema for a specific MCP tool.

    Args:
        name: Tool name as returned by search_mcp_tools (`server__tool`).
    """
    tool_info = mcp_client.resolve_tool(name)
    if tool_info is None:
        raise ValueError(f"MCP Tool '{name}' not found.")
    return json.dumps(tool_info.get("schema", {}), indent=2)
