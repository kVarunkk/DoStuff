from typing import Any
from dostuff.lib.mcp.mcp_client import MCPClient
import json 

def get_mcp_tool_details(name: str) -> str:
    """Inspect and return the full JSON input schema for a specific MCP tool.
    
    Args:
        name: Exact name of the MCP tool discovered via search_tools.
    """
    return ""

def _impl_get_mcp_tool_details(name: str, mcp_client: MCPClient) -> str:
    if name not in mcp_client.mcp_tools:
        raise ValueError(f"MCP Tool '{name}' not found.")
    tool_info = mcp_client.mcp_tools[name]
    schema = tool_info.get("schema", {})

    return json.dumps(schema, indent=2)