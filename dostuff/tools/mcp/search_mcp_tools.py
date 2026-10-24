import json
import re

from dostuff.lib.mcp.mcp_client import MCPClient, tool_key

_MAX_RESULTS = 5


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 1}


def keyword_search(query: str, mcp_client: MCPClient, top_k: int = _MAX_RESULTS) -> list[dict]:
    """Rank connected tools by how many query tokens appear in their name/description."""
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    scored = []
    for key, info in mcp_client.mcp_tools.items():
        desc = info["schema"].get("description") or ""
        haystack = _tokens(f"{info['tool_name']} {desc}")
        overlap = len(query_tokens & haystack)
        if overlap:
            scored.append((overlap, key, desc, info["server_name"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [{"name": key, "description": desc, "server": server} for _, key, desc, server in scored[:top_k]]


async def _impl_search_mcp_tools(query: str, mcp_client: MCPClient) -> str:
    """Search tools of the connected MCP servers by keyword or description.

    Args:
        query: Keywords or description to search for matching tools.

    Returns:
        JSON list of matching tools (name, description, server). Pass `name` to
        get_mcp_tool_details and call_mcp_tool.
    """
    if not query.strip():
        return "Query cannot be empty."

    connected = list(mcp_client.servers)
    results = []

    # Primary: semantic search in the tool index, limited to servers connected right now
    registry = getattr(mcp_client, "tool_registry", None)  # None without chromadb
    if registry is not None and connected:
        try:
            for tool in await registry.search_tools(user_intent=query, top_k=_MAX_RESULTS, server_names=connected):
                key = tool_key(tool["server_name"], tool["name"])
                if key in mcp_client.mcp_tools:
                    results.append({"name": key, "description": tool["description"], "server": tool["server_name"]})
        except Exception:
            results = []

    # Fallback: token-overlap search over the in-memory tool list
    if not results:
        results = keyword_search(query, mcp_client)

    if not results:
        return f"No MCP tools found matching query: '{query}'."

    return json.dumps(results, indent=2)
