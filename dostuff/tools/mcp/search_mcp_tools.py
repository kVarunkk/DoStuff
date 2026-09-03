from dostuff.lib.mcp.mcp_client import MCPClient
import json

async def search_mcp_tools(query: str) -> str:
    """Search available MCP tools by keyword across connected MCP servers.
    
    Args:
        query: Keywords or description to search for matching tools.
        
    Returns:
        String of matching tools with tool name and short description only.
    """
    return ""

async def _impl_search_mcp_tools(query: str, mcp_client: MCPClient) -> str:
    """Discovers relevant MCP tools using semantic vector search in ChromaDB,
    falling back to substring search if vector search returns no results.
    """
    if not query.strip():
        return "Query cannot be empty."

    results = []

    # Primary: Semantic Vector Search via ChromaDB
    if hasattr(mcp_client, "tool_registry") and mcp_client.tool_registry:
        try:
            matched_tools = await mcp_client.tool_registry.search_tools(
                user_intent=query, 
                top_k=5
            )
            for tool in matched_tools:
                results.append({
                    "name": tool["name"],
                    "description": tool["description"],
                    "server": tool["server_name"],
                })
        except Exception as e:
            print(f"[ToolRegistry] Semantic search failed ({e}), falling back to memory search.")

    # Fallback: In-memory keyword search if vector search fails or yields no results
    if not results and hasattr(mcp_client, "mcp_tools"):
        query_lower = query.lower()
        for tool_name, tool_info in mcp_client.mcp_tools.items():
            desc = tool_info["schema"].get("description", "")
            if query_lower in tool_name.lower() or query_lower in desc.lower():
                results.append({
                    "name": tool_name,
                    "description": desc,
                    "server": tool_info["server_name"],
                })

    if not results:
        return f"No MCP tools found matching query: '{query}'."

    return json.dumps(results, indent=2)