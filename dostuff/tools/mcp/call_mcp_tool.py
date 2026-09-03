from typing import Any
from dostuff.lib.mcp.mcp_client import MCPClient
import time
from mcp.shared.exceptions import MCPError
import asyncio

async def call_mcp_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute a discovered MCP tool with the provided arguments.

    Args:
        name: Name of the target MCP tool to execute.
        arguments: Key-value dictionary of arguments matching the tool's input schema.
    """
    return ""


def _extract_text_from_block(block: Any) -> str | None:
    text = getattr(block, "text", None)
    if isinstance(text, str) and text:
        return text
    return None

def _format_tool_result(result) -> str:
    text_parts = [text for block in result.content if (text := _extract_text_from_block(block)) is not None]
    return "\n".join(text_parts) if text_parts else "Tool executed successfully."    


async def _impl_call_mcp_tool(
    name: str, arguments: dict[str, Any], mcp_client: MCPClient
) -> str:
    if name not in mcp_client.mcp_tools:
        raise KeyError(f"Tool '{name}' not found across any connected MCP servers.")

    server_name = mcp_client.mcp_tools[name]["server_name"]
    session = mcp_client.servers[server_name]
    server_info = mcp_client.server_meta.get(server_name, {})
    url = server_info.get("url")

    try:
        result = await session.call_tool(name, arguments)
        
        return _format_tool_result(result)
    except Exception as err:
        from dostuff.helpers.ui.emit import emit
        emit(f"Error calling MCP tool '{name}' on server '{server_name}': {err}", msg_type="error")
        is_mcp_internal_error = isinstance(err, MCPError) and err.error.code == -32603

        cached = mcp_client._tokens.get(server_name)
        token_likely_expired = cached is not None and time.time() >= cached["expires_at"]
    
        is_retryable_auth_failure = is_mcp_internal_error and token_likely_expired
    
        if not is_retryable_auth_failure or not url or server_info.get("transport") not in ("http", "streamable-http", "sse"):
            raise err

        from dostuff.helpers.ui.emit import emit
        emit(f"Attempting to refresh token for server '{server_name}' and retry tool call '{name}'...", msg_type="system")
        if "headers" in server_info and "Authorization" in server_info["headers"]:
            del server_info["headers"]["Authorization"]

        mcp_client._tokens.pop(server_name, None)

        try:
            await asyncio.wait_for(
                mcp_client.connect_to_server(
                    server_name=server_name,
                    transport=server_info.get("transport", "streamable-http"),
                    url=url,
                    headers=server_info.get("headers")
                ),
                timeout=120,  # 2 minutes to complete re-auth, or give up
            )
        except asyncio.TimeoutError:
            raise Exception(f"Re-authentication for '{server_name}' timed out — token refresh requires manual browser approval.")
        
        from dostuff.helpers.ui.emit import emit
        emit("after connect to server", msg_type="system")     
        fresh_session = mcp_client.servers[server_name]          
        from dostuff.helpers.ui.emit import emit
        emit(f"Retrying tool call '{name}' with fresh connection channel...", msg_type="system")
        result = await fresh_session.call_tool(name, arguments)

        return _format_tool_result(result)       
