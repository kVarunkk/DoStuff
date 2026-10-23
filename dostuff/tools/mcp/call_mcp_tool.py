import asyncio
import json
import time
from typing import Any

from mcp.shared.exceptions import MCPError

from dostuff.lib.mcp.mcp_client import HTTP_TRANSPORTS, MCPClient

REAUTH_TIMEOUT_S = 120


def _extract_text_from_block(block: Any) -> str | None:
    text = getattr(block, "text", None)
    if isinstance(text, str) and text:
        return text
    return None


def _format_tool_result(result) -> str:
    text_parts = [text for block in result.content if (text := _extract_text_from_block(block)) is not None]
    return "\n".join(text_parts) if text_parts else "Tool executed successfully."


def _status_code(err: BaseException) -> int | None:
    response = getattr(err, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _is_auth_failure(err: BaseException, mcp_client: MCPClient, server_name: str) -> bool:
    """HTTP 401 anywhere in the error chain, or an internal error while our token is expired."""
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _status_code(current) == 401 or "401 Unauthorized" in str(current):
            return True
        nested = getattr(current, "exceptions", None)  # exception groups from anyio
        if nested:
            if any(_is_auth_failure(e, mcp_client, server_name) for e in nested):
                return True
        current = current.__cause__ or current.__context__
    cached = mcp_client._tokens.get(server_name)
    token_expired = cached is not None and time.time() >= cached["expires_at"]
    return isinstance(err, MCPError) and err.error.code == -32603 and token_expired


async def _impl_call_mcp_tool(
    name: str, arguments: dict[str, Any], mcp_client: MCPClient
) -> str:
    """Execute a discovered MCP tool with the provided arguments.

    Args:
        name: Tool name as returned by search_mcp_tools (`server__tool`).
        arguments: Key-value dictionary of arguments matching the tool's input schema.
    """
    tool_info = mcp_client.resolve_tool(name)
    if tool_info is None:
        raise KeyError(f"Tool '{name}' not found across any connected MCP servers.")

    # Tolerant fallback: some models (via litellm) send `arguments` as a JSON-encoded string
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            raise TypeError(
                f"Tool '{name}' received arguments as a string that is not valid JSON: {arguments!r}"
            )
    if not isinstance(arguments, dict):
        raise TypeError(
            f"Tool '{name}' arguments must be a dict, got {type(arguments).__name__}"
        )

    server_name = tool_info["server_name"]
    tool_name = tool_info["tool_name"]
    session = mcp_client.servers.get(server_name)
    if session is None:
        raise ConnectionError(f"MCP server '{server_name}' is not connected.")

    try:
        return _format_tool_result(await session.call_tool(tool_name, arguments))
    except Exception as err:
        server_info = mcp_client.server_meta.get(server_name, {})
        is_http = bool(server_info.get("url")) and server_info.get("transport") in HTTP_TRANSPORTS
        if not is_http or not _is_auth_failure(err, mcp_client, server_name):
            raise

    # Auth failure: re-authenticate once and retry on the fresh connection
    from dostuff.helpers.ui.emit import emit
    emit(f"Re-authenticating with MCP server '{server_name}'...", msg_type="system")
    try:
        await asyncio.wait_for(mcp_client.reauthenticate(server_name), timeout=REAUTH_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise Exception(f"Re-authentication for '{server_name}' timed out — it needs browser approval.")
    try:
        return _format_tool_result(await mcp_client.servers[server_name].call_tool(tool_name, arguments))
    except Exception as err:
        emit(f"MCP tool '{name}' failed after re-authenticating: {err}", msg_type="error")
        raise
