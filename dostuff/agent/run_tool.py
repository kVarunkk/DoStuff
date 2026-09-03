from dostuff.tools.definitions import TOOL_MAP
from typing import Any
from dostuff.lib.tracing import traced
import inspect
import asyncio
from dostuff.tools.mcp.call_mcp_tool import _impl_call_mcp_tool
from dostuff.tools.mcp.get_mcp_tool_details import _impl_get_mcp_tool_details
from dostuff.tools.mcp.search_mcp_tools import _impl_search_mcp_tools
from dostuff.lib.mcp.mcp_client import MCPClient
from dostuff.tools.delegate_to_subagent import _impl_delegate_to_subagent

MCP_META_TOOLS = {
    "search_mcp_tools": _impl_search_mcp_tools,
    "get_mcp_tool_details": _impl_get_mcp_tool_details,
    "call_mcp_tool": _impl_call_mcp_tool,
}

SUBAGENT_META_TOOLS = {
    "delegate_to_subagent": _impl_delegate_to_subagent
}

@traced("tool_call")
async def run_tool(fn_name: str | None, fn_args: dict[str, Any], mcp_client: MCPClient | None = None, session_id: str | None = None, turn_id: str | None = None) -> Any:
    if not isinstance(fn_name, str):
        raise ValueError("Tool name must be a string.")

    if fn_name in SUBAGENT_META_TOOLS:
        if not session_id or not turn_id or not mcp_client:
            raise ValueError(f"Cannot spawn subagent without all required arguments.")
        fn = SUBAGENT_META_TOOLS[fn_name]
        if inspect.iscoroutinefunction(fn):
            return await fn(**fn_args, mcp_client=mcp_client, session_id = session_id, turn_id=turn_id)
        else:
            return await asyncio.to_thread(fn, **fn_args, mcp_client=mcp_client, session_id = session_id, turn_id=turn_id)

    if fn_name in MCP_META_TOOLS:
        if not mcp_client:
            raise ValueError(f"Cannot execute MCP tool '{fn_name}' without an active mcp_client.")
        fn = MCP_META_TOOLS[fn_name]
        if inspect.iscoroutinefunction(fn):
            return await fn(**fn_args, mcp_client=mcp_client)
        else:
            return await asyncio.to_thread(fn, **fn_args, mcp_client=mcp_client)

    if fn_name not in TOOL_MAP:
        raise KeyError(f"Tool '{fn_name}' not found.")

    fn = TOOL_MAP[fn_name]
    if inspect.iscoroutinefunction(fn):
        return await fn(**fn_args)
    else:
        return await asyncio.to_thread(fn, **fn_args)