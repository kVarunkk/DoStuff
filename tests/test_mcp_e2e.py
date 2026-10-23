"""End-to-end MCP: connect to a real stdio server, search, call, clean up on one loop (REQ-4.2)."""
import asyncio
import sys
from pathlib import Path

import pytest

from dostuff.lib.mcp.mcp_client import MCPClient
from dostuff.tools.mcp.call_mcp_tool import _impl_call_mcp_tool
from dostuff.tools.mcp.search_mcp_tools import _impl_search_mcp_tools

SERVER = Path(__file__).with_name("mcp_echo_server.py")


class NullRegistry:
    async def register_tools(self, server_name, tools):
        pass

    async def search_tools(self, user_intent, top_k=5, server_names=None):
        return []


def test_connect_call_and_cleanup_on_one_loop():
    pytest.importorskip("mcp.server.mcpserver")

    async def scenario():
        client = MCPClient(registration_store=None, tool_registry_store=NullRegistry())  # type: ignore[arg-type]
        try:
            await asyncio.wait_for(
                client.connect_to_server("echo", command=sys.executable, args=[str(SERVER)]),
                timeout=60,
            )
            assert "echo__echo" in client.mcp_tools
            assert "echo__echo" in await _impl_search_mcp_tools("echo some text", client)
            assert await _impl_call_mcp_tool("echo__echo", {"text": "hi"}, client) == "echo: hi"
            # Reconnecting replaces the old connection cleanly
            await client.connect_to_server("echo", command=sys.executable, args=[str(SERVER)])
            assert await _impl_call_mcp_tool("echo", {"text": "again"}, client) == "echo: again"
        finally:
            await client.cleanup()
        assert client.servers == {} and client._connections == {}

    asyncio.run(scenario())
