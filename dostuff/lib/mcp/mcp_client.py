
from contextlib import AsyncExitStack
import time
import httpx
import httpx2
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from typing import Any, Dict, List, Optional
import sys
import os
from dostuff.helpers.mcp.mcp_oauth import authenticate_via_oauth, refresh_access_token
from dostuff.lib.mcp.mcp_client_registration_store import MCPClientRegistrationStore
from dostuff.lib.mcp.mcp_tool_registry_store import MCPToolRegistryStore

timeout = httpx2.Timeout(
    connect=15.0,  # Time allowed to establish TCP connection
    read=60.0,     # Time allowed to wait for response data (wakeup time)
    write=10.0,
    pool=10.0
)

class MCPClient:
    def __init__(self, registration_store: MCPClientRegistrationStore, tool_registry_store: MCPToolRegistryStore):
        self.servers: dict[str, ClientSession] = {}
        self.mcp_tools: dict[str, dict[str, Any]] = {}
        self._tokens: dict[str, dict] = {}
        self.server_meta: dict[str, dict] = {}
        self.server_stacks: dict[str, AsyncExitStack] = {}
        self._retired_stacks: list[AsyncExitStack] = []
        self.registration_store = registration_store
        self.tool_registry = tool_registry_store

    async def connect_to_server(
        self,
        server_name: str,
        command: str = sys.executable,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        transport: str = "stdio",
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Establishes an async stdio session with an MCP server and updates tool index."""
        
        if server_name in self.server_stacks:
                self._retired_stacks.append(self.server_stacks[server_name])
        stack = AsyncExitStack()
        self.server_stacks[server_name] = stack
        
        req_headers = headers or {}

        self.server_meta[server_name] = {
            "url": url,
            "transport": transport,
            "headers": req_headers,
        }

        if transport in ("http", "streamable-http", "sse") or url:
            if not url:
                raise ValueError(f"Server '{server_name}' requires a 'url' for transport '{transport}'.")

            if "Authorization" not in req_headers:
                token = await self._get_or_refresh_token(server_name, url)
                if token:
                    req_headers["Authorization"] = f"Bearer {token}"

            if transport in ("http", "streamable-http"):
                if streamable_http_client is None:
                    raise ImportError(
                        "streamable_http_client not available — upgrade with `pip install --upgrade mcp`."
                    )
                http_client = await stack.enter_async_context(httpx2.AsyncClient(headers=req_headers, timeout=timeout, event_hooks={"response": [self._log_response]}))
                self.server_meta[server_name]["http_client"] = http_client
                http_ctx = streamable_http_client(url=url, http_client=http_client)
                read, write, *_ = await stack.enter_async_context(http_ctx)
            else:
                sse_ctx = sse_client(url=url, headers=req_headers)
                read, write = await stack.enter_async_context(sse_ctx)

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

        else:
            if not command:
                raise ValueError(f"Server '{server_name}' requires a 'command' for stdio transport.")

            merged_env = os.environ.copy()
            if env:
                merged_env.update(env)

            server_params = StdioServerParameters(command=command, args=args or [], env=merged_env)
            read, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

        self.servers[server_name] = session

        tools_response = await session.list_tools()
        await self.tool_registry.register_tools(server_name, tools_response.tools)
        for tool in tools_response.tools:
            self.mcp_tools[tool.name] = {
                "server_name": server_name,
                "schema": {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema
                },
            }

        # print(f"Connected to '{server_name}' ({len(tools_response.tools)} tools indexed)")

    async def _get_or_refresh_token(self, server_name: str, url: str) -> str | None:
        """Probes the server for an auth challenge and runs the OAuth flow if needed.
        Caches the token per server for the lifetime of this MCPClient instance.

        Note: does not yet handle token expiry/refresh mid-session — a 401 on a
        later call_tool would currently fail rather than re-authenticating.
        """

        cached = self._tokens.get(server_name)
    
        if cached:
            if time.time() < cached["expires_at"] - 60:  # 60s safety margin
                return cached["access_token"]
    
            if cached.get("refresh_token"):
                token_data = await refresh_access_token(cached["token_endpoint"], cached["refresh_token"])
                self._store_token(server_name, cached["token_endpoint"], token_data)
                return token_data["access_token"]

        try:
            async with httpx.AsyncClient() as probe_client:
                res = await probe_client.post(
                    url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, timeout=5.0
                )

            if res.status_code == 401:
                www_auth = res.headers.get("WWW-Authenticate", "")
                token_data = await authenticate_via_oauth(url, www_auth, self.registration_store, server_name)
                self._store_token(server_name, token_data["token_endpoint"], token_data)
                return token_data["access_token"]

        except Exception as e:
            # print(f"[{server_name}] Auth pre-flight check failed: {e}")
            pass

        return None

    def _store_token(self, server_name: str, token_endpoint: str, token_data: dict) -> None:
        self._tokens[server_name] = {
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "expires_at": time.time() + token_data.get("expires_in", 3600),
            "token_endpoint": token_endpoint,
        }

    async def cleanup(self) -> None:
        import contextlib, sys, os
        # Suppress noisy anyio/mcp generator-close messages (library-level noise)
        with open(os.devnull, 'w') as null, contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
            # 1. First-In, Last-Out (FILO) teardown for active stacks
            for name, stack in reversed(list(self.server_stacks.items())):
                try:
                    await stack.aclose()
                except BaseException:
                    pass
            # 2. Teardown retired stacks in FILO order
            for stack in reversed(self._retired_stacks):
                try:
                    await stack.aclose()
                except BaseException:
                    pass
        # Restore references
        self._retired_stacks.clear()
        self.server_stacks.clear()
        self.servers.clear()
        self.mcp_tools.clear()

    async def _log_response(self, response):
        if response.is_error:
            await response.aread()
            # print(f"HTTP ERROR {response.status_code}: {response.text}")    


