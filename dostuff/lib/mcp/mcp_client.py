import asyncio
from contextlib import AsyncExitStack
import logging
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
from dostuff.helpers.mcp import token_store
from dostuff.lib.mcp.mcp_client_registration_store import MCPClientRegistrationStore
from dostuff.lib.mcp.mcp_tool_registry_store import MCPToolRegistryStore

logger = logging.getLogger(__name__)

timeout = httpx2.Timeout(
    connect=15.0,  # Time allowed to establish TCP connection
    read=60.0,     # Time allowed to wait for response data (wakeup time)
    write=10.0,
    pool=10.0
)

HTTP_TRANSPORTS = ("http", "streamable-http", "sse")
TOOL_KEY_SEPARATOR = "__"


def tool_key(server_name: str, tool_name: str) -> str:
    """Key under which an MCP tool is exposed to the model: `server__tool`."""
    return f"{server_name}{TOOL_KEY_SEPARATOR}{tool_name}"


class _ServerConnection:
    """A server's session plus the task that owns its context managers.

    anyio cancel scopes must be exited in the task that entered them, so each server's
    transports are opened and closed inside one long-lived task.
    """

    def __init__(self) -> None:
        self.task: asyncio.Task | None = None
        self.stop = asyncio.Event()


class MCPClient:
    def __init__(self, registration_store: MCPClientRegistrationStore, tool_registry_store: MCPToolRegistryStore):
        self.servers: dict[str, ClientSession] = {}
        # server__tool -> {"server_name", "tool_name", "schema"}
        self.mcp_tools: dict[str, dict[str, Any]] = {}
        self._tokens: dict[str, dict] = {}
        self.server_meta: dict[str, dict] = {}
        self._connections: dict[str, _ServerConnection] = {}
        self.registration_store = registration_store
        self.tool_registry = tool_registry_store

    def resolve_tool(self, name: str) -> dict[str, Any] | None:
        """Look up a tool by `server__tool` key, or by bare tool name when that is unambiguous."""
        if name in self.mcp_tools:
            return self.mcp_tools[name]
        matches = [info for info in self.mcp_tools.values() if info["tool_name"] == name]
        return matches[0] if len(matches) == 1 else None

    async def connect_to_server(
        self,
        server_name: str,
        command: str | None = sys.executable,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        transport: str = "stdio",
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """(Re)connects to an MCP server and updates the tool index. Raises if connecting fails."""
        await self._disconnect(server_name)

        req_headers = dict(headers or {})
        self.server_meta[server_name] = {
            "url": url,
            "transport": transport,
            "headers": req_headers,
            "command": command,
            "args": args,
            "env": env,
        }
        if transport in HTTP_TRANSPORTS or url:
            if not url:
                raise ValueError(f"Server '{server_name}' requires a 'url' for transport '{transport}'.")
            if "Authorization" not in req_headers:
                token = await self._get_or_refresh_token(server_name, url)
                if token:
                    req_headers["Authorization"] = f"Bearer {token}"
        elif not command:
            raise ValueError(f"Server '{server_name}' requires a 'command' for stdio transport.")

        conn = _ServerConnection()
        ready: asyncio.Future[ClientSession] = asyncio.get_running_loop().create_future()
        conn.task = asyncio.create_task(
            self._serve(server_name, transport, url, req_headers, command, args, env, ready, conn.stop),
            name=f"mcp:{server_name}",
        )
        self._connections[server_name] = conn
        try:
            session = await ready
        except BaseException:
            conn.task.cancel()
            await asyncio.gather(conn.task, return_exceptions=True)
            self._connections.pop(server_name, None)
            raise

        self.servers[server_name] = session
        tools_response = await session.list_tools()
        await self.tool_registry.register_tools(server_name, tools_response.tools)
        for key in [k for k, v in self.mcp_tools.items() if v["server_name"] == server_name]:
            del self.mcp_tools[key]
        for tool in tools_response.tools:
            key = tool_key(server_name, tool.name)
            if key in self.mcp_tools:
                logger.warning("MCP server '%s' lists tool '%s' more than once; keeping the first", server_name, tool.name)
                continue
            self.mcp_tools[key] = {
                "server_name": server_name,
                "tool_name": tool.name,
                "schema": {
                    "name": key,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema
                },
            }

    async def _serve(self, server_name, transport, url, req_headers, command, args, env, ready: asyncio.Future, stop: asyncio.Event) -> None:
        """Owns one server's transports: opens them, hands back the session, closes on stop."""
        try:
            async with AsyncExitStack() as stack:
                if transport in HTTP_TRANSPORTS or url:
                    if transport in ("http", "streamable-http"):
                        http_client = await stack.enter_async_context(httpx2.AsyncClient(headers=req_headers, timeout=timeout, event_hooks={"response": [self._log_response]}))
                        self.server_meta[server_name]["http_client"] = http_client
                        read, write, *_ = await stack.enter_async_context(streamable_http_client(url=url, http_client=http_client))
                    else:
                        read, write = await stack.enter_async_context(sse_client(url=url, headers=req_headers))
                else:
                    merged_env = os.environ.copy()
                    if env:
                        merged_env.update(env)
                    server_params = StdioServerParameters(command=command, args=args or [], env=merged_env)
                    read, write = await stack.enter_async_context(stdio_client(server_params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                if not ready.done():
                    ready.set_result(session)
                await stop.wait()
        except asyncio.CancelledError:
            if not ready.done():
                ready.set_exception(RuntimeError(f"Connecting to MCP server '{server_name}' was cancelled."))
            raise
        except BaseException as e:  # incl. anyio's BaseExceptionGroup
            if not ready.done():
                ready.set_exception(e if isinstance(e, Exception) else RuntimeError(f"MCP server '{server_name}' failed: {e!r}"))
            else:
                # The connection died after it was up; forget the dead session
                logger.warning("MCP server '%s' disconnected: %r", server_name, e)
                if not ready.cancelled() and ready.exception() is None and self.servers.get(server_name) is ready.result():
                    self.servers.pop(server_name, None)

    async def _disconnect(self, server_name: str, timeout_s: float = 3.0) -> None:
        conn = self._connections.pop(server_name, None)
        self.servers.pop(server_name, None)
        if conn is None or conn.task is None:
            return
        conn.stop.set()
        try:
            await asyncio.wait_for(asyncio.shield(conn.task), timeout=timeout_s)
        except (asyncio.TimeoutError, Exception):
            conn.task.cancel()
            await asyncio.gather(conn.task, return_exceptions=True)

    async def reauthenticate(self, server_name: str) -> None:
        """Drop the cached token and reconnect (runs the OAuth flow again if needed)."""
        meta = self.server_meta.get(server_name, {})
        url = meta.get("url")
        self._tokens.pop(server_name, None)
        if url:
            token_store.delete_token(url)
        headers = dict(meta.get("headers") or {})
        headers.pop("Authorization", None)
        await self.connect_to_server(
            server_name=server_name,
            transport=meta.get("transport", "streamable-http"),
            command=meta.get("command") or sys.executable,
            args=meta.get("args"),
            env=meta.get("env"),
            url=url,
            headers=headers,
        )

    async def _get_or_refresh_token(self, server_name: str, url: str) -> str | None:
        """Cached (memory, then keyring) token, refreshed if expired; else probe the server and
        run the OAuth flow when it answers 401."""
        cached = self._tokens.get(server_name) or token_store.load_token(url)

        if cached:
            if time.time() < cached["expires_at"] - 60:  # 60s safety margin
                self._tokens[server_name] = cached
                return cached["access_token"]
            if cached.get("refresh_token"):
                try:
                    token_data = await refresh_access_token(
                        cached["token_endpoint"], cached["refresh_token"],
                        client_id=cached.get("client_id"), resource=cached.get("resource"),
                    )
                    token_data.setdefault("refresh_token", cached["refresh_token"])
                    token_data["client_id"] = cached.get("client_id")
                    token_data["resource"] = cached.get("resource")
                    self._store_token(server_name, url, cached["token_endpoint"], token_data)
                    return token_data["access_token"]
                except httpx.HTTPError:
                    pass  # refresh token expired/revoked: fall through to a fresh flow

        try:
            async with httpx.AsyncClient() as probe_client:
                res = await probe_client.post(
                    url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, timeout=5.0
                )
        except httpx.HTTPError:
            return None  # unreachable now; the transport will report the real error

        if res.status_code == 401:
            www_auth = res.headers.get("WWW-Authenticate", "")
            token_data = await authenticate_via_oauth(url, www_auth, self.registration_store, server_name)
            self._store_token(server_name, url, token_data["token_endpoint"], token_data)
            return token_data["access_token"]
        return None

    def _store_token(self, server_name: str, url: str, token_endpoint: str, token_data: dict) -> None:
        token = {
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "expires_at": time.time() + token_data.get("expires_in", 3600),
            "token_endpoint": token_endpoint,
            "client_id": token_data.get("client_id"),
            "resource": token_data.get("resource"),
        }
        self._tokens[server_name] = token
        token_store.save_token(url, token)

    async def cleanup(self) -> None:
        """Close every server connection (each in the task that opened it)."""
        await asyncio.gather(*(self._disconnect(name) for name in list(self._connections)), return_exceptions=True)
        self.servers.clear()
        self.mcp_tools.clear()

    async def _log_response(self, response):
        if response.is_error:
            await response.aread()
