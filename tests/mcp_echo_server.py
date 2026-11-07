"""Minimal stdio MCP server used by test_mcp_e2e.py."""
import os

from mcp.server.mcpserver import MCPServer

server = MCPServer("echo")


@server.tool()
def echo(text: str) -> str:
    """Echo the given text back."""
    return f"echo: {text}"


@server.tool()
def read_env(name: str) -> str:
    """The value of an environment variable."""
    return os.environ.get(name, "")


if __name__ == "__main__":
    server.run()
