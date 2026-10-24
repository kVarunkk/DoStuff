"""Build and check one `mcpServers` entry; shared by `dostuff mcp add` and the set_mcp_server tool."""
from typing import Any

TRANSPORTS = ("stdio", "http", "sse", "streamable-http")


def build_server_config(
    command: list[str] | None = None,
    url: str | None = None,
    transport: str | None = None,
    env: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The entry for a stdio server (`command` = program + arguments) or an HTTP one (`url`).

    Raises ValueError with a message for the user when the combination doesn't make sense.
    """
    if bool(url) == bool(command):
        raise ValueError("give either a command (stdio) or a url (http), not both")
    transport = (transport or ("http" if url else "stdio")).lower()
    if transport not in TRANSPORTS:
        raise ValueError("transport must be stdio, http or sse")
    if url and transport == "stdio":
        raise ValueError("a url needs transport http or sse")
    if command and transport != "stdio":
        raise ValueError("a command needs transport stdio")
    cfg: dict[str, Any] = {"transport": transport}
    if command:
        cfg.update(command=command[0], args=list(command[1:]))
        if env:
            cfg["env"] = dict(env)
    else:
        cfg["url"] = url
        if headers:
            cfg["headers"] = dict(headers)
    return cfg
