"""Build and check one `mcpServers` entry; shared by `dostuff mcp add` and the set_mcp_server tool."""
import os
import re
from typing import Any

from dostuff.helpers.config_secrets import MCP_SECRET_HINT, find_secrets

TRANSPORTS = ("stdio", "http", "sse", "streamable-http")
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_]\w*)\}")
ENV_FIX = "add it to .dostuff/.env and restart dostuff"


def _references(values: Any) -> list[str]:
    if not isinstance(values, dict):
        return []
    return [name for value in values.values() if isinstance(value, str) for name in _ENV_REFERENCE.findall(value)]


def missing_env_references(cfg: dict[str, Any]) -> list[str]:
    """The `${NAME}`s in a server's `headers` and `env` with no value; an empty one could not sign in either."""
    names = _references(cfg.get("headers")) + _references(cfg.get("env"))
    return sorted({name for name in names if not os.environ.get(name)})


def expand_env_references(values: dict[str, Any] | None) -> dict[str, Any]:
    """`values` with each `${NAME}` replaced from the environment, which `.env` is loaded into."""
    return {key: _ENV_REFERENCE.sub(lambda m: os.environ[m[1]], value) if isinstance(value, str) else value
            for key, value in (values or {}).items()}


def key_warnings(servers: dict[str, Any]) -> list[str]:
    """What's wrong with the servers' keys: a `${NAME}` with no value, a plain key. Names only, never a value."""
    warnings = [f"MCP server '{name}': {', '.join(missing)} not set: {ENV_FIX}"
                for name, cfg in servers.items() if isinstance(cfg, dict)
                for missing in [missing_env_references(cfg)] if missing]
    return warnings + [f"{where} in the MCP config looks like a key: {MCP_SECRET_HINT}" for where in find_secrets(servers)]


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
