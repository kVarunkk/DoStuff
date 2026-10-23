import json
from pathlib import Path
from typing import Dict, Any
from dostuff.helpers.ui.emit import emit


def global_mcp_config_path() -> Path:
    return Path.home() / ".dostuff" / "mcp_config.json"


def project_mcp_config_path() -> Path:
    return Path.cwd() / ".dostuff" / "mcp_config.json"


def read_mcp_file(file_path: Path) -> Dict[str, Any]:
    """The `mcpServers` object of one config file ({} if the file is missing). Raises on bad JSON."""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    servers = data.get("mcpServers", {}) if isinstance(data, dict) else {}
    return servers if isinstance(servers, dict) else {}


def write_mcp_file(file_path: Path, servers: Dict[str, Any]) -> None:
    """Replace the `mcpServers` object of a config file, keeping any other top-level keys."""
    data: Dict[str, Any] = {}
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        data = loaded if isinstance(loaded, dict) else {}
    data["mcpServers"] = servers
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def mcp_config_sources(config_path: str | None = None) -> list[Path]:
    """Files that make up the MCP config, lowest priority first (later files override by server name).

    An explicit path (argument, or `mcp.config_path` in config.yaml) is used alone. Otherwise
    ~/.dostuff/mcp_config.json, then the project's .dostuff/mcp_config.json; ./mcp_config.json
    (legacy) only when neither exists.
    """
    if not config_path:
        from dostuff.config import get_config
        mcp_block = get_config().raw.get("mcp")
        config_path = mcp_block.get("config_path") if isinstance(mcp_block, dict) else None
    if config_path:
        return [Path(str(config_path)).expanduser()]
    sources = [p for p in (global_mcp_config_path(), project_mcp_config_path()) if p.exists()]
    legacy = Path("mcp_config.json")
    if not sources and legacy.exists():
        sources = [legacy]
    return sources


def load_mcp_config(config_path: str | None = None) -> Dict[str, Any]:
    """Loads MCP server configurations from JSON: global, overridden per server by the project file."""
    servers: Dict[str, Any] = {}
    sources = mcp_config_sources(config_path)
    for file_path in sources:
        try:
            servers.update(read_mcp_file(file_path))
            emit(f"[MCP] Loaded config from: {file_path}", msg_type="system")
        except Exception as e:
            emit(f"❌ Failed to load '{file_path}': {e}", msg_type="error")
    if not sources:
        emit("⚠️  Config file not found (tried global + .dostuff/ + cwd). Starting without MCP servers.", msg_type="error")
    return servers
