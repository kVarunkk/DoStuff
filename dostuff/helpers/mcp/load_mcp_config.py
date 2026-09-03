import json
from pathlib import Path
from typing import Dict, Any

def load_mcp_config(config_path: str | None = None) -> Dict[str, Any]:
    """Loads MCP server configurations from JSON.
    Checks global ~/.dostuff/mcp_config.json first, then .dostuff/mcp_config.json (project), then cwd.
    """
    candidates = []
    # 1. Explicit path
    if config_path:
        candidates.append(Path(config_path))
    # 2. Global
    candidates.append(Path.home() / ".dostuff" / "mcp_config.json")
    # 3. Project override
    candidates.append(Path.cwd() / ".dostuff" / "mcp_config.json")
    # 4. CWD fallback (legacy)
    candidates.append(Path("mcp_config.json"))

    for file_path in candidates:
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(f"[MCP] Loaded config from: {file_path}")
                    return data.get("mcpServers", {})
            except Exception as e:
                print(f"❌ Failed to load '{file_path}': {e}")
                return {}

    print("⚠️  Config file not found (tried global + .dostuff/ + cwd). Starting without MCP servers.")
    return {}