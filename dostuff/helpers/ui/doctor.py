"""`dostuff doctor` and /doctor: setup checks, each with a fix for a problem."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Check:
    label: str
    ok: bool | None  # None: information only
    fix: str = ""


def doctor_checks() -> list[Check]:
    """Config files, the model's API key, MCP config, data dir, memory and permission mode."""
    from dostuff.config import get_config
    from dostuff.helpers.mcp.load_mcp_config import mcp_config_sources, read_mcp_file
    from dostuff.lib.memory.chroma import MEMORY_INSTALL_HINT, chroma_available, embedding_model_missing
    from dostuff.setup_wizard import missing_api_keys

    cfg = get_config()
    checks = [Check(f"Global config: {cfg.global_path}", cfg.global_path.exists(), "run `dostuff setup`")]
    proj = Path.cwd() / ".dostuff" / "config.yaml"
    checks.append(Check(f"Project config: {proj}" + ("" if proj.exists() else " (none; `dostuff init` creates one)"), None))
    for env_path in (cfg.global_path.parent / ".env", Path.cwd() / ".dostuff" / ".env"):
        checks.append(Check(f".env: {env_path}" + ("" if env_path.exists() else " (none)"), None))
    model = cfg.model_name
    missing = missing_api_keys(model)
    checks.append(Check(
        f"Model: {model}" + (f" (missing {', '.join(missing)})" if missing else ""), not missing,
        f"run `dostuff setup`, or add {missing[0]}=... to {cfg.global_path.parent / '.env'}" if missing else "",
    ))
    servers: dict = {}
    for source in mcp_config_sources():
        try:
            servers.update(read_mcp_file(source))
        except Exception as e:
            checks.append(Check(f"MCP config could not be read: {source}: {e}", False, "fix the JSON in that file"))
    checks.append(Check(f"MCP servers: {len(servers)} (`dostuff mcp list`)" if servers
                        else "MCP servers: none configured (optional; `dostuff mcp add`)", None))
    checks.append(Check(f"Data dir: {cfg.data_dir}" + ("" if cfg.data_dir.exists() else " (created on first run)"), None))
    if not chroma_available():
        checks.append(Check(f"Long-term memory: off, chromadb not installed (optional; `{MEMORY_INSTALL_HINT}`)", None))
    else:
        checks.append(Check("Memory embedding model: " + ("not downloaded yet (~80 MB; the TUI fetches it on first start)"
                                                          if embedding_model_missing() else "downloaded"), None))
    checks.append(Check(f"Permission mode: {cfg.permission_mode}", None))
    return checks
