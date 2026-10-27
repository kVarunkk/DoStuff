"""Settings tools: read and change dostuff's own config.yaml and MCP servers on the user's behalf.

Writes always ask, in every permission mode (kind "settings"): config.yaml holds the permission
rules and an MCP server is a command that runs on the user's machine. `.env` files (API keys) are
never read or written here. Each change is applied to the running app through the adapter.
"""
import copy
import json
from pathlib import Path
from typing import Any, Literal

import yaml

from dostuff.config import STATUS_BAR_ITEMS, get_config
from dostuff.helpers.mcp.load_mcp_config import (global_mcp_config_path, project_mcp_config_path, read_mcp_file,
                                                 write_mcp_file)
from dostuff.helpers.mcp.server_config import build_server_config
from dostuff.helpers.tools.diff import diff_stats, plain_diff, report_change, unified_diff

Scope = Literal["user", "project"]
REDACTED = "<redacted>"
# Top-level keys config.yaml understands (see templates/config.example.yaml)
CONFIG_KEYS = ("tracing", "models", "model", "loop", "memory", "permissions", "mcp", "ui", "search")
_CRLF, _LF = "\r\n", "\n"


def config_path(scope: Scope = "user") -> Path:
    if scope == "project":
        return Path.cwd() / ".dostuff" / "config.yaml"
    return Path.home() / ".dostuff" / "config.yaml"


def mcp_path(scope: Scope = "user") -> Path:
    """The MCP file for `scope`; `mcp.config_path` in config.yaml replaces both when it's set."""
    block = get_config().raw.get("mcp")
    explicit = block.get("config_path") if isinstance(block, dict) else None
    if explicit:
        return Path(str(explicit)).expanduser()
    return project_mcp_config_path() if scope == "project" else global_mcp_config_path()


def _check_scope(scope: str) -> Scope:
    if scope not in ("user", "project"):
        raise ValueError("scope must be 'user' (~/.dostuff) or 'project' (./.dostuff)")
    return scope  # type: ignore[return-value]


def redact_servers(servers: dict[str, Any]) -> dict[str, Any]:
    """Servers with every env / header value hidden: they are usually tokens."""
    out = copy.deepcopy(servers)
    for cfg in out.values():
        if isinstance(cfg, dict):
            for key in ("env", "headers"):
                if isinstance(cfg.get(key), dict):
                    cfg[key] = {k: REDACTED for k in cfg[key]}
    return out


def _display(path: Path) -> str:
    return str(path).replace(str(Path.home()), "~")


def _has_crlf(path: Path) -> bool:
    """Whether the file uses Windows line endings (read_text turns them into plain newlines)."""
    return path.is_file() and _CRLF.encode() in path.read_bytes()


def _apply(adapter: Any, what: str) -> None:
    """Re-read the config and tell the UI, so the change shows without a restart."""
    get_config().reload()
    if adapter is not None and hasattr(adapter, "emit"):
        adapter.emit("settings_changed", what)


def read_settings() -> str:
    """Shows dostuff's own settings: config.yaml (user and project), the MCP servers, and a
    reference of every config.yaml option. Read this before edit_config, set_mcp_server or
    remove_mcp_server. MCP env and header values are shown as <redacted>.

    Returns:
        The settings files with their paths, then the option reference.
    """
    from dostuff.setup_wizard import read_template
    parts = []
    for scope, label, path in (("user", "User config", config_path("user")),
                               ("project", "Project config (overrides the user config, key by key)", config_path("project"))):
        text = path.read_text(encoding="utf-8") if path.is_file() else "(not created yet)"
        parts.append(f"## {label} — scope '{scope}': {_display(path)}\n{text.rstrip()}")
    mcp_files = [("user", mcp_path("user")), ("project", mcp_path("project"))]
    if mcp_files[0][1] == mcp_files[1][1]:
        mcp_files = mcp_files[:1]  # mcp.config_path: one file for both scopes
    for scope, path in mcp_files:
        try:
            body = json.dumps(redact_servers(read_mcp_file(path)), indent=2) if path.is_file() else "(not created yet)"
        except Exception as e:
            body = f"(unreadable: {e})"
        parts.append(f"## MCP servers — scope '{scope}': {_display(path)}\n{body}")
    parts.append("## Reference: every config.yaml option (the shipped template)\n" + read_template("config.example.yaml").rstrip())
    parts.append("API keys live in ~/.dostuff/.env or ./.dostuff/.env; these tools don't read or change them. "
                 f"ui.status_bar items: {', '.join(STATUS_BAR_ITEMS)}.")
    return "\n\n".join(parts)


def _config_warnings(data: dict[str, Any]) -> list[str]:
    warnings = [f"unknown top-level key '{k}'" for k in data if k not in CONFIG_KEYS]
    ui = data.get("ui")
    items = ui.get("status_bar") if isinstance(ui, dict) else None
    if isinstance(items, list):
        warnings += [f"unknown ui.status_bar item '{i}'" for i in items if str(i).strip().lower() not in (*STATUS_BAR_ITEMS, "timer")]
    perms = data.get("permissions")
    mode = perms.get("mode") if isinstance(perms, dict) else None
    if mode is not None and str(mode).lower() not in ("ask", "allowlist", "auto", "plan"):
        warnings.append(f"permissions.mode '{mode}' isn't ask, allowlist, auto or plan (it falls back to ask)")
    return warnings


def edit_config(old_str: str, new_str: str, scope: Scope = "user", adapter: Any = None) -> str:
    """Changes dostuff's config.yaml (models, loop limits, memory, permissions, status bar, tracing)
    by replacing exact text, keeping the file's comments. Call read_settings first. The change
    applies right away; tracing changes need a restart. The user is always asked first.

    Args:
        old_str: Exact text to replace; must occur once. Empty appends new_str to the end of the
            file (e.g. a new top-level section), creating the file if needed.
        new_str: Replacement text. The result must be valid YAML.
        scope: 'user' for ~/.dostuff/config.yaml (all projects), 'project' for ./.dostuff/config.yaml
            (this project only, overrides the user file).

    Returns:
        The diff, plus any warnings about the new config.
    """
    path, content, updated, data = _edited_config(old_str, new_str, scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    crlf = _has_crlf(path)
    path.write_text(updated.replace(_LF, _CRLF) if crlf else updated, encoding="utf-8", newline="")
    result = report_change("Settings changed", _display(path), content, updated)
    _apply(adapter, "config")
    warnings = _config_warnings(data)
    return result + ("\nWarnings: " + "; ".join(warnings) if warnings else "")


def _edited_config(old_str: str, new_str: str, scope: str) -> tuple[Path, str, str, dict[str, Any]]:
    """(path, current text, edited text, edited data), LF line endings; raises if the edit can't apply."""
    path = config_path(_check_scope(scope))
    raw = path.read_text(encoding="utf-8") if path.is_file() else ""
    content, old_str, new_str = (s.replace(_CRLF, _LF) for s in (raw, old_str, new_str))
    if old_str == new_str:
        raise ValueError("old_str and new_str are identical; nothing to change.")
    if not old_str:
        updated = content.rstrip(_LF) + (_LF * 2 if content.strip() else "") + new_str.strip(_LF) + _LF
    else:
        count = content.count(old_str)
        if count != 1:
            raise ValueError(f"old_str occurs {count} times in {_display(path)}; it must occur once. "
                             "Call read_settings and copy the text exactly.")
        updated = content.replace(old_str, new_str, 1)
    try:
        data = yaml.safe_load(updated)
    except yaml.YAMLError as e:
        raise ValueError(f"The edit would make {_display(path)} invalid YAML, so nothing was written: {e}") from e
    if data is not None and not isinstance(data, dict):
        raise ValueError("config.yaml must be a mapping of sections; nothing was written.")
    return path, content, updated, data or {}


# ── /config: the user edits a whole settings file in the TUI ──────────────────

SettingsKind = Literal["config", "mcp"]


def settings_file(kind: SettingsKind, scope: Scope) -> Path:
    return config_path(scope) if kind == "config" else mcp_path(scope)


def new_settings_text(kind: SettingsKind, scope: Scope) -> str:
    """What the editor starts with for a file that doesn't exist yet."""
    if kind == "mcp":
        return '{\n  "mcpServers": {}\n}\n'
    from dostuff.setup_wizard import read_template
    where = "this project only; overrides the user config key by key" if scope == "project" else "all projects"
    reference = "\n".join(f"# {line}".rstrip() for line in read_template("config.example.yaml").splitlines())
    return f"# dostuff config ({where}). Every option, commented out:\n#\n{reference}\n"


def check_settings_text(kind: SettingsKind, text: str) -> list[str]:
    """Warnings for an edited settings file; raises ValueError when it can't be saved."""
    if kind == "config":
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}") from e
        if data is not None and not isinstance(data, dict):
            raise ValueError("config.yaml must be a mapping of sections.")
        return _config_warnings(data or {})
    try:
        data = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError('The MCP config must be a JSON object like {"mcpServers": {...}}.')
    servers = data.get("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError('"mcpServers" must be an object of name → server.')
    warnings = []
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"MCP server '{name}' must be an object.")
        if "__" in name:
            warnings.append(f"server name '{name}' contains '__', which breaks its tool names")
        if not cfg.get("command") and not cfg.get("url"):
            warnings.append(f"server '{name}' has neither a command nor a url, so it can't connect")
    return warnings


def save_settings_file(kind: SettingsKind, scope: Scope, text: str, adapter: Any = None) -> str:
    """Validate and write an edited settings file, then apply it. Returns a one-line summary
    (no diff: an MCP file usually holds tokens). Raises ValueError and writes nothing if invalid."""
    warnings = check_settings_text(kind, text)
    path = settings_file(kind, scope)
    old = path.read_text(encoding="utf-8") if path.is_file() else ""
    text = text.replace(_CRLF, _LF)
    if not text.endswith(_LF):
        text += _LF
    path.parent.mkdir(parents=True, exist_ok=True)
    crlf = _has_crlf(path)
    path.write_text(text.replace(_LF, _CRLF) if crlf else text, encoding="utf-8", newline="")
    added, removed = diff_stats(unified_diff(old, text, _display(path)))
    _apply(adapter, kind)
    summary = f"Saved {_display(path)} (+{added} / -{removed} lines)"
    return summary + ("\nWarnings: " + "; ".join(warnings) if warnings else "")


SETTINGS_TOOLS = frozenset({"edit_config", "set_mcp_server", "remove_mcp_server"})


def settings_preview(tool: str, args: dict[str, Any]) -> str:
    """What the approval prompt shows: the config diff, or the MCP entry (secrets redacted)."""
    try:
        if tool == "edit_config":
            path, content, updated, _ = _edited_config(str(args.get("old_str", "")), str(args.get("new_str", "")),
                                                       str(args.get("scope") or "user"))
            return plain_diff(unified_diff(content, updated, _display(path)))
        path = mcp_path(_check_scope(str(args.get("scope") or "user")))
        name = str(args.get("name", ""))
        if tool == "remove_mcp_server":
            entry = read_mcp_file(path).get(name)
            return json.dumps(redact_servers({name: entry}), indent=2) if entry else "  (no such server; it will fail)"
        cfg = build_server_config(command=args.get("command"), url=args.get("url"), transport=args.get("transport"),
                                  env=args.get("env"), headers=args.get("headers"))
        if cfg.get("command"):
            cfg = {**cfg, "runs": " ".join([cfg["command"], *map(str, cfg["args"])])}
        return f"  in {_display(path)}\n" + json.dumps(redact_servers({name: cfg}), indent=2)
    except Exception as e:
        return f"  ({e}; it will fail)"


def set_mcp_server(
    name: str,
    command: list[str] | None = None,
    url: str | None = None,
    transport: str | None = None,
    env: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    scope: Scope = "user",
    adapter: Any = None,
) -> str:
    """Adds an MCP server, or replaces the one with this name, and connects it right away.
    Give either `command` (a local stdio server) or `url` (an HTTP server). Take the package or
    URL from the server's official docs; never use a placeholder. Most hosted servers sign in with
    OAuth in the browser when they connect, so give only `url` and no `headers` unless the docs
    require an API key. The user is always asked first. To keep an existing secret, pass
    "<redacted>" as its value.

    Args:
        name: Server name; its tools are called <name>__<tool>.
        command: stdio: the program and its arguments, e.g. ["uvx", "mcp-server-git"].
        url: HTTP server URL, e.g. "https://mcp.example.com/mcp".
        transport: stdio | http | sse. Default: stdio for a command, http for a url.
        env: Environment variables for a stdio server, e.g. {"GITHUB_TOKEN": "..."}.
        headers: HTTP headers, only when the server's docs require a key, e.g.
            {"Authorization": "Bearer ..."}. Leave out for OAuth servers.
        scope: 'user' for ~/.dostuff/mcp_config.json (all projects), 'project' for
            ./.dostuff/mcp_config.json (this project only).

    Returns:
        What changed, with secrets redacted.
    """
    name = name.strip()
    if not name or "__" in name:
        raise ValueError("name must be non-empty and must not contain '__'")
    path = mcp_path(_check_scope(scope))
    servers = read_mcp_file(path)
    found = servers.get(name)
    old: dict[str, Any] = found if isinstance(found, dict) else {}

    def unredacted(key: str, given: dict[str, str] | None) -> dict[str, str] | None:
        """"<redacted>" (as read_settings shows it) keeps the stored value."""
        if not given:
            return given
        found_values = old.get(key)
        stored: dict[str, str] = found_values if isinstance(found_values, dict) else {}
        out = {}
        for k, v in given.items():
            if v == REDACTED and k not in stored:
                raise ValueError(f"{key} '{k}' is <redacted> but no value is stored for it; ask the user for it")
            out[k] = stored[k] if v == REDACTED else v
        return out

    cfg = build_server_config(command=command, url=url, transport=transport,
                              env=unredacted("env", env), headers=unredacted("headers", headers))
    before = json.dumps(redact_servers({name: old}) if old else {}, indent=2)
    servers[name] = cfg
    write_mcp_file(path, servers)
    _apply(adapter, "mcp")
    verb = "Updated" if old else "Added"
    return report_change(f"{verb} MCP server '{name}' (connecting now)", _display(path), before,
                         json.dumps(redact_servers({name: cfg}), indent=2))


def remove_mcp_server(name: str, scope: Scope = "user", adapter: Any = None) -> str:
    """Removes an MCP server and disconnects it. The user is always asked first.

    Args:
        name: Server name, as read_settings shows it.
        scope: 'user' (~/.dostuff/mcp_config.json) or 'project' (./.dostuff/mcp_config.json).

    Returns:
        A confirmation.
    """
    path = mcp_path(_check_scope(scope))
    servers = read_mcp_file(path)
    if name not in servers:
        raise ValueError(f"No server '{name}' in {_display(path)}. read_settings shows which file each one is in.")
    del servers[name]
    write_mcp_file(path, servers)
    _apply(adapter, "mcp")
    return f"Removed MCP server '{name}' from {_display(path)}."
