"""`dostuff doctor` and /doctor: setup checks, each with a fix for a problem."""
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dostuff.helpers.hooks.load_hooks import HookEntry

# First tokens a hook command may start with that are shell builtins/keywords, not programs on PATH
SHELL_BUILTINS = {"cd", "echo", "exit", "true", "false", "set", "export", "test", "[", ":", ".", "source",
                  "pushd", "popd", "type", "if", "for", "while", "write-output", "write-host", "set-location"}
SCRIPT_SUFFIXES = {".py", ".sh", ".bash", ".ps1", ".js", ".mjs", ".ts", ".rb", ".pl", ".bat", ".cmd"}


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
    return checks + hooks_checks()


def _tokens(command: str) -> list[str]:
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        tokens = command.split()
    tokens = [t.strip("\"'") for t in tokens]
    return tokens[1:] if tokens[:1] == ["&"] else tokens  # PowerShell's call operator


def _first_token_missing(command: str) -> str:
    """The command's program if it can't be found, else ""."""
    tokens = _tokens(command)
    if not tokens or tokens[0].lower() in SHELL_BUILTINS:
        return ""
    program = tokens[0]
    if "/" in program or "\\" in program:
        return "" if Path(program).expanduser().exists() else program
    return "" if shutil.which(program) else program


def _scripts_outside_hooks_dir(command: str, root: Path) -> list[str]:
    hooks_dir = (root / ".dostuff" / "hooks").resolve()
    outside = []
    for token in _tokens(command)[1:]:
        if Path(token).suffix.lower() not in SCRIPT_SUFFIXES:
            continue
        path = Path(token).expanduser()
        resolved = (path if path.is_absolute() else root / path).resolve()
        if not resolved.is_relative_to(hooks_dir):
            outside.append(token)
    return outside


def hook_label(e: "HookEntry") -> str:
    """How doctor and /hooks name one hook: `PreToolUse 'block-main-push'`."""
    return f"{e.event} '{e.name}'"


def hooks_checks() -> list[Check]:
    """Hooks: load warnings (incl. broken fail_closed entries), programs missing from PATH, project trust,
    and project scripts outside .dostuff/hooks/ (not covered by trust, editable by the model)."""
    from dostuff.config import get_config
    from dostuff.helpers.hooks import trust

    cfg = get_config()
    entries = [e for group in cfg.hooks.values() for e in group]
    checks = [Check(w, False, "fix the hooks block in that config file; `/hooks` shows what loaded")
              for w in cfg.hooks_warnings]
    if not entries:
        return checks or [Check("Hooks: none configured (optional; see `hooks:` in config.example.yaml)", None)]
    for e in entries:
        missing = "" if e.broken else _first_token_missing(e.command)
        if missing:
            checks.append(Check(f"Hook {hook_label(e)}: '{missing}' not found on PATH", False,
                                f"install it or fix the command in {e.origin}"))
    project = trust.project_entries(cfg.hooks)
    if project:
        root = trust.project_root_of(project[0])
        for e in project:
            for script in _scripts_outside_hooks_dir(e.command, root):
                checks.append(Check(
                    f"Project hook {hook_label(e)} runs {script}, outside .dostuff/hooks/", False,
                    "move the script into .dostuff/hooks/: trust doesn't cover it elsewhere, "
                    "and the model can edit it without a settings prompt"))
        if cfg.hooks_project_trust == "deny":
            checks.append(Check(f"Project hooks: {len(project)} disabled (hooks.project_trust: deny)", None))
        elif cfg.hooks_project_trust == "ask" and not trust.is_trusted(trust.trust_key(project)):
            checks.append(Check(f"Project hooks: {len(project)} from {project[0].origin} not trusted yet "
                                "(the TUI asks before the first one runs)", None))
    if not any(c.ok is False for c in checks):
        user = len(entries) - len(project)
        checks.insert(0, Check(f"Hooks: {len(entries)} configured ({user} user, {len(project)} project)", True))
    return checks


HOOKS_DOCS_HINT = ("See the Hooks section of the README, and the commented `hooks:` block in "
                   "config.example.yaml (`dostuff init` writes one to .dostuff/config.yaml).")


def _hook_details(e: "HookEntry") -> str:
    parts = [f"matcher {e.matcher}"] if e.matcher else []
    if e.if_pattern:
        parts.append(f"if {e.if_pattern}")
    parts.append(f"timeout {e.timeout:g}s")
    if e.fail_closed:
        parts.append("fail_closed")
    return " · ".join(parts)


def hooks_listing() -> str:
    """/hooks: every loaded hook per event, in the order the engine runs them, with where it came from and
    whether it will run. The debugging surface for "why didn't my hook fire" (REQ-3.3)."""
    from dostuff.config import get_config
    from dostuff.helpers.hooks import trust
    from dostuff.helpers.hooks.load_hooks import EVENTS, TOOL_EVENTS
    from dostuff.helpers.tools.project_root import project_root
    from dostuff.tools.bash import bash_command

    cfg = get_config()
    entries = [e for event in EVENTS for e in cfg.hooks.get(event, [])]
    warnings = [f"  ✗ {w}" for w in cfg.hooks_warnings]
    if not entries:
        return "\n".join(["No hooks configured.", HOOKS_DOCS_HINT, *warnings])
    project_status = trust.status(cfg.hooks_project_trust, cfg.hooks)
    shell, argv = bash_command.select_shell()
    lines = [
        "Hooks run with your privileges, unsandboxed, and are not permission-checked. "
        "Keep project hook scripts in .dostuff/hooks/: trust covers them there, and the model can't edit them "
        "without a settings prompt.",
        f"Each runs as: {' '.join(argv)} \"<command>\" ({shell}), in {project_root()}",
    ]
    for event in EVENTS:
        group = cfg.hooks.get(event)
        if not group:
            continue
        lines.append(event)
        for e in group:
            mark = "broken" if e.broken else ("user" if e.source == "user" else project_status)
            problem = f" ({e.broken})" if e.broken else ""
            lines.append(f"  {mark:<9} {e.name}  {_hook_details(e)}  $ {e.command}  — {e.origin}{problem}")
    project = trust.project_entries(cfg.hooks)
    if project:
        lines.append(f"Project hooks: {project_status} (hooks.project_trust: {cfg.hooks_project_trust}"
                     + ("; the TUI asks before the first one runs)" if project_status == "untrusted" else ")"))
    worst = sum(e.timeout for event in TOOL_EVENTS for e in cfg.hooks.get(event, []))
    if worst:
        lines.append(f"Worst-case added time per tool call: {worst:g}s (every PreToolUse + PostToolUse timeout)")
    if warnings:
        lines += ["Warnings:", *warnings]
    lines.append("`/hooks r` re-reads the config files.")
    return "\n".join(lines)
