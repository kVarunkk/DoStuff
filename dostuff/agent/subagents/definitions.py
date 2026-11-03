"""Agent types: built-in ones, and Markdown files with YAML frontmatter (spec D2, D3)."""
import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dostuff.agent.subagents import RESTRICTED_SUBAGENT_TOOLS


@dataclass(frozen=True)
class AgentDef:
    name: str
    description: str
    prompt: str = ""  # the type's system prompt (a file's body)
    tools: tuple[str, ...] | None = None  # None: every tool a subagent of this type may have
    model: str = "inherit"
    max_iterations: int | None = None  # None: the configured subagent cap
    source: str = "built-in"  # a file path, or "built-in"
    read_only_shell: bool = False  # bash_command limited to permissions.is_read_only_shell (D3)


# explore / plan: look, don't touch. The shell is limited to read-only commands (read_only_shell).
READ_ONLY_TOOLS = (
    "read_file", "list_files", "grep", "glob", "bash_command", "get_current_datetime", "web_fetch", "web_search",
    "search_mcp_tools", "get_mcp_tool_details", "call_mcp_tool",
)

BUILTIN_TYPES: dict[str, AgentDef] = {
    "general-purpose": AgentDef(
        name="general-purpose",
        description="Research and multi-step tasks, including changing files. The default.",
        prompt=(
            "Research and multi-step tasks: investigate, make the changes the task asks for, and check your "
            "work. Report what you did and what you found."
        ),
    ),
    "explore": AgentDef(
        name="explore",
        description="Read-only: finds code fast and answers where things are and how they work.",
        prompt=(
            "You locate code quickly. Search broadly first (glob, grep), then read only the parts that matter. "
            "You are read-only: don't try to change files, and the shell runs only read-only commands. Answer with "
            "file:line references and conclusions, not file dumps."
        ),
        tools=READ_ONLY_TOOLS,
        read_only_shell=True,
    ),
    "plan": AgentDef(
        name="plan",
        description="Read-only: investigates and returns a step-by-step implementation plan.",
        prompt=(
            "You design implementation plans. Investigate the code the task touches, then return a step-by-step "
            "plan: the critical files (with paths), the changes in order, risks and trade-offs, and how to test it. "
            "You are read-only: don't try to change files, and the shell runs only read-only commands."
        ),
        tools=READ_ONLY_TOOLS,
        read_only_shell=True,
    ),
}
RESERVED_TYPES = frozenset({"fork"})  # built into the harness (D11); a file can't define it

# Claude Code tool names in an agent file's `tools:` → DoStuff tools (D2). DoStuff's own names pass through.
CLAUDE_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "Read": ("read_file",),
    "Write": ("write_file",),
    "Edit": ("edit_file",),
    "MultiEdit": ("edit_file",),
    "Glob": ("glob",),
    "Grep": ("grep",),
    "LS": ("list_files",),
    "Bash": ("bash_command",),
    "WebFetch": ("web_fetch",),
    "WebSearch": ("web_search",),
    # Known so an agent file that lists it gets the restricted-tool warning, not "unknown" (ask-user D10)
    "AskUserQuestion": ("ask_user",),
}
MCP_TOOLS = ("search_mcp_tools", "get_mcp_tool_details", "call_mcp_tool")

_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)^---[ \t]*(?:\r?\n|\Z)(.*)\Z", re.DOTALL | re.MULTILINE)


@dataclass
class Definitions:
    types: dict[str, AgentDef] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class UnknownAgentType(ValueError):
    def __init__(self, name: str, available: list[str]):
        super().__init__(f"Unknown subagent_type '{name}'. Available types: {', '.join(available)}.")
        self.available = available


def agent_folders(project: Path, home: Path, read_claude: bool = True) -> list[Path]:
    """Where agent files live, first match wins (D2)."""
    folders = [project / ".dostuff" / "agents"]
    if read_claude:
        folders.append(project / ".claude" / "agents")
    folders.append(home / ".dostuff" / "agents")
    if read_claude:
        folders.append(home / ".claude" / "agents")
    return folders


def map_tools(raw: Any, source: str) -> tuple[tuple[str, ...], list[str]]:
    """`tools:` (comma-separated string or YAML list) → (DoStuff tool names, warnings). Unknown and
    restricted names are dropped with one warning each."""
    from dostuff.tools.names import BUILTIN_TOOL_NAMES  # not TOOL_MAP: that imports litellm
    names = [str(n).strip() for n in raw] if isinstance(raw, list) else str(raw).split(",")
    mapped: list[str] = []
    warnings: list[str] = []
    for name in (n.strip() for n in names):
        if not name:
            continue
        if name.startswith("mcp__"):
            targets: tuple[str, ...] = MCP_TOOLS
        elif name in CLAUDE_TOOL_NAMES:
            targets = CLAUDE_TOOL_NAMES[name]
        elif name in BUILTIN_TOOL_NAMES:
            targets = (name,)
        else:
            warnings.append(f"{source}: tool '{name}' dropped (unknown tool)")
            continue
        # Checked after the alias lookup as well as before it: AskUserQuestion → ask_user must not slip
        # past the restriction by arriving under Claude Code's name (ask-user D10/D11)
        if any(t in RESTRICTED_SUBAGENT_TOOLS for t in targets):
            warnings.append(f"{source}: tool '{name}' dropped (not available to subagents)")
            continue
        mapped.extend(t for t in targets if t not in mapped)
    return tuple(mapped), warnings


def parse_agent_file(path: Path) -> tuple[AgentDef | None, list[str]]:
    """One agent file → (definition, warnings). A broken file gives (None, [reason]); never raises."""
    def skip(reason: str) -> tuple[None, list[str]]:
        return None, [f"{path}: skipped ({reason})"]
    try:
        text = path.read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
    except UnicodeDecodeError:
        return skip("not valid UTF-8")
    except OSError as e:
        return skip(f"could not read: {e}")
    match = _FRONTMATTER.match(text)
    if not match:
        return skip("no YAML frontmatter between --- lines")
    try:
        front = yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        return skip(f"bad YAML: {str(e).splitlines()[0]}")
    if not isinstance(front, dict):
        return skip("frontmatter is not a mapping")
    name = str(front.get("name") or "").strip()
    description = str(front.get("description") or "").strip()
    if not name:
        return skip("no name")
    if not description:
        return skip("no description")
    warnings: list[str] = []
    tools: tuple[str, ...] | None = None
    if front.get("tools") not in (None, ""):
        tools, warnings = map_tools(front["tools"], str(path))
    max_iterations: int | None = None
    if front.get("max_iterations") is not None:
        try:
            max_iterations = max(int(front["max_iterations"]), 1)
        except (TypeError, ValueError):
            warnings.append(f"{path}: max_iterations '{front['max_iterations']}' is not a number; ignored")
    agent_def = AgentDef(
        name=name, description=description, prompt=match.group(2).strip(), tools=tools,
        model=str(front.get("model") or "inherit").strip(), max_iterations=max_iterations, source=str(path),
    )
    return agent_def, warnings


def _agent_files(folder: Path) -> list[Path]:
    try:
        return sorted(p for p in folder.glob("*.md") if p.is_file())
    except OSError:
        return []


def _load_files(project: Path, home: Path, read_claude: bool) -> Definitions:
    """Agent files in search order, first `name` wins; no built-in types. Blocking."""
    loaded = Definitions()
    for folder in agent_folders(project, home, read_claude):
        for path in _agent_files(folder):
            agent_def, warnings = parse_agent_file(path)
            loaded.warnings.extend(warnings)
            if agent_def is not None and agent_def.name in RESERVED_TYPES:
                loaded.warnings.append(f"{path}: skipped ('{agent_def.name}' is a reserved type name)")
            elif agent_def is not None and agent_def.name not in loaded.types:
                loaded.types[agent_def.name] = agent_def
    return loaded


def _with_builtins(files: Definitions) -> Definitions:
    """File types plus the built-in ones they don't override."""
    types = dict(files.types)
    for name, agent_def in BUILTIN_TYPES.items():
        types.setdefault(name, agent_def)
    return Definitions(types, list(files.warnings))


def load_definitions(project: Path, home: Path, read_claude: bool = True) -> Definitions:
    """Every agent type: files in search order (first `name` wins), then the built-in types. Blocking;
    call it through asyncio.to_thread on the event loop (CON-004)."""
    return _with_builtins(_load_files(project, home, read_claude))


# --- the session's definitions, reloaded when an agent file changes (D2) -----------------------------

_Where = tuple[Path, Path, bool]
_Signature = tuple[tuple[str, int, int], ...]


@dataclass
class _Cache:
    where: _Where
    signature: _Signature
    files: Definitions
    builtins: dict[str, AgentDef]  # BUILTIN_TYPES when `merged` was built
    merged: Definitions


_cache: _Cache | None = None


def _where() -> _Where:
    from dostuff.config import get_config
    from dostuff.helpers.tools.project_root import project_root
    return project_root(), Path.home(), bool(getattr(get_config(), "agents_read_claude_agents", True))


def _signature(folders: list[Path]) -> _Signature:
    sig: list[tuple[str, int, int]] = []
    for folder in folders:
        for path in _agent_files(folder):
            try:
                st = path.stat()
            except OSError:
                continue
            sig.append((str(path), st.st_mtime_ns, st.st_size))
    return tuple(sig)


def _merged(cache: _Cache) -> Definitions:
    if cache.builtins != BUILTIN_TYPES:  # a type registered at run time (tests)
        cache.builtins, cache.merged = dict(BUILTIN_TYPES), _with_builtins(cache.files)
    return cache.merged


def current_definitions() -> Definitions:
    """The session's agent types and load warnings, re-read only when a file in an agent folder was added,
    removed or changed (path, mtime, size). Blocking: on the event loop use refresh_definitions()."""
    global _cache
    where = _where()
    sig = _signature(agent_folders(*where))
    if _cache is None or _cache.where != where or _cache.signature != sig:
        files = _load_files(*where)
        _cache = _Cache(where, sig, files, dict(BUILTIN_TYPES), _with_builtins(files))
    return _merged(_cache)


def reload() -> Definitions:
    """Re-read every agent file (`/agents reload`)."""
    global _cache
    _cache = None
    return current_definitions()


async def refresh_definitions() -> Definitions:
    """current_definitions() in a worker thread (CON-004); called before a run starts."""
    return await asyncio.to_thread(current_definitions)


def get_definitions() -> dict[str, AgentDef]:
    """Every agent type this session can start, by name, as of the last check (loaded now if never)."""
    return (_merged(_cache) if _cache is not None else current_definitions()).types


# --- models (D15) -----------------------------------------------------------------------------------

CLAUDE_MODEL_ALIASES = frozenset({"sonnet", "opus", "haiku"})  # Claude Code's; mapped through agents.models


def resolve_model(call_model: str, agent_def: AgentDef, config: Any) -> tuple[str, str | None]:
    """The litellm model a run uses, and a warning when an alias couldn't be resolved.

    The call's `model=` wins, then the type's `model:` (the built-in explore type: `agents.explore_model`), then
    the main model. Each is `inherit`, an alias from `agents.models` (Claude Code's sonnet/opus/haiku are aliases
    too), or a litellm string, passed on as is.
    """
    main = config.model_name
    wanted = (call_model or "").strip()
    if not wanted:
        wanted = agent_def.model.strip() or "inherit"
        if wanted == "inherit" and agent_def.name == "explore":
            wanted = str(getattr(config, "agents_explore_model", "") or "inherit").strip()
    aliases: dict[str, str] = dict(getattr(config, "agents_models", {}) or {})
    seen: set[str] = set()
    while wanted in aliases and wanted not in seen:  # an alias may point at another alias
        seen.add(wanted)
        wanted = aliases[wanted].strip() or "inherit"
    if wanted == "inherit":
        return main, None
    if wanted in CLAUDE_MODEL_ALIASES and wanted not in seen:
        return main, f"`{wanted}` isn't in agents.models; using the main model ({main})"
    return wanted, None


def types_listing(loaded: Definitions, config: Any) -> str:
    """`/agents types`: each type with its source, resolved model and tools, then the warnings."""
    from dostuff.agent.subagents.run import resolve_subagent_tools
    lines: list[str] = []
    warnings = list(loaded.warnings)
    for name in sorted(loaded.types):
        agent_def = loaded.types[name]
        model, model_warning = resolve_model("", agent_def, config)
        if model_warning:
            warnings.append(f"{agent_def.source}: {model_warning}")
        if agent_def.tools is None:
            tools = "every subagent tool"
        else:
            tools = ", ".join(resolve_subagent_tools(list(agent_def.tools))[0]) or "(none)"
        if agent_def.read_only_shell:
            tools += " (read-only shell)"
        lines += [f"{name}  ·  {agent_def.source}", f"  {agent_def.description}",
                  f"  model: {model}", f"  tools: {tools}"]
    lines += _fork_summary(config)
    lines += ["", "/agents types <name> shows a type's system prompt and full config."]
    if warnings:
        lines += ["", "Warnings:"] + [f"  ⚠ {w}" for w in warnings]
    return "\n".join(lines)


def _fork_summary(config: Any) -> list[str]:
    from dostuff.agent.subagents.fork import FORK_DESCRIPTION, FORK_TYPE
    return [f"{FORK_TYPE}  ·  built-in", f"  {FORK_DESCRIPTION}", f"  model: {config.model_name} (always the main model)",
            "  tools: the main agent's tools, minus agent, send_message, stop_agent and the other restricted ones"]


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" if line.strip() else "" for line in text.splitlines())


def type_detail(name: str, loaded: Definitions, config: Any) -> str | None:
    """`/agents types <name>`: one type's whole config, its system prompt included. None for an unknown name."""
    from dostuff.agent.subagents.fork import FORK_DIRECTIVE, FORK_TYPE
    from dostuff.agent.subagents.run import resolve_subagent_tools
    cap = f"{config.agents_max_iterations} (agents.max_iterations)"
    if name == FORK_TYPE:
        return "\n".join(_fork_summary(config) + [
            f"  max iterations: {cap}",
            "", "System prompt: the main agent's, unchanged (a fork inherits the whole conversation).",
            "", "Task message it gets after the copied conversation:", "",
            _indent(FORK_DIRECTIVE.replace("{prompt}", "<the agent tool's prompt>"))])
    agent_def = loaded.types.get(name)
    if agent_def is None:
        return None
    model, model_warning = resolve_model("", agent_def, config)
    if agent_def.tools is None:
        tools = "every subagent tool"
    else:
        tools = ", ".join(resolve_subagent_tools(list(agent_def.tools))[0]) or "(none)"
    if agent_def.read_only_shell:
        tools += " (read-only shell)"
    model_line = f"  model: {model}"
    if agent_def.model.strip() not in ("", "inherit", model):
        model_line += f" (from `{agent_def.model.strip()}`)"
    iterations = f"{agent_def.max_iterations} (set in the file)" if agent_def.max_iterations is not None else cap
    lines = [f"{agent_def.name}  ·  {agent_def.source}", f"  {agent_def.description}", model_line,
             f"  tools: {tools}", f"  max iterations: {iterations}"]
    if model_warning:
        lines.append(f"  ⚠ {model_warning}")
    lines += ["", "System prompt:", "", _indent(agent_def.prompt) if agent_def.prompt.strip() else "  (empty)"]
    return "\n".join(lines)
