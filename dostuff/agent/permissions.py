"""Permission layer: decides whether a tool call runs, asks the user, or is denied.

Tools never approve themselves. `run_tool` strips any model-supplied approval keys and
calls `check()` before dispatch; ASK decisions are resolved through `adapter.ask()`.

An adapter's `ask()` may return a plain string ("y", "always" …) or an `ApprovalReply`,
which can carry feedback for the model. Adapters with `ask_batch()` get one prompt for
all calls of a model response that need approval.

Modes (`permissions.mode` in config, overridable per session):
  ask       — every mutating call asks, except read-only shell commands (allow-list below)
  allowlist — as `ask`, plus `permissions.allow` patterns from config are auto-approved
  auto      — nothing asks (explicit opt-in); secrets and its own rules are refused instead
  plan      — read-only: reads and read-only shell commands run, file writes / other shell
              commands / settings changes are denied with a "plan only" note; web and MCP
              calls ask as in `ask`

Settings changes (kind "settings": config.yaml, MCP servers) ask in `ask` and `allowlist` and have
no "always": they can change these rules or add a command that runs on the user's machine. Auto
runs them, because it never waits for the user and already runs any shell command. What it refuses,
without a prompt, is a secret under `.dostuff` and a change to its own rules: see `_auto_refusal`.
Reading a `.dostuff` dir from the shell is not a settings change: see `_dostuff_refs_are_reads`.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlparse

from dostuff.helpers.tools.project_root import project_root
from dostuff.helpers.tools.resolve_safe_path import is_denied_for_read, resolve_safe_path

PermissionMode = Literal["ask", "allowlist", "auto", "plan"]
ToolKind = Literal["read", "write", "shell", "mcp", "network", "settings"]
Action = Literal["allow", "ask", "deny"]

PERMISSION_MODES: tuple[str, ...] = ("ask", "allowlist", "auto", "plan")
# Order of the TUI's Shift+Tab cycle
MODE_CYCLE: tuple[str, ...] = ("ask", "auto", "plan", "allowlist")

PLAN_MODE_INSTRUCTION = (
    "## Plan mode\nPlan mode is on: you can read files, search and run read-only commands, but file "
    "changes, other shell commands and settings changes are denied. Investigate, then reply with a "
    "concrete step-by-step plan. In plan mode, ask what you need *before* writing the plan, with "
    "`ask_user` — not after. Don't make the changes until the user leaves plan mode (Shift+Tab or /permissions)."
)
PLAN_MODE_OFF_INSTRUCTION = (
    "## Plan mode\nPlan mode is off: the earlier plan-mode restrictions no longer apply. Normal permission checks "
    "do, so you can make changes now."
)


def next_mode(mode: str) -> str:
    """The mode after `mode` in the Shift+Tab cycle."""
    i = MODE_CYCLE.index(mode) if mode in MODE_CYCLE else -1
    return MODE_CYCLE[(i + 1) % len(MODE_CYCLE)]


# ── Shell read-only allow-list (design D2) ────────────────────────────────────
# Value None → any arguments; a set → the first argument must be one of these
# (after skipping `--no-pager` for git); an empty set → no arguments at all.
_ANY = None
READ_ONLY_COMMANDS: dict[str, set[str] | None] = {
    "ls": _ANY, "dir": _ANY, "pwd": _ANY, "cd": _ANY, "cat": _ANY, "type": _ANY,
    "head": _ANY, "tail": _ANY, "wc": _ANY, "echo": _ANY, "which": _ANY, "where": _ANY,
    "grep": _ANY, "rg": _ANY, "find": _ANY, "tree": _ANY, "stat": _ANY, "file": _ANY,
    "date": {"+", "-u", "--utc", "-R", "-I", "--iso-8601"}, "whoami": _ANY,
    "git": {"status", "diff", "log", "show", "branch", "rev-parse", "remote"},
    "python": {"--version", "-V"}, "python3": {"--version", "-V"}, "py": {"--version", "-V"},
    "pip": {"list", "show", "freeze", "--version"}, "pip3": {"list", "show", "freeze", "--version"},
    "node": {"--version", "-v"}, "npm": {"ls", "list", "--version", "-v"},
    "get-childitem": _ANY, "get-content": _ANY, "get-location": _ANY, "select-string": _ANY,
}
# `git branch` / `git remote` mutate with most flags; only these listing forms are read-only.
_GIT_SAFE_ARGS: dict[str, set[str]] = {
    "branch": {"", "-a", "-r", "-v", "-vv", "--list", "--show-current", "--all", "--remotes"},
    "remote": {"", "-v", "--verbose"},
}
# `env` / `printenv` are intentionally absent: they dump secrets into the transcript.

_FORBIDDEN_SUBSTRINGS = (
    ">", "<", "`", "$(", "${", "out-file", "set-content", "add-content", "env:", ".env",
)
# A `.dostuff` path (`.dostuff/config.yaml`, `~/.dostuff`, cwd=".dostuff"), not a dotted attribute such
# as `cli_tui.DostuffTUI` — that is ordinary code, and treating it as one denied reads in plan mode.
_DOSTUFF_REF = re.compile(r"(?<![\w-])\.dostuff(?![\w-])", re.IGNORECASE)
# Flags that make otherwise read-only commands write, delete or execute (find -exec, rg --pre, git --output …).
_FORBIDDEN_FLAGS = re.compile(
    r"(?<![\w-])(-delete|-exec\w*|-ok\w*|-fprint\w*|-fls|--output|--pre)(?![\w-])",
    re.IGNORECASE,
)
_ENV_EXPANSION = re.compile(r"\$[A-Za-z_]")
# Redirections that create no file. Anything after them (`2>&1x` writes a file named `1x`) isn't one.
_HARMLESS_REDIRECT = re.compile(r"(?<![\w>&])2>(?:&1|/dev/null)(?=[\s;&|]|$)")
# These read every file below a directory, so on all of `.dostuff` they reach `.dostuff/.env`
_SEARCH_COMMANDS = {"grep", "rg", "select-string"}
_GLOB_CHARS = set("*?[{")  # `cat .dostuff/.e?v` reads what the literal `.env` check is there to stop
# These print names, not contents, so a wildcard on them can't put a key in the transcript
_LISTING_COMMANDS = {"ls", "dir", "tree", "stat", "file", "find", "get-childitem"}
_DIR_CHANGERS = {"cd", "chdir", "pushd", "popd", "set-location", "sl", "push-location", "pop-location"}
# For a command that can't be split into arguments (a redirect, a substitution); backslashes already made `/`
_SECRET_PATH = re.compile(r"(?<![\w-])\.dostuff/(?:data/|(?:[^\s\"'`]*/)?(?:\.env|mcp_config\.json))", re.IGNORECASE)
_SECRET_FILES = (".env", "mcp_config.json")
# What sets the agent's own limits. A settings-kind command already names `.dostuff`, so the bare name is enough
_RULES_FILE = re.compile(r"(?<![\w-])(?:config\.yaml|trusted_hooks\.json)", re.IGNORECASE)
_SECRET_REFUSAL = (
    "`.env` files, MCP configs and dostuff's data dir hold keys and private history, so they stay closed. Use "
    "read_settings or set_mcp_server for MCP servers, or tell the user what needs changing."
)
RULES_REFUSAL = (
    "Auto mode doesn't change its own limits: the permission rules, hook trust and trusted_hooks.json are the "
    "user's to edit. Tell the user what needs changing."
)
_CONFIG_REFUSAL = (
    "config.yaml sets the agent's own limits, so auto mode only changes it through edit_config, which checks the "
    "edit. Read it with read_file or read_settings."
)
# Git worktrees live in .dostuff/worktrees/: reviewing one is not touching dostuff's settings
_WORKTREES_PREFIX = re.compile(r"\.dostuff[/\\]worktrees[/\\]")
_SEPARATORS = {"&&", "||", ";", "|", "&", "|&", ";;"}


def split_shell_segments(command: str) -> list[list[str]] | None:
    """Tokenize `command` and split it on `&&`, `||`, `;`, `|`, `&` and newlines.

    Returns None when the command cannot be parsed safely (unbalanced quotes,
    subshell parentheses …) — callers treat that as "ask".
    """
    normalized = command.replace("\r\n", "\n").replace("\n", " ; ")
    lexer = shlex.shlex(normalized, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return None
    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _SEPARATORS:
            segments.append([])
        elif tok and set(tok) <= set("();<>|&"):
            return None
        else:
            segments[-1].append(tok)
    return [s for s in segments if s]


def _segment_is_read_only(tokens: list[str]) -> bool:
    cmd = tokens[0].lower()
    if cmd.endswith(".exe"):
        cmd = cmd[:-4]
    if cmd not in READ_ONLY_COMMANDS:
        return False
    if cmd == "tree":  # `tree -o <file>` writes it, and short flags bundle (`-ao`)
        return not any(a.startswith("-") and not a.startswith("--") and "o" in a for a in tokens[1:])
    allowed = READ_ONLY_COMMANDS[cmd]
    if allowed is None:
        return True
    args = tokens[1:]
    if cmd == "git":
        while args and args[0] == "--no-pager":
            args = args[1:]
        if not args or args[0] not in allowed:
            return False
        sub = args[0]
        if sub in _GIT_SAFE_ARGS:
            return all(a in _GIT_SAFE_ARGS[sub] for a in (args[1:] or [""]))
        return True
    if cmd == "date":  # `date -s` sets the clock; allow only formatting flags
        return all(a.startswith("+") or a in allowed for a in args)
    return bool(args) and args[0] in allowed


def _has_unsafe_syntax(command: str) -> bool:
    """Redirection, substitution, env expansion, or destructive flags anywhere in the string."""
    lowered = command.lower()
    return (
        any(s in lowered for s in _FORBIDDEN_SUBSTRINGS)
        or bool(_FORBIDDEN_FLAGS.search(command))
        or bool(_ENV_EXPANSION.search(command))
    )


def has_unsafe_constructs(command: str) -> bool:
    """Unsafe syntax, or a `.dostuff` path outside a subagent worktree."""
    return _has_unsafe_syntax(command) or _mentions_dostuff_dir(command)


def _is_dostuff_read(cmd: str, token: str) -> bool:
    """`token`, an argument of `cmd` naming `.dostuff`, is a path the file tools could read too."""
    parts = [part.lower() for part in PurePosixPath(token).parts]
    if token.startswith("-") or _GLOB_CHARS & set(token) or ".." in parts or ".dostuff" not in parts:
        return False
    try:
        resolve_safe_path(token, for_read=True)
    except (ValueError, OSError, RuntimeError):  # outside the project and ~/.dostuff, a `.env*` file, `mcp_config.json`, unknown `~user`
        return False
    return cmd not in _SEARCH_COMMANDS or not _search_reaches_secrets(token)


def _search_reaches_secrets(token: str) -> bool:
    """Searching `token` recursively reads a secret: all of a `.dostuff` dir, of `.dostuff/worktrees`, of one
    worktree (each gets a copy of `.env` and `mcp_config.json`), of a worktree's own `.dostuff`, or of `data`."""
    parts = tuple(part.lower() for part in PurePosixPath(token).parts)
    if ".dostuff" not in parts:
        return False
    below = parts[parts.index(".dostuff") + 1:]
    if below[:1] == ("worktrees",):
        below = below[2:]
    return below in ((), (".dostuff",), ("data",))


def _could_name_secret(token: str) -> bool:
    """`token` is a `.dostuff` path that is, or whose wildcard could expand to, one of `_SECRET_FILES` (or `.env*`)."""
    parts = [part.lower() for part in PurePosixPath(token).parts]
    if ".dostuff" not in parts:
        return False
    name = parts[-1]
    if parts[parts.index(".dostuff") + 1:][:1] == ["data"]:
        return True
    return name.startswith(".env") or "{" in name or any(fnmatch(secret, name) for secret in _SECRET_FILES)


def _exposes_secret(command: str) -> bool:
    """`command` reads or writes a secret file under `.dostuff`, worktrees included.

    The first argument of a search is its pattern, not a path: `grep -rn ".dostuff" src` reads no setting.
    A heuristic like the rest of the `.dostuff` check: `cd .dostuff && cat .env` names no such path.
    """
    if ".dostuff" not in command.lower():
        return False
    text = _HARMLESS_REDIRECT.sub(" ", command).replace("\\", "/")
    segments = split_shell_segments(text)
    if segments is None:
        return bool(_SECRET_PATH.search(text))
    for segment in segments:
        cmd = segment[0].lower().removesuffix(".exe")
        if cmd in _LISTING_COMMANDS:
            continue
        args = segment[1:]
        if cmd in _SEARCH_COMMANDS:
            args = [t for t in args if not t.startswith("-")][1:]
            if any(_search_reaches_secrets(t) for t in args):
                return True
        if any(_could_name_secret(t) for t in args):
            return True
    return False


def _dostuff_refs_are_reads(command: str) -> bool:
    """Every `.dostuff` path is an argument to a read-only command, and one `read_file` could open.

    Only the segment naming a `.dostuff` path has to be read-only: `grep FAIL <log> | sed … | sort -u`
    reads `.dostuff` in its `grep` segment. The other segments are not judged here, only by the ordinary
    shell rules, so `cat .dostuff/x | sh` passes this check and then asks in `ask` like any `sh`. Unsafe
    syntax anywhere still voids it (`find ~/.dostuff -delete` is one read-only segment carrying a
    destructive flag), and so does changing directory: the relative path then means somewhere else, and
    `cd .dostuff && grep -r key .` names no path to check. A command whose own name is a `.dostuff` path
    (`./.dostuff/config.yaml`) runs it, so it is never a read.
    """
    # `2>&1` makes shlex's punctuation split fail and its `>` reads as a redirect, so drop the harmless
    # ones first; a destructive flag (`find ~/.dostuff -delete`) is still caught on the stripped text
    stripped = _HARMLESS_REDIRECT.sub(" ", command)
    if _has_unsafe_syntax(stripped):
        return False
    segments = split_shell_segments(stripped.replace("\\", "/"))  # shlex eats backslashes: `.dostuff\agents`
    if segments is None:
        return False
    for segment in segments:
        cmd = segment[0].lower().removesuffix(".exe")
        if cmd in _DIR_CHANGERS or _mentions_dostuff_dir(segment[0]):
            return False
        refs = [t for t in segment[1:] if _mentions_dostuff_dir(t)]
        if refs and not (_segment_is_read_only(segment) and all(_is_dostuff_read(cmd, t) for t in refs)):
            return False
    return True


def is_read_only_shell(command: str) -> bool:
    """True only if every segment is allow-listed, no unsafe syntax appears, and any `.dostuff` path is a plain read."""
    command = _HARMLESS_REDIRECT.sub(" ", command)
    if not command.strip() or _has_unsafe_syntax(command):
        return False
    segments = split_shell_segments(command)
    if not segments or not all(_segment_is_read_only(seg) for seg in segments):
        return False
    if _exposes_secret(command):
        return False
    return not _mentions_dostuff_dir(command) or _dostuff_refs_are_reads(command)


def shell_approval_pattern(tokens: list[str]) -> str:
    """The pattern remembered by "always": command name, plus a sub-command if it has one."""
    if len(tokens) > 1 and not tokens[1].startswith("-") and re.fullmatch(r"[A-Za-z][\w.-]*", tokens[1]):
        return f"{tokens[0]} {tokens[1]}"
    return tokens[0]


def _shell_pattern_matches(pattern: str, tokens: list[str]) -> bool:
    want = pattern.split()
    return [t.lower() for t in tokens[: len(want)]] == [w.lower() for w in want]


# ── Session state ─────────────────────────────────────────────────────────────

@dataclass
class SessionPermissions:
    """Per-session approvals (`a`/`always`), an optional mode override, and `allow` patterns
    (`dostuff -p --allowed-tools`) that apply in every mode: the `permissions.allow` syntax, or a
    tool name for every call of that tool."""
    mode: PermissionMode | None = None
    approved: set[tuple[str, str]] = field(default_factory=set)
    allow: list[str] = field(default_factory=list)

    def approve(self, keys: list[tuple[str, str]]) -> None:
        self.approved.update(keys)

    def is_approved(self, tool: str, pattern: str) -> bool:
        return (tool, pattern) in self.approved


@dataclass
class PermissionDecision:
    action: Action
    prompt: str = ""
    reason: str = ""
    # What "always" stores in the session; empty → "always" behaves like "yes".
    approval_keys: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class PermissionConfig:
    mode: PermissionMode = "ask"
    allow: list[str] = field(default_factory=list)


_EDIT_TOOLS = ("edit_file", "multi_edit")


def _config_allows(tool: str, tokens: list[str] | None, cfg: PermissionConfig) -> bool:
    return _patterns_allow(cfg.allow, tool, tokens)


def _patterns_allow(patterns: list[str], tool: str, tokens: list[str] | None) -> bool:
    for pattern in patterns:
        if tokens is None:
            if pattern == tool:
                return True
        elif _shell_pattern_matches(pattern, tokens):
            return True
    return False


def _same_grant(tool: str) -> tuple[str, ...]:
    """The tools a rule for `tool` covers: `edit_file` and `multi_edit` grant the same capability, in both directions."""
    return _EDIT_TOOLS if tool in _EDIT_TOOLS else (tool,)


def _describe_call(tool: str, args: dict[str, Any]) -> str:
    if tool == "bash_command":
        mode = ""
        if args.get("run_in_background"):  # the user is approving something that keeps running
            mode = f' (background job "{args["background_id"]}")' if args.get("background_id") else " (background job)"
        return f"Run shell command{mode}:\n  {args.get('command', '')}"
    if tool == "call_mcp_tool":
        return f"Call MCP tool '{args.get('name', '')}' with arguments:\n  {str(args.get('arguments', {}))[:500]}"
    if tool == "web_fetch":
        return f"Fetch URL:\n  {str(args.get('url', ''))[:500]}"
    if tool == "web_search":
        return f"Search the web:\n  {str(args.get('query', ''))[:500]}"
    if tool == "edit_config":
        return f"Change dostuff settings ({args.get('scope') or 'user'} config.yaml)"
    if tool == "set_mcp_server":
        return f"Add or replace MCP server '{args.get('name', '')}' ({args.get('scope') or 'user'} mcp_config.json)"
    if tool == "create_agent_type":
        where = "~/.dostuff/agents" if args.get("scope") == "user" else ".dostuff/agents"
        return f"Create agent type '{args.get('name', '')}' ({where})"
    if tool == "remove_mcp_server":
        return f"Remove MCP server '{args.get('name', '')}' ({args.get('scope') or 'user'} mcp_config.json)"
    if "path" in args:
        return f"{tool} on '{args['path']}'"
    return f"{tool}({str(args)[:500]})"


def url_host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _is_settings_path(path: str) -> bool:
    """A `.dostuff` dir in the path below the cwd; the cwd itself may be inside one (a git worktree).

    A worktree is a checkout of the project, so its files are ordinary ones, as they are for the shell;
    its own `.dostuff` still counts, and `..` could leave it.
    """
    pure = PurePath(path)
    if pure.is_absolute():
        try:
            pure = pure.relative_to(project_root())
        except ValueError:
            pass
    parts = pure.parts
    if len(parts) > 3 and ".." not in parts and tuple(p.lower() for p in parts[:2]) == (".dostuff", "worktrees"):
        parts = parts[3:]
    return ".dostuff" in parts


def _plan_denies(kind: ToolKind, args: dict[str, Any]) -> bool:
    if kind in ("write", "settings"):
        return True
    if kind == "shell":
        return not (is_read_only_shell(str(args.get("command", ""))) and not has_unsafe_constructs(str(args.get("cwd") or "")))
    return False


def _mentions_dostuff_dir(text: str) -> bool:
    """`.dostuff` named as a path in a shell command or cwd, other than a path into a subagent worktree."""
    lowered = text.lower()
    if ".." not in lowered:  # same worktree exemption as has_unsafe_constructs
        lowered = _WORKTREES_PREFIX.sub("", lowered)
    return bool(_DOSTUFF_REF.search(lowered))


def _shell_touches_settings(args: dict[str, Any]) -> bool:
    """Whether a shell command could write a `.dostuff` path, which makes it a settings change.

    A `.dostuff` path is safe when the segment naming it is a read-only command and the path is one
    `read_file` could open. Segments that name no `.dostuff` path are irrelevant here: they run under
    the ordinary shell rules, so `grep FAIL <job log> | sed … | sort -u` reads a log in its `grep`
    segment and only filters that text afterwards. `cwd` is stricter: running anything *inside* the
    settings dir stays a settings change.
    """
    command, cwd = str(args.get("command") or ""), str(args.get("cwd") or "")
    if _mentions_dostuff_dir(cwd) or _exposes_secret(command):  # the worktree exemption doesn't cover its copied secrets
        return True
    if not _mentions_dostuff_dir(command):
        return False
    in_root = (project_root() / cwd).resolve() == project_root().resolve()  # `.dostuff/x` is relative to the cwd
    return not (in_root and _dostuff_refs_are_reads(command))


def effective_kind(kind: ToolKind, args: dict[str, Any]) -> ToolKind:
    """write_file / edit_file on ./.dostuff/config.yaml, mcp_config.json, .env … count as settings changes.

    So does a shell command naming `.dostuff` (in `command` or `cwd`), unless it only reads what `read_file`
    could: `echo … >> ~/.dostuff/config.yaml` adds a user hook, which no trust check covers, so `ask` and
    `allowlist` must prompt for it however the permission rules are set.
    Only a `.dostuff` path counts — `cli_tui.DostuffTUI` is a dotted attribute, not a settings path.
    A heuristic: a name built indirectly (`.dost''uff`, a variable holding it) is not caught here.
    """
    if kind == "write" and _is_settings_path(str(args.get("path") or "")):
        return "settings"
    if kind == "shell" and _shell_touches_settings(args):
        return "settings"
    return kind


def _auto_refusal(tool: str, args: dict[str, Any]) -> str:
    """Why auto mode refuses this settings call, or "". Auto never waits, so what must stay closed is refused.

    Everything else in `.dostuff` runs. A heuristic for the shell: a name built indirectly is not caught.
    """
    if args.get("path"):
        path = Path(str(args["path"]))
        if is_denied_for_read(path):
            return _SECRET_REFUSAL
        name = path.name.lower()
        return RULES_REFUSAL if name == "trusted_hooks.json" else _CONFIG_REFUSAL if name == "config.yaml" else ""
    if tool != "bash_command":
        return ""
    command = str(args.get("command") or "")
    if _exposes_secret(command):
        return _SECRET_REFUSAL
    return _CONFIG_REFUSAL if _RULES_FILE.search(command) else ""


def forced_ask(tool: str, args: dict[str, Any], preview: str = "", note: str = "") -> PermissionDecision:
    """A prompt in every mode, without "always" — e.g. when a PreToolUse hook asks (hooks spec D2 step 4)."""
    description = _describe_call(tool, args)
    prompt = "\n".join(part for part in (description, note, preview) if part)
    return PermissionDecision("ask", prompt=prompt, reason=description)


def check(
    tool: str,
    kind: ToolKind,
    args: dict[str, Any],
    session: SessionPermissions | None,
    cfg: PermissionConfig,
    preview: str = "",
) -> PermissionDecision:
    """Decide whether `tool(args)` may run. No prompting, and no I/O beyond resolving paths."""
    mode = (session.mode if session and session.mode else cfg.mode)
    description = _describe_call(tool, args)
    prompt = f"{description}\n{preview}".rstrip()
    kind = effective_kind(kind, args)
    if mode == "plan" and _plan_denies(kind, args):
        return PermissionDecision("deny", reason=(
            f"Plan mode: not run ({description.splitlines()[0]}). Only read-only tools run in plan mode; "
            "put this step in your plan instead."
        ))
    if kind == "settings" and mode != "auto":  # no "always" (no approval keys)
        return PermissionDecision("ask", prompt=prompt, reason=description)
    refusal = _auto_refusal(tool, args) if kind == "settings" else ""
    if refusal:
        return PermissionDecision("deny", reason=f"Auto mode: not run ({description.splitlines()[0]}). {refusal}")
    if kind == "read" or mode == "auto":
        return PermissionDecision("allow")
    grants = _same_grant(tool)
    if session and any(name in session.allow for name in grants):
        return PermissionDecision("allow")

    if kind == "shell":
        command = str(args.get("command", ""))
        cwd = str(args.get("cwd") or "")
        segments = split_shell_segments(command) or []
        if is_read_only_shell(command) and not has_unsafe_constructs(cwd):
            return PermissionDecision("allow")
        unsafe = has_unsafe_constructs(command) or has_unsafe_constructs(cwd) or not segments
        keys = [] if unsafe else [(tool, shell_approval_pattern(seg)) for seg in segments]
        if not unsafe:
            def segment_ok(seg: list[str]) -> bool:
                if _segment_is_read_only(seg) or (session and session.is_approved(tool, shell_approval_pattern(seg))):
                    return True
                if session and _patterns_allow(session.allow, tool, seg):
                    return True
                return mode == "allowlist" and _config_allows(tool, seg, cfg)
            if all(segment_ok(seg) for seg in segments):
                return PermissionDecision("allow")
        return PermissionDecision("ask", prompt=prompt, reason=command, approval_keys=keys)

    if kind == "mcp":
        pattern = f"{tool}:{args.get('name', '')}"
    elif kind == "network" and "url" in args:  # "always" and permissions.allow work per domain ("web_fetch:docs.python.org")
        pattern = f"{tool}:{url_host(str(args.get('url', '')))}"
    else:
        pattern = tool
    if session and (session.is_approved(tool, pattern) or pattern in session.allow):
        return PermissionDecision("allow")
    if mode == "allowlist" and any(_config_allows(name, None, cfg) for name in (*grants, pattern)):
        return PermissionDecision("allow")
    keys = [(name, name) for name in grants] if len(grants) > 1 else [(tool, pattern)]
    return PermissionDecision("ask", prompt=prompt, reason=description, approval_keys=keys)


Answer = Literal["yes", "always", "no", "each"]


@dataclass
class ApprovalReply:
    """The user's answer to an approval prompt. "each" only answers a batch prompt
    (ask about the calls one by one); `feedback` goes to the model with a denial."""
    answer: Answer
    feedback: str = ""

    def __str__(self) -> str:
        return self.answer


def to_reply(raw: Any) -> ApprovalReply:
    """Normalize what an adapter's ask()/ask_batch() returned."""
    if isinstance(raw, ApprovalReply):
        return raw
    if str(raw).strip().lower() == "each":
        return ApprovalReply("each")
    return ApprovalReply(parse_answer(str(raw)))


_ALWAYS_NAMES = {
    "write_file": "file writes", "edit_file": "file edits", "multi_edit": "file edits",
    "delete_file": "file deletions",
}


def describe_approval_keys(keys: list[tuple[str, str]]) -> str:
    """What "always" will allow, for the prompt option: "file writes, `git commit` commands"."""
    names: list[str] = []
    for tool, pattern in keys:
        if tool == "bash_command":
            name = f"`{pattern}` commands"
        elif tool == "call_mcp_tool":
            name = f"MCP tool {pattern.partition(':')[2] or pattern}"
        elif tool == "web_fetch":
            name = f"fetches from {pattern.partition(':')[2] or 'this site'}"
        elif tool == "web_search":
            name = "web searches"
        else:
            name = _ALWAYS_NAMES.get(tool, tool)
        if name not in names:
            names.append(name)
    return ", ".join(names)


def parse_answer(text: str) -> Literal["yes", "always", "no"]:
    """Map a free-text reply to an approval answer; anything unrecognised denies."""
    answer = text.strip().lower()
    if answer in ("y", "yes"):
        return "yes"
    if answer in ("a", "always"):
        return "always"
    return "no"
