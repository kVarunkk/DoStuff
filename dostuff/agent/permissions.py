"""Permission layer: decides whether a tool call runs, asks the user, or is denied.

Tools never approve themselves. `run_tool` strips any model-supplied approval keys and
calls `check()` before dispatch; ASK decisions are resolved through `adapter.ask()`.

An adapter's `ask()` may return a plain string ("y", "always" …) or an `ApprovalReply`,
which can carry feedback for the model. Adapters with `ask_batch()` get one prompt for
all calls of a model response that need approval.

Modes (`permissions.mode` in config, overridable per session):
  ask       — every mutating call asks, except read-only shell commands (allow-list below)
  allowlist — as `ask`, plus `permissions.allow` patterns from config are auto-approved
  auto      — nothing asks (explicit opt-in)

Settings changes (kind "settings": config.yaml, MCP servers) ask in every mode and have no
"always": they can change these rules or add a command that runs on the user's machine.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Literal
from urllib.parse import urlparse

from dostuff.helpers.tools.project_root import project_root

PermissionMode = Literal["ask", "allowlist", "auto"]
ToolKind = Literal["read", "write", "shell", "mcp", "network", "settings"]
Action = Literal["allow", "ask", "deny"]

PERMISSION_MODES: tuple[str, ...] = ("ask", "allowlist", "auto")

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
    ">", "<", "`", "$(", "${", "out-file", "set-content", "add-content", "env:", ".env", ".dostuff",
)
# Flags that make otherwise read-only commands write, delete or execute (find -exec, rg --pre, git --output …).
_FORBIDDEN_FLAGS = re.compile(
    r"(?<![\w-])(-delete|-exec\w*|-ok\w*|-fprint\w*|-fls|--output|--pre)(?![\w-])",
    re.IGNORECASE,
)
_ENV_EXPANSION = re.compile(r"\$[A-Za-z_]")
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


def has_unsafe_constructs(command: str) -> bool:
    """Redirection, substitution, env expansion, or destructive flags anywhere in the string."""
    lowered = command.lower()
    if ".." not in lowered:  # `.dostuff/worktrees/../config.yaml` stays forbidden
        lowered = _WORKTREES_PREFIX.sub("", lowered)
    return (
        any(s in lowered for s in _FORBIDDEN_SUBSTRINGS)
        or bool(_FORBIDDEN_FLAGS.search(command))
        or bool(_ENV_EXPANSION.search(command))
    )


def is_read_only_shell(command: str) -> bool:
    """True only if every segment is allow-listed and no unsafe construct appears."""
    if not command.strip() or has_unsafe_constructs(command):
        return False
    segments = split_shell_segments(command)
    if not segments:
        return False
    return all(_segment_is_read_only(seg) for seg in segments)


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
    """Per-session approvals (`a`/`always`) and an optional mode override."""
    mode: PermissionMode | None = None
    approved: set[tuple[str, str]] = field(default_factory=set)

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


def _config_allows(tool: str, tokens: list[str] | None, cfg: PermissionConfig) -> bool:
    for pattern in cfg.allow:
        if tokens is None:
            if pattern == tool:
                return True
        elif _shell_pattern_matches(pattern, tokens):
            return True
    return False


def _describe_call(tool: str, args: dict[str, Any]) -> str:
    if tool == "bash_command":
        return f"Run shell command:\n  {args.get('command', '')}"
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
    """A `.dostuff` dir in the path below the cwd; the cwd itself may be inside one (a git worktree)."""
    pure = PurePath(path)
    if pure.is_absolute():
        try:
            pure = pure.relative_to(project_root())
        except ValueError:
            pass
    return ".dostuff" in pure.parts


def check(
    tool: str,
    kind: ToolKind,
    args: dict[str, Any],
    session: SessionPermissions | None,
    cfg: PermissionConfig,
    preview: str = "",
) -> PermissionDecision:
    """Decide whether `tool(args)` may run. Pure function — no I/O, no prompting."""
    mode = (session.mode if session and session.mode else cfg.mode)
    description = _describe_call(tool, args)
    prompt = f"{description}\n{preview}".rstrip()
    if kind == "write" and _is_settings_path(str(args.get("path") or "")):
        kind = "settings"  # write_file / edit_file on ./.dostuff/config.yaml, mcp_config.json, .env …
    if kind == "settings":  # every mode, and no "always" (no approval keys)
        return PermissionDecision("ask", prompt=prompt, reason=description)
    if kind == "read" or mode == "auto":
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
    if session and session.is_approved(tool, pattern):
        return PermissionDecision("allow")
    if mode == "allowlist" and (_config_allows(tool, None, cfg) or _config_allows(pattern, None, cfg)):
        return PermissionDecision("allow")
    return PermissionDecision("ask", prompt=prompt, reason=description, approval_keys=[(tool, pattern)])


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
    "write_file": "file writes", "edit_file": "file edits", "delete_file": "file deletions",
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
