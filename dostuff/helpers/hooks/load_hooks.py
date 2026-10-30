"""Parse and validate the `hooks:` block of one config file, and merge user + project files (spec D9, D10).

Nothing here raises on bad config: invalid items are dropped with a warning (CON-003), except
`fail_closed` entries, which are kept as *broken* entries that always deny (D7).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop", "SubagentStop")
TOOL_EVENTS = ("PreToolUse", "PostToolUse")
FAIL_CLOSED_EVENTS = ("PreToolUse", "UserPromptSubmit")  # the events where a hook can block before anything happens
SESSION_START_SOURCES = ("startup", "resume", "clear", "compact")
PROJECT_TRUST = ("ask", "allow", "deny")
SETTINGS_KEYS = ("project_trust", "stop_block_limit")

DEFAULT_TIMEOUT_S = 30
TIMEOUT_RANGE = (1, 300)
DEFAULT_STOP_BLOCK_LIMIT = 3
STOP_BLOCK_LIMIT_RANGE = (1, 10)

Source = Literal["user", "project"]


@dataclass(frozen=True)
class HookEntry:
    event: str
    command: str
    name: str
    source: Source
    origin: str
    matcher: str | None = None
    if_pattern: str | None = None
    timeout: float = DEFAULT_TIMEOUT_S
    fail_closed: bool = False
    broken: str | None = None  # the validation warning, for a fail_closed entry that always denies
    # Compiled once at load; None = match everything (no pattern, "*", or an invalid pattern on a broken entry)
    matcher_re: re.Pattern[str] | None = field(default=None, repr=False, compare=False)
    if_re: re.Pattern[str] | None = field(default=None, repr=False, compare=False)


@dataclass
class LoadedHooks:
    """Both files' hooks, merged per D10."""
    hooks: dict[str, list[HookEntry]] = field(default_factory=dict)
    project_trust: str = "ask"
    stop_block_limit: int = DEFAULT_STOP_BLOCK_LIMIT
    warnings: list[str] = field(default_factory=list)


def _compile(pattern: str) -> tuple[re.Pattern[str] | None, str | None]:
    try:
        return re.compile(pattern), None
    except re.error as e:
        return None, str(e)


def _timeout(value: Any, where: str, warnings: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        warnings.append(f"{where}.timeout: {value!r} is not a number; using {DEFAULT_TIMEOUT_S}s.")
        return DEFAULT_TIMEOUT_S
    low, high = TIMEOUT_RANGE
    if not low <= value <= high:
        clamped = max(low, min(value, high))
        warnings.append(f"{where}.timeout: {value!r} is outside {low}-{high}s; using {clamped}s.")
        return clamped
    return value


def _entry(
    event: str, group: dict[str, Any], hook: dict[str, Any], *, where: str, index: int,
    source: Source, origin: str, warnings: list[str],
) -> HookEntry | None:
    """One hook of a group, with the group's matcher/if/fail_closed/timeout; None = dropped."""
    fail_closed = hook.get("fail_closed", group.get("fail_closed", False))
    if not isinstance(fail_closed, bool):
        warnings.append(f"{where}.fail_closed: {fail_closed!r} is not true/false; using false.")
        fail_closed = False
    if fail_closed and event not in FAIL_CLOSED_EVENTS:
        warnings.append(f"{where}.fail_closed: only applies to {' and '.join(FAIL_CLOSED_EVENTS)}; ignored.")
        fail_closed = False
    timeout = _timeout(hook["timeout"], where, warnings) if "timeout" in hook else (
        _timeout(group["timeout"], where, warnings) if "timeout" in group else DEFAULT_TIMEOUT_S)
    name = hook.get("name")
    name = str(name) if name else f"{event}#{index}"

    problems: list[str] = []
    command = hook.get("command")
    if hook.get("type") != "command":
        problems.append(f"type {hook.get('type')!r} is not supported (only 'command')")
    if not isinstance(command, str) or not command.strip():
        problems.append("missing 'command'")
        command = ""

    matcher = group.get("matcher")
    matcher_re = None
    if matcher is not None and not isinstance(matcher, str):
        problems.append(f"matcher {matcher!r} is not a string")
        matcher = None
    if matcher in ("", "*"):
        matcher = None
    if matcher is not None:
        matcher_re, error = _compile(matcher)
        if error:
            problems.append(f"invalid matcher regex {matcher!r} ({error})")

    if_pattern = group.get("if")
    if_re = None
    if if_pattern is not None and event not in TOOL_EVENTS:
        warnings.append(f"{where}.if: only applies to {' and '.join(TOOL_EVENTS)}; ignored.")
        if_pattern = None
    if if_pattern is not None and not isinstance(if_pattern, str):
        problems.append(f"if {if_pattern!r} is not a string")
        if_pattern = None
    if if_pattern:
        if_re, error = _compile(if_pattern)
        if error:
            hint = " — inline flags such as (?i) must come first" if "global flags" in error else ""
            problems.append(f"invalid if regex {if_pattern!r} ({error}){hint}")

    broken = None
    if problems:
        problem = "; ".join(problems)
        if not fail_closed:
            warnings.append(f"{where} '{name}': {problem}; entry dropped.")
            return None
        # A typo in a guardrail must not turn into no guardrail (D7)
        broken = problem
        warnings.append(f"{where} '{name}': {problem}; fail_closed, so it denies every call it matches.")
    return HookEntry(
        event=event, command=command, name=name, source=source, origin=origin,
        matcher=matcher, if_pattern=if_pattern or None, timeout=timeout, fail_closed=fail_closed,
        broken=broken, matcher_re=matcher_re, if_re=if_re,
    )


def load_hooks(
    raw: Any, *, source: Source, origin: str, counter: dict[str, int] | None = None,
) -> tuple[list[HookEntry], list[str]]:
    """Typed entries from one file's `hooks:` block, in config order, plus warnings.

    Unnamed hooks are called `<event>#<n>`, n counting every hook of the event in config order.
    Pass the same `counter` for the next file so user and project names don't collide.
    """
    warnings: list[str] = []
    entries: list[HookEntry] = []
    if raw is None:
        return entries, warnings
    if not isinstance(raw, dict):
        warnings.append(f"hooks ({source} config): expected a mapping of events, got {type(raw).__name__}; ignored.")
        return entries, warnings
    counts = counter if counter is not None else {}
    for event, groups in raw.items():
        if event in SETTINGS_KEYS:
            continue
        if event not in EVENTS:
            warnings.append(f"hooks.{event} ({source} config): unknown event ignored. Known: {', '.join(EVENTS)}.")
            continue
        if groups is None:
            continue
        if not isinstance(groups, list):
            warnings.append(f"hooks.{event} ({source} config): expected a list of matcher groups; ignored.")
            continue
        for gi, group in enumerate(groups):
            where = f"hooks.{event}[{gi}] ({source} config)"
            if not isinstance(group, dict):
                warnings.append(f"{where}: expected a mapping with 'hooks'; ignored.")
                continue
            group_hooks = group.get("hooks")
            if not isinstance(group_hooks, list):
                warnings.append(f"{where}.hooks: expected a list of hooks; group ignored.")
                continue
            for hi, hook in enumerate(group_hooks):
                hook_where = f"hooks.{event}[{gi}].hooks[{hi}] ({source} config)"
                if not isinstance(hook, dict):
                    warnings.append(f"{hook_where}: expected a mapping; ignored.")
                    continue
                counts[event] = counts.get(event, 0) + 1
                entry = _entry(event, group, hook, where=hook_where, index=counts[event],
                               source=source, origin=origin, warnings=warnings)
                if entry is not None:
                    entries.append(entry)
    return entries, warnings


def _read_block(path: Path, source: Source, warnings: list[str]) -> Any:
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        warnings.append(f"hooks ({source} config): could not read {path}: {e}")
        return None
    return data.get("hooks") if isinstance(data, dict) else None


def _stop_block_limit(value: Any, source: Source, warnings: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        warnings.append(
            f"hooks.stop_block_limit ({source} config): {value!r} is not a whole number; "
            f"using {DEFAULT_STOP_BLOCK_LIMIT}.")
        return DEFAULT_STOP_BLOCK_LIMIT
    low, high = STOP_BLOCK_LIMIT_RANGE
    if not low <= value <= high:
        clamped = max(low, min(value, high))
        warnings.append(f"hooks.stop_block_limit ({source} config): {value} is outside {low}-{high}; using {clamped}.")
        return clamped
    return value


def load_merged_hooks(user_path: Path, project_path: Path) -> LoadedHooks:
    """Hooks from `~/.dostuff/config.yaml` then `./.dostuff/config.yaml`, loaded separately (D10).

    Project entries are appended after the user's per event, never replacing them;
    `project_trust` is read from the user file only, and `stop_block_limit` is the stricter of the two.
    """
    loaded = LoadedHooks()
    limits: list[int] = []
    counter: dict[str, int] = {}
    files: tuple[tuple[Path, Source], ...] = ((user_path, "user"), (project_path, "project"))
    for path, source in files:
        raw = _read_block(path, source, loaded.warnings)
        entries, warnings = load_hooks(raw, source=source, origin=str(path), counter=counter)
        loaded.warnings += warnings
        for entry in entries:
            loaded.hooks.setdefault(entry.event, []).append(entry)
        if not isinstance(raw, dict):
            continue
        if "stop_block_limit" in raw:
            limits.append(_stop_block_limit(raw["stop_block_limit"], source, loaded.warnings))
        if "project_trust" in raw:
            trust = str(raw["project_trust"]).lower()
            if source == "project":
                # Otherwise a cloned repo could trust its own hooks (D4)
                loaded.warnings.append(
                    "hooks.project_trust (project config): only read from ~/.dostuff/config.yaml; ignored.")
            elif trust in PROJECT_TRUST:
                loaded.project_trust = trust
            else:
                loaded.warnings.append(
                    f"hooks.project_trust (user config): {raw['project_trust']!r} is not one of "
                    f"{', '.join(PROJECT_TRUST)}; using ask.")
    if limits:
        loaded.stop_block_limit = min(limits)
    return loaded
