"""Trust for project-supplied hooks (spec D4).

User hooks always run. Project hooks arrive with a cloned repo, so they run only once the user trusts them:
`hooks.project_trust` (user file only) is `allow`, `deny`, or `ask` — ask once per project before the first
project hook runs, and remember a yes in `~/.dostuff/trusted_hooks.json` keyed by the project root and a sha256
over the normalised project entries plus every file under `.dostuff/hooks/`. Editing either re-asks.
"""
import hashlib
import json
from pathlib import Path
from typing import Any

from dostuff.helpers.hooks.load_hooks import HookEntry

HOOKS_DIR_CAP = 1_000_000  # bytes of .dostuff/hooks/ hashed; over it the digest is None and trust is never stored
TRUST_PROMPT = ("This project defines hooks: commands dostuff would run on your machine, unsandboxed, "
                "with your privileges.\n{listing}\nRun this project's hooks?")

# (project root, digest) → the user's answer in this process; a worktree subagent shares the session's root
_decisions: dict[tuple[str, str | None], bool] = {}


def trust_file() -> Path:
    return Path.home() / ".dostuff" / "trusted_hooks.json"


def project_root_of(entry: HookEntry) -> Path:
    """The project a project entry came from: `<root>/.dostuff/config.yaml` → `<root>`."""
    return Path(entry.origin).parent.parent


def project_entries(hooks: dict[str, list[HookEntry]]) -> list[HookEntry]:
    return [e for entries in hooks.values() for e in entries if e.source == "project"]


def digest(entries: list[HookEntry], hooks_dir: Path) -> str | None:
    """sha256 over the parsed entries (so comment-only YAML edits don't count) and the hook scripts' bytes."""
    h = hashlib.sha256()
    normalised = [[e.event, e.name, e.command, e.matcher, e.if_pattern, e.timeout, e.fail_closed, e.broken]
                  for e in entries]
    h.update(json.dumps(normalised).encode())
    if hooks_dir.is_dir():
        total = 0
        for path in sorted((p for p in hooks_dir.rglob("*") if p.is_file()),
                           key=lambda p: p.relative_to(hooks_dir).as_posix()):
            data = path.read_bytes()
            total += len(data)
            if total > HOOKS_DIR_CAP:
                return None
            h.update(path.relative_to(hooks_dir).as_posix().encode() + b"\0" + data + b"\0")
    return h.hexdigest()


def trust_key(entries: list[HookEntry]) -> tuple[str, str | None]:
    root = project_root_of(entries[0]).resolve()
    return str(root), digest(entries, root / ".dostuff" / "hooks")


def _records() -> dict[str, str]:
    try:
        data = json.loads(trust_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def is_trusted(key: tuple[str, str | None]) -> bool:
    root, hashed = key
    return hashed is not None and _records().get(root) == hashed


def remember(key: tuple[str, str | None]) -> None:
    root, hashed = key
    if hashed is None:
        return
    records = _records()
    records[root] = hashed
    path = trust_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def decide(key: tuple[str, str | None], allowed: bool) -> None:
    """Settle the question for this process without asking (headless decides up front)."""
    _decisions[key] = allowed


def decision(key: tuple[str, str | None]) -> bool | None:
    """True/False if already known without asking, None if the user still has to be asked."""
    if key in _decisions:
        return _decisions[key]
    if is_trusted(key):
        _decisions[key] = True
        return True
    return None


def status(policy: str, hooks: dict[str, list[HookEntry]]) -> str:
    """How project hooks stand right now, without asking: "trusted", "untrusted" or "disabled" (for /hooks)."""
    entries = project_entries(hooks)
    if policy == "allow":
        return "trusted"
    if policy == "deny":
        return "disabled"
    return "trusted" if entries and decision(trust_key(entries)) else "untrusted"


def headless_decision(policy: str, hooks: dict[str, list[HookEntry]], trust_flag: bool) -> tuple[str, str]:
    """(refusal, warning) for a run with nobody to ask (`-p`); settles the trust question for this process (D4).

    A trust record or `--trust-project-hooks` lets project hooks run. Otherwise they are skipped with a warning,
    unless one is `fail_closed`: skipping a guardrail would be failing open, so the run is refused.
    """
    entries = project_entries(hooks)
    if policy != "ask" or not entries:
        return "", ""
    key = trust_key(entries)
    allowed = trust_flag or is_trusted(key)
    decide(key, allowed)
    if allowed:
        return "", ""
    origin = entries[0].origin
    guards = [e for e in entries if e.fail_closed]
    if guards:
        names = ", ".join(f"{e.event} '{e.name}'" for e in guards)
        return (f"Refusing to start: untrusted fail_closed project hook(s) {names} in {origin}. Review them and "
                "rerun with --trust-project-hooks, or trust them once in the TUI."), ""
    return "", (f"{len(entries)} project hooks from {origin} not trusted; skipped. Review them and rerun with "
                "--trust-project-hooks, or trust them once in the TUI.")


def prompt_text(entries: list[HookEntry]) -> str:
    lines = [f"  {e.event} '{e.name}' ({e.origin}): {e.command}" for e in entries]
    return TRUST_PROMPT.format(listing="\n".join(lines))


async def project_allowed(policy: str, hooks: dict[str, list[HookEntry]], adapter: Any) -> bool:
    """Whether project hooks may run now, asking the user at most once per (project, digest)."""
    if policy in ("allow", "deny"):
        return policy == "allow"
    entries = project_entries(hooks)
    if not entries:
        return False
    key = trust_key(entries)
    known = decision(key)
    if known is not None:
        return known
    if adapter is None or not hasattr(adapter, "ask"):
        return False  # nobody to ask now; a later dispatch with a UI may still ask
    from dostuff.agent.permissions import to_reply  # lazy: permissions imports the hooks package
    # The TUI's ask_trust has a plain "no"; ask()'s "no" asks what the model should do differently
    ask = getattr(adapter, "ask_trust", None) or adapter.ask
    allowed = to_reply(await ask(prompt_text(entries))).answer in ("yes", "always")
    _decisions[key] = allowed
    if allowed:
        remember(key)
    return allowed
