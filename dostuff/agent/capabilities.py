"""Skills and agent types that appear after the system prompt was built (plan/2026-11-11-mid-run-skill-agent-refresh).

The system prompt lists the skills and agent types once, at build time, and never changes between turns (the cached
prefix, cache-stable-context CON-003). What changed since is announced in an append-only hidden step instead:

- the **stat key** (`skills_signature()` + the agent folders' own signature) decides *whether* to look again;
- the **capability set** — sorted `skill:<name>:<hash>` / `agent:<name>:<hash>` strings — says *what* the model would
  be told; a body or whitespace edit leaves it equal;
- the **baseline** is the set the prompt was built with (`build_base_instructions()`'s third element, a `loop()`
  parameter — never module state), the **belief** the set the newest capability step announced.

Light on import: the TUI loads this before its first frame.
"""
import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dostuff.helpers.skills import discover_skills as ds
from dostuff.helpers.tools.resolve_safe_path import set_skill_roots, skill_roots

ALL_KINDS: tuple[str, ...] = ("skill", "agent")
SKILL_KINDS: tuple[str, ...] = ("skill",)  # a typed subagent: its prompt lists no agent types, and it can't start one

NOTE_HEAD = ("## Skills and agent types: changes\n"
             "These update the lists in the system prompt and any earlier note like this one.")


@dataclass(frozen=True)
class Snapshot:
    """What is true on disk now: the discovered skill rows (manual-only ones included, for the TUI's own view), the
    agent types by name → description, and the capability set restricted to the kinds asked for."""
    skills: list[dict[str, Any]]
    agents: dict[str, str]
    capabilities: tuple[str, ...]


def _hash(description: str) -> str:
    return hashlib.sha1(" ".join(description.split()).encode("utf-8")).hexdigest()[:12]


def capability_set(skills: Iterable[dict[str, Any]], agents: dict[str, str], kinds: Iterable[str]) -> tuple[str, ...]:
    """The sorted capability strings for `kinds`. A manual-only skill is never in it (CON-008): the prompt does not
    list one, so neither does what it claims, and one appearing or going away announces nothing."""
    kinds = tuple(kinds)
    entries: list[str] = []
    if "skill" in kinds:
        entries += [f"skill:{s['name']}:{_hash(str(s.get('description', '')))}" for s in ds.model_visible_skills(list(skills))]
    if "agent" in kinds:
        entries += [f"agent:{name}:{_hash(description)}" for name, description in agents.items()]
    return tuple(sorted(entries))


def agent_type_descriptions() -> dict[str, str]:
    """Every agent type the prompt lists, by name, fork included (subagents spec D11, D13). Blocking: re-parses only
    the agent files whose (path, mtime, size) moved."""
    from dostuff.agent.subagents.definitions import current_definitions
    from dostuff.agent.subagents.fork import FORK_DESCRIPTION, FORK_TYPE
    descriptions = {name: d.description for name, d in current_definitions().types.items()}
    descriptions[FORK_TYPE] = FORK_DESCRIPTION
    return descriptions


# The last discovery's rows and the read roots it set, against the stat key they were read under. A cache of the
# disk, not a baseline: every loop shares it safely, because it only ever answers "what is on disk now".
_lock = threading.Lock()
_cached: tuple[tuple, list[dict[str, Any]], frozenset[Path]] | None = None


def reset_cache() -> None:
    global _cached
    with _lock:
        _cached = None


def _discover(signature: tuple) -> list[dict[str, Any]]:
    global _cached
    skills = ds.discover_skills()  # sets the roots itself
    with _lock:
        _cached = (signature, skills, skill_roots())
    return skills


def discover_for_prompt() -> list[dict[str, Any]]:
    """discover_skills(), and remember its rows under the stat key taken just before (a change in between only
    makes the next snapshot look again). For the prompt builders, so their discovery primes the snapshot cache."""
    return _discover(ds.skills_signature())


def current_skills() -> list[dict[str, Any]]:
    """The skills on disk now: discovery only when the stat key moved (CON-009), else the cached rows. Either way the
    read roots are the ones those rows set, so what is announced and what file tools may read never differ (D8)."""
    signature = ds.skills_signature()
    with _lock:
        cached = _cached
    if cached is None or cached[0] != signature:
        return _discover(signature)
    if skill_roots() != cached[2]:  # something else replaced them since (resolving every dir again is the slow part)
        set_skill_roots(cached[2])
    return cached[1]


def take_snapshot(kinds: Iterable[str] = ALL_KINDS) -> Snapshot:
    """The skills and agent types on disk now, and their capability set for `kinds`. Blocking (a stat pass, and a
    discovery when something moved): on the event loop, call it through asyncio.to_thread."""
    kinds = tuple(kinds)
    skills = current_skills() if "skill" in kinds else []
    agents = agent_type_descriptions() if "agent" in kinds else {}
    return Snapshot(skills, agents, capability_set(skills, agents, kinds))


def _parse(entry: str) -> tuple[str, str]:
    kind, _, rest = entry.partition(":")
    return kind, rest.rpartition(":")[0]


def delta_text(belief: Iterable[str], snapshot: Snapshot) -> str:
    """The note telling the model what differs between what it was told (`belief`) and `snapshot`; "" when nothing
    does. A pure function of its inputs, so the same state always gets the same bytes. Additions carry the name,
    description and (for a skill) dir, so a model that reads only this note still knows where to look."""
    told, now = set(belief), set(snapshot.capabilities)
    added = [_parse(e) for e in sorted(now - told)]
    removed = [_parse(e) for e in sorted(told - now)]
    added_names = set(added)
    changed = [e for e in added if e in set(removed)]
    new = [e for e in added if e not in set(changed)]
    gone = [e for e in removed if e not in added_names]
    skills = {s["name"]: s for s in ds.model_visible_skills(snapshot.skills)}

    def skill_lines(names: list[str]) -> list[str]:
        lines = []
        for name in names:
            s = skills.get(name, {})
            lines += [f"- name: {name}", f"  description: {s.get('description', '')}", f"  dir: {s.get('dir', '')}"]
        return lines

    def agent_lines(names: list[str]) -> list[str]:
        lines = []
        for name in names:
            lines += [f"- name: {name}", f"  description: {' '.join(snapshot.agents.get(name, '').split())}"]
        return lines

    def names(entries: list[tuple[str, str]], kind: str) -> list[str]:
        return sorted(name for k, name in entries if k == kind)

    lines: list[str] = []
    for title, kind, render in (
        ("New skill:", "skill", skill_lines),
        ("New agent type (start it with agent(subagent_type=<name>)):", "agent", agent_lines),
    ):
        picked = names(new, kind)
        if picked:
            lines += [title, *render(picked)]
    for title, kind, render in (("Changed skill:", "skill", skill_lines), ("Changed agent type:", "agent", agent_lines)):
        picked = names(changed, kind)
        if picked:
            lines += [title, *render(picked)]
    if gone:
        label = {"skill": "skill", "agent": "agent type"}
        listed = [f"{label[k]} '{name}'" for k in ("skill", "agent") for name in names(gone, k)]
        lines.append(f"No longer available: {', '.join(listed)}.")
    return f"{NOTE_HEAD}\n" + "\n".join(lines) if lines else ""
