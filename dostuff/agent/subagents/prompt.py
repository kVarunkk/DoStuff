"""The subagent's system prompt (spec D13, subagent side)."""
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dostuff.helpers.tools.project_root import project_root
from dostuff.lib.worktree import Worktree

if TYPE_CHECKING:
    from dostuff.agent.subagents.definitions import AgentDef

SUBAGENT_BASE = """You are a subagent: the main agent started you to do one task and report back.

- You cannot see the main conversation. Everything you know about the task is in its first message.
- Use your tools to find facts; don't guess or make things up.
- When you are done, end with a clear final report as your answer: what you did, what you found (with file paths
  and line numbers where they help), and anything left open.
- Your report goes to the main agent, not to the user. Don't address the user or ask them questions.
"""


MAX_PRELOADED_SKILLS = 32
PRELOAD_HEAD = (
    "## Preloaded skills\n"
    "These skills are part of your role: follow them. You already have their text below, so don't read their SKILL.md again."
)


@dataclass(frozen=True)
class Preloaded:
    """The skills a typed agent's `skills:` put in its system prompt (subagent capabilities D1)."""
    text: str = ""  # the whole section, "" when nothing was preloaded
    names: tuple[str, ...] = ()  # the skills it holds
    skipped: tuple[str, ...] = ()  # "name (reason)" for each one left out, for the run's report


def load_preloaded(agent_def: "AgentDef") -> Preloaded:
    """Read the SKILL.md of each skill in `agent_def.skills`, in list order, until the limits: 32 skills and
    `agents.max_preload_chars` characters. A skill that is unknown, manual-only or unreadable is skipped, and so is
    everything after the first one that does not fit. Blocking (reads files); call it through asyncio.to_thread."""
    if not agent_def.skills:
        return Preloaded()
    from dostuff.agent.capabilities import current_skills
    from dostuff.config import get_config
    from dostuff.helpers.skills.skill_command import skill_block, skill_body
    budget = get_config().agents_max_preload_chars
    by_name = {s["name"]: s for s in current_skills()}
    entries: list[str] = []
    names: list[str] = []
    skipped: list[str] = []
    used = 0
    full = False
    for index, name in enumerate(agent_def.skills):
        if index >= MAX_PRELOADED_SKILLS:
            skipped.append(f"{name} (over the limit of {MAX_PRELOADED_SKILLS} skills)")
            continue
        if full:
            skipped.append(f"{name} (over budget)")
            continue
        skill = by_name.get(name)
        if skill is None:
            skipped.append(f"{name} (not found)")
            continue
        if skill.get("disable_model_invocation"):
            skipped.append(f"{name} (manual-only)")
            continue
        location = Path(skill["location"])
        try:
            body = skill_body(location)
        except (OSError, UnicodeDecodeError):
            skipped.append(f"{name} (unreadable)")
            continue
        entry = f"Files this skill refers to are relative to {location.parent}.\n{skill_block(name, location, body)}"
        if used + len(entry) > budget:
            full = True
            skipped.append(f"{name} (over budget)")
            continue
        used += len(entry)
        entries.append(entry)
        names.append(name)
    text = "\n\n".join([PRELOAD_HEAD, *entries]) if entries else ""
    return Preloaded(text, tuple(names), tuple(skipped))


def worktree_note(worktree: Worktree) -> str:
    return (
        f"You work in a git worktree at {worktree.path} (branch {worktree.branch}), a separate checkout of "
        "the project; use paths relative to it. Your changes stay there for the parent agent to review."
    )


def _sections(agent_def: "AgentDef", worktree: Worktree | None,
              preloaded: Preloaded | None = None) -> tuple[str, tuple[str, ...]]:
    """Blocking (reads files, runs git); see build_subagent_prompt."""
    from dostuff.agent.capabilities import SKILL_KINDS, capability_set, discover_for_prompt
    from dostuff.agent.instructions import project_instructions_section, skills_summary
    from dostuff.helpers.agent.environment_context import build_environment_block

    parts = [SUBAGENT_BASE.strip()]
    if agent_def.prompt:
        parts.append(agent_def.prompt)
    if preloaded is not None and preloaded.text:
        parts.append(preloaded.text)
    if worktree is not None:
        parts.append(worktree_note(worktree))
    parts.append(build_environment_block(worktree.path if worktree is not None else project_root()))
    project = project_instructions_section(project_root())
    if project:
        parts.append(project)
    skills = discover_for_prompt()
    held = set(preloaded.names) if preloaded is not None else set()
    parts.append("Available skills (before using one, read the SKILL.md in its dir; the files it refers to are "
                 "relative to that dir):\n" + skills_summary([s for s in skills if s["name"] not in held]))
    # Skills only: the prompt lists no agent types, and a subagent can't start one (mid-run refresh D10). The
    # capability set covers every skill, preloaded ones included, so a mid-run refresh doesn't call them new.
    return "\n\n".join(parts), capability_set(skills, {}, SKILL_KINDS)


async def build_subagent_prompt(agent_def: "AgentDef", worktree: Worktree | None = None,
                                preloaded: Preloaded | None = None) -> tuple[str, tuple[str, ...]]:
    """(base rules, the type's body, its preloaded skills, the worktree note, the environment, project instructions
    and the other skills; the capability set its skills list claims, which the run's loop announces later changes
    against).

    No episode recall: a subagent has nothing of the user's conversation to recall against. Saved facts arrive as a
    facts step in its history instead (memory recall D14), unless its type sets omit_memory. Built once per run.
    """
    return await asyncio.to_thread(_sections, agent_def, worktree, preloaded)
