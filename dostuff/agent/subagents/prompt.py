"""The subagent's system prompt (spec D13, subagent side)."""
import asyncio
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


def worktree_note(worktree: Worktree) -> str:
    return (
        f"You work in a git worktree at {worktree.path} (branch {worktree.branch}), a separate checkout of "
        "the project; use paths relative to it. Your changes stay there for the parent agent to review."
    )


def _sections(agent_def: "AgentDef", worktree: Worktree | None) -> tuple[str, tuple[str, ...]]:
    """Blocking (reads files, runs git); see build_subagent_prompt."""
    from dostuff.agent.capabilities import SKILL_KINDS, capability_set, discover_for_prompt
    from dostuff.agent.instructions import project_instructions_section, skills_summary
    from dostuff.helpers.agent.environment_context import build_environment_block

    parts = [SUBAGENT_BASE.strip()]
    if agent_def.prompt:
        parts.append(agent_def.prompt)
    if worktree is not None:
        parts.append(worktree_note(worktree))
    parts.append(build_environment_block(worktree.path if worktree is not None else project_root()))
    project = project_instructions_section(project_root())
    if project:
        parts.append(project)
    skills = discover_for_prompt()
    parts.append("Available skills (before using one, read the SKILL.md in its dir; the files it refers to are "
                 "relative to that dir):\n" + skills_summary(skills))
    # Skills only: the prompt lists no agent types, and a subagent can't start one (mid-run refresh D10)
    return "\n\n".join(parts), capability_set(skills, {}, SKILL_KINDS)


async def build_subagent_prompt(agent_def: "AgentDef", worktree: Worktree | None = None) -> tuple[str, tuple[str, ...]]:
    """(base rules, the type's body, the worktree note, the environment, project instructions and skills; the
    capability set its skills list claims, which the run's loop announces later changes against).

    No episode recall: a subagent has nothing of the user's conversation to recall against. Saved facts arrive as a
    facts step in its history instead (memory recall D14), unless its type sets omit_memory. Built once per run.
    """
    return await asyncio.to_thread(_sections, agent_def, worktree)
