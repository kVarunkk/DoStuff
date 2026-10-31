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


def _sections(agent_def: "AgentDef", worktree: Worktree | None) -> str:
    """Blocking (reads files, runs git); see build_subagent_prompt."""
    from dostuff.agent.instructions import project_instructions_section, skills_summary
    from dostuff.helpers.agent.environment_context import build_environment_block
    from dostuff.helpers.skills.discover_skills import discover_skills

    parts = [SUBAGENT_BASE.strip()]
    if agent_def.prompt:
        parts.append(agent_def.prompt)
    if worktree is not None:
        parts.append(worktree_note(worktree))
    parts.append(build_environment_block(worktree.path if worktree is not None else project_root()))
    project = project_instructions_section(project_root())
    if project:
        parts.append(project)
    parts.append("Available skills (read the full SKILL.md at the given path before using one):\n"
                 + skills_summary(discover_skills()))
    return "\n\n".join(parts)


async def build_subagent_prompt(agent_def: "AgentDef", worktree: Worktree | None = None) -> str:
    """Base rules, the type's body, the worktree note, the environment, project instructions and skills.

    No memory recall: a subagent has nothing of the user's conversation to recall against. Built once per run.
    """
    return await asyncio.to_thread(_sections, agent_def, worktree)
