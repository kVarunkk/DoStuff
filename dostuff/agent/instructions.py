"""The system prompt and the per-turn context, shared by the TUI and `dostuff -p` (headless)."""
import asyncio
from pathlib import Path
from typing import Any

from dostuff.config import get_config
from dostuff.helpers.agent.constants import (
    AGENT_BACKGROUND, BACKGROUND_WAIT_HEADLESS, BACKGROUND_WAIT_INTERACTIVE, MEMORY_INSTRUCTIONS, SYSTEM_INSTRUCTIONS,
)
from dostuff.helpers.agent.environment_context import build_environment_block
from dostuff.helpers.agent.load_identity import load_identity
from dostuff.helpers.agent.load_project_instructions import load_project_instructions
from dostuff.helpers.agent.turn_context import compose_turn_context
from dostuff.helpers.skills.discover_skills import discover_skills
from dostuff.helpers.ui.emit import emit
from dostuff.lib.memory import chroma


def _background_wait(interactive: bool) -> str:
    """How to wait for a background job: the TUI resumes the agent when one ends (unless
    jobs.wake_on_end is off, then nothing is said); headless kills them at the end of the run."""
    if not interactive:
        return BACKGROUND_WAIT_HEADLESS
    return BACKGROUND_WAIT_INTERACTIVE if get_config().jobs_wake_on_end else ""


def skills_summary(skills: list[dict[str, Any]]) -> str:
    """The skills list as the system prompt shows it."""
    summary = "\n\n".join(
        f"- name: {s['name']}\n  description: {s['description']}\n  location: {s['location']}"
        for s in skills
    )
    return summary or "(none available)"


def project_instructions_section(cwd: Path | None = None) -> str:
    """`## Project Instructions` + every applicable AGENTS.md / CLAUDE.md, or "" when there are none."""
    project_instructions = load_project_instructions(cwd)
    return "## Project Instructions\n" + project_instructions if project_instructions else ""


def agent_types_list() -> str:
    """One `  - name: description` line per agent type, fork included (subagents spec D11, D13)."""
    from dostuff.agent.subagents.definitions import current_definitions
    from dostuff.agent.subagents.fork import FORK_DESCRIPTION, FORK_TYPE
    descriptions = {name: d.description for name, d in current_definitions().types.items()}
    descriptions[FORK_TYPE] = FORK_DESCRIPTION
    return "\n".join(f"  - {name}: {descriptions[name]}" for name in sorted(descriptions))


def build_base_instructions(interactive: bool = False) -> tuple[str, list[dict[str, Any]]]:
    """(identity + skills + project instructions, the skills found). Built once per session.

    `interactive`: the TUI, which wakes the agent when a background job ends; headless doesn't.
    """
    skills = discover_skills()
    instructions = SYSTEM_INSTRUCTIONS.format(
        dostuff_identity=load_identity(),
        skills_summary=skills_summary(skills),
        memory_section=MEMORY_INSTRUCTIONS if chroma.chroma_available() else "",
        background_wait=_background_wait(interactive),
        agent_types=agent_types_list(),
        agent_background=AGENT_BACKGROUND if interactive or get_config().agents_wake_on_end else "",
    )
    project = project_instructions_section()
    if project:
        instructions += "\n\n" + project
    return instructions, skills


def build_system_instructions(base: str) -> str:
    """The system prompt of every turn: `base`, byte for byte. Anything that changes per turn goes in
    `build_turn_context` instead: one differing byte here invalidates the provider's whole cached prefix."""
    return base


async def build_turn_context(user_text: str, *, memory_store: Any, episodic_store: Any, user_id: str,
                             prompt_context: str = "") -> str:
    """The turn's hidden context step: the environment block, memories and past episodes relevant to user_text,
    and UserPromptSubmit context.

    The stores are None when long-term memory is off (chromadb not installed): no recall then.
    """
    # Recall is extra context: if it fails (e.g. the embedding model could not be downloaded),
    # say so and run the turn without it
    memories: list[dict] = []
    episodes: list[dict] = []
    if memory_store is not None and episodic_store is not None:
        # Not at module level: the store module pulls in litellm, which the TUI loads after its first frame
        from dostuff.lib.memory.semantic_memory_store import MIN_FACT_SIMILARITY
        try:
            memories = await memory_store.query(user_id, query_text=user_text, top_k=5, min_similarity=MIN_FACT_SIMILARITY)
            episodes = await episodic_store.query(
                user_id=user_id,
                query_text=user_text,
                top_k=3,
                working_dir=str(Path.cwd().resolve()) if get_config().episodic_scope == "project" else None,
            )
        except Exception as e:
            emit(f"Memory recall failed, continuing without it: {e}", "error")
            memories, episodes = [], []
    memory_text = "\n".join(f"- {m['key']}: {m['value']}" for m in memories)
    episodic_text = "\n".join(
        f"  • [{ep['event_type']}] {ep['anchor_event']}\n"
        f"    Summary: {ep['summary']}\n"
        f"    Date: {ep['created_at_iso']}"
        for ep in episodes
    )

    environment = await asyncio.to_thread(build_environment_block)
    return compose_turn_context(environment, memory_text, episodic_text, prompt_context)
