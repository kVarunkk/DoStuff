"""System prompt for a turn, shared by the TUI and `dostuff -p` (headless)."""
import asyncio
from pathlib import Path
from typing import Any

from dostuff.config import get_config
from dostuff.helpers.agent.constants import SYSTEM_INSTRUCTIONS
from dostuff.helpers.agent.environment_context import build_environment_block
from dostuff.helpers.agent.load_identity import load_identity
from dostuff.helpers.agent.load_project_instructions import load_project_instructions
from dostuff.helpers.skills.discover_skills import discover_skills


def build_base_instructions() -> tuple[str, list[dict[str, Any]]]:
    """(identity + skills + project instructions, the skills found). Built once per session."""
    skills = discover_skills()
    skills_summary = "\n\n".join(
        f"- name: {s['name']}\n  description: {s['description']}\n  location: {s['location']}"
        for s in skills
    )
    instructions = SYSTEM_INSTRUCTIONS.format(
        dostuff_identity=load_identity(),
        skills_summary=skills_summary or "(none available)",
    )
    project_instructions = load_project_instructions()
    if project_instructions:
        instructions += "\n\n## Project Instructions\n" + project_instructions
    return instructions, skills


async def build_turn_instructions(base: str, user_text: str, memory_store: Any, episodic_store: Any, user_id: str) -> str:
    """Base instructions + the environment block + memories and past episodes relevant to user_text."""
    memories = await memory_store.query(user_id, query_text=user_text, top_k=5)
    memory_text = "\n".join(f"- {m['key']}: {m['value']}" for m in memories)

    episodes = await episodic_store.query(
        user_id=user_id,
        query_text=user_text,
        top_k=3,
        working_dir=str(Path.cwd().resolve()) if get_config().episodic_scope == "project" else None,
    )
    episodic_text = "\n".join(
        f"  • [{ep['event_type']}] {ep['anchor_event']}\n"
        f"    Summary: {ep['summary']}\n"
        f"    Date: {ep['created_at_iso']}"
        for ep in episodes
    )

    instructions = base + "\n\n" + await asyncio.to_thread(build_environment_block)
    if memory_text:
        instructions += f"\n\n<relevant_user_facts>\n{memory_text}\n</relevant_user_facts>"
    if episodic_text:
        instructions += f"\n\n<past_episodes>\n{episodic_text}\n</past_episodes>"
    return instructions
