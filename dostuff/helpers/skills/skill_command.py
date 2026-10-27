"""Skills and custom commands as slash commands: `/<name> [args]` runs a turn with the skill's
SKILL.md, or the command's prompt template (helpers/skills/custom_commands.py), in it."""
import re
from pathlib import Path
from typing import Any

from dostuff.helpers.agent.constants import COMMAND_DESCRIPTIONS
from dostuff.helpers.skills.custom_commands import command_prompt

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n?", re.DOTALL)
_NAME_RE = re.compile(r"^[\w.-]+$")


def skill_commands(skills: list[dict[str, Any]], custom: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """"/name" → custom command or skill. Built-in commands win a clash, then custom commands
    (project before user), then skills in the order found; names that can't be typed as one
    word are left out."""
    commands: dict[str, dict[str, Any]] = {}
    for skill in [*(custom or []), *skills]:
        name = str(skill.get("name", "")).strip().lower()
        command = "/" + name
        if not _NAME_RE.match(name) or command in COMMAND_DESCRIPTIONS or command in commands:
            continue
        commands[command] = skill
    return commands


def parse_skill_command(text: str, commands: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str] | None:
    """(skill, args) when `text` starts with a skill's command, else None."""
    parts = text.strip().split(maxsplit=1)
    skill = commands.get(parts[0].lower()) if parts and parts[0].startswith("/") else None
    return (skill, parts[1] if len(parts) > 1 else "") if skill else None


def skill_prompt(skill: dict[str, Any], args: str) -> str:
    """The user message for a skill run: its SKILL.md body (frontmatter dropped), then the args.
    For a custom command, its template with $ARGUMENTS filled in. Raises OSError if the file
    can't be read."""
    if skill.get("kind") == "command":
        return command_prompt(skill, args)
    location = Path(skill["location"])
    body = _FRONTMATTER_RE.sub("", location.read_text(encoding="utf-8"), count=1).strip()
    return (
        f"The user ran the /{skill['name']} skill. Follow its instructions below. Files it refers to "
        f"are relative to {location.parent}.\n"
        f'<skill name="{skill["name"]}" path="{location}">\n{body}\n</skill>\n\n'
        + (f"Arguments: {args}" if args else "(no arguments)")
    )
