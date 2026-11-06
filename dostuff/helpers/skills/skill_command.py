"""Skills and custom commands as slash commands: `/<name> [args]` runs a turn with the skill's
SKILL.md, or the command's prompt template (helpers/skills/custom_commands.py), in it."""
import re
from pathlib import Path
from typing import Any

from dostuff.helpers.agent.constants import COMMAND_DESCRIPTIONS
from dostuff.helpers.agent.file_mentions import MENTION_NOTE_HEAD
from dostuff.helpers.skills.custom_commands import command_prompt

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n?", re.DOTALL)
_NAME_RE = re.compile(r"^[\w.-]+$")
_MENTION_RE = re.compile(r"(?<=\s)/\S+")
_MENTION_END = ".,!?:;"
PARTIAL_SLASH_RE = re.compile(r"(?:^|\s)(/[\w.-]*)$")


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


def skill_mentions(text: str, commands: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The skills named mid-message in `text`, once each, in order of first appearance.

    Pass what the user typed, before @mentions and pastes expand: a `/deploy` inside an attached file must not
    permit a skill. A `/skill` token follows whitespace and may end in one sentence mark; custom commands never
    count, and a message that starts with a skill or custom command is that command's run, not scanned.
    """
    if parse_skill_command(text, commands):
        return []
    found: dict[str, dict[str, Any]] = {}
    for match in _MENTION_RE.finditer(text.lstrip()):
        token = match.group().lower()
        skill = commands.get(token)
        if skill is None and token[-1] in _MENTION_END:
            skill = commands.get(token[:-1])
        if skill is not None and skill.get("kind") != "command":
            found.setdefault(skill["name"], skill)
    return list(found.values())


def mention_note(skills: list[dict[str, Any]]) -> str:
    """The block telling the model what the user named. It carries each skill's description and dir because a
    manual-only skill is not in the system prompt, and the model has no other way to find it."""
    if not skills:
        return ""
    rows = "\n".join(f"- name: {s['name']}\n  description: {s['description']}\n  dir: {s['dir']}" for s in skills)
    return (
        f"{MENTION_NOTE_HEAD} That permits using them for this message only; use one only if the message asks for "
        f"it. Before using one, read the SKILL.md in its dir; the files it refers to are relative to that dir.\n"
        f"{rows}\n</skill-mentions>"
    )


def with_mention_note(content: str, skills: list[dict[str, Any]]) -> str:
    """`content` as the model reads it. With nothing mentioned it comes back unchanged. The note rides in the user
    message, never the system prompt, which stays byte-stable across turns."""
    return f"{content}\n\n{mention_note(skills)}" if skills else content


def mention_line(skills: list[dict[str, Any]]) -> str:
    """Tells the user the mention was recognised; a typo shows by this line's absence."""
    return f"Skill{'s' if len(skills) > 1 else ''} available for this message: " + ", ".join(s["name"] for s in skills)


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
