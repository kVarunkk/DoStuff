"""The `create_agent_type` tool: define a new subagent type as an agent file (subagents spec D2).

The file is a Markdown file with YAML frontmatter in `.dostuff/agents/` (project) or `~/.dostuff/agents/`
(user), the same format `/agents types` lists and the `agent` tool starts. A type is a system prompt plus a
tool grant, so this is a settings change: it asks in every mode but auto, and the prompt shows the whole file.
"""
import re
from pathlib import Path
from typing import Any, Literal

import yaml

from dostuff.agent.subagents.definitions import (BUILTIN_TYPES, RESERVED_TYPES, map_tools, parse_agent_file,
                                                  reload)

Scope = Literal["project", "user"]
NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


def agent_type_path(name: str, scope: str) -> Path:
    if scope == "project":
        from dostuff.helpers.tools.project_root import project_root
        return project_root() / ".dostuff" / "agents" / f"{name}.md"
    if scope == "user":
        return Path.home() / ".dostuff" / "agents" / f"{name}.md"
    raise ValueError("scope must be 'project' (./.dostuff/agents) or 'user' (~/.dostuff/agents)")


def _display(path: Path) -> str:
    return str(path).replace(str(Path.home()), "~")


def check_name(name: str) -> str:
    name = name.strip()
    if not NAME_RE.fullmatch(name):
        raise ValueError(f"name '{name}' must be lowercase letters, digits and hyphens (e.g. 'code-reviewer'), "
                         "at most 64 characters, starting with a letter or digit")
    if name in RESERVED_TYPES:
        raise ValueError(f"'{name}' is reserved by the harness; pick another name")
    return name


def agent_file_text(name: str, description: str, prompt: str, tools: list[str] | None = None,
                    model: str = "", max_iterations: int | None = None) -> tuple[str, list[str]]:
    """(the agent file's text, warnings). Raises ValueError for a type that couldn't load."""
    name = check_name(name)
    description = " ".join(description.split())
    if not description:
        raise ValueError("description must not be empty: the main agent picks a type by it")
    if not prompt.strip():
        raise ValueError("prompt (the type's system prompt) must not be empty")
    front: dict[str, Any] = {"name": name, "description": description}
    warnings: list[str] = []
    if tools:
        mapped, warnings = map_tools(tools, "tools")
        if not mapped:
            raise ValueError("none of the tools can be given to a subagent: " + "; ".join(warnings))
        front["tools"] = ", ".join(mapped)
    if model.strip() and model.strip() != "inherit":
        front["model"] = model.strip()
    if max_iterations is not None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        front["max_iterations"] = int(max_iterations)
    head = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=10_000)
    return f"---\n{head}---\n\n{prompt.strip()}\n", warnings


def create_agent_type_preview(args: dict[str, Any]) -> str:
    """What the approval prompt shows: where the file goes and its whole text."""
    try:
        name = check_name(str(args.get("name", "")))
        path = agent_type_path(name, str(args.get("scope") or "project"))
        text, warnings = agent_file_text(
            name, str(args.get("description", "")), str(args.get("prompt", "")), args.get("tools"),
            str(args.get("model") or ""), args.get("max_iterations"))
    except Exception as e:
        return f"  ({e}; it will fail)"
    verb = "replaces" if path.is_file() else "creates"
    lines = [f"  {verb} {_display(path)}"] + [f"  ⚠ {w}" for w in warnings] + [""]
    return "\n".join(lines + [f"  {line}".rstrip() for line in text.splitlines()])


def create_agent_type(
    name: str,
    description: str,
    prompt: str,
    tools: list[str] | None = None,
    model: str = "",
    max_iterations: int | None = None,
    scope: Scope = "project",
    replace: bool = False,
) -> str:
    """Defines a new subagent type (an agent file) that `agent` can start right away with
    subagent_type=<name>. Use it when the user asks for a new kind of agent (e.g. "make a code-reviewer
    agent") or a role is needed again and again; for a one-off task, just call `agent` with a built-in type.
    The user is always asked first.

    Args:
        name: The type's name, used as subagent_type: lowercase letters, digits and hyphens, e.g. "code-reviewer".
        description: One sentence on what it does and when to use it; it is how you pick the type later.
        prompt: The type's system prompt: its role, how it works and what its report should contain. It is
            added to the standard subagent rules; it never sees this conversation.
        tools: The tools it may use, e.g. ["read_file", "grep", "glob"]. Leave out for every tool a subagent
            may have. Claude Code names (Read, Edit, Bash...) work too; agent, todo_write, settings and memory
            tools are never given to subagents.
        model: An alias from agents.models or a full model name. Empty uses the session's model.
        max_iterations: Its tool-loop cap. Leave out for the configured agents.max_iterations.
        scope: 'project' for ./.dostuff/agents (this project), 'user' for ~/.dostuff/agents (every project).
        replace: True replaces an existing file of the same name in that scope.

    Returns:
        Where the file was written, its text, and how to start it.
    """
    name = check_name(name)
    path = agent_type_path(name, scope)
    existed = path.is_file()
    if existed and not replace:
        raise ValueError(f"{_display(path)} already exists. Pass replace=true to replace it, or pick another name.")
    text, warnings = agent_file_text(name, description, prompt, tools, model, max_iterations)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    agent_def, parse_warnings = parse_agent_file(path)
    if agent_def is None:  # can't happen for text built above; never leave a broken file behind
        path.unlink(missing_ok=True)
        raise ValueError("the agent file didn't load, so it was removed: " + "; ".join(parse_warnings))
    loaded = reload()
    notes = list(warnings)
    winner = loaded.types.get(name)
    if winner is not None and winner.source != str(path):
        notes.append(f"a '{name}' type in {winner.source} comes first in the search order, so `agent` starts "
                     "that one instead")
    elif name in BUILTIN_TYPES:
        notes.append(f"this replaces the built-in '{name}' type")
    head = f"{'Replaced' if existed else 'Created'} agent type '{name}' at {_display(path)}."
    usage = f'Start it with agent(subagent_type="{name}", description=..., prompt=...); /agents types lists it.'
    body = [head, usage] + [f"Warning: {n}" for n in notes] + ["", text.rstrip()]
    return "\n".join(body)
