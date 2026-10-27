"""Custom slash commands: `.dostuff/commands/<name>.md` (project) and `~/.dostuff/commands/<name>.md` (user).

The file is a prompt template: `/<name> args` sends its body (frontmatter dropped) with
`$ARGUMENTS` replaced by the args. Optional frontmatter: `description:` for the "/" menu.
"""
import re
from pathlib import Path
from typing import Any

PLACEHOLDER = "$ARGUMENTS"
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_DESCRIPTION_RE = re.compile(r"^description:\s*(.+)$", re.MULTILINE)


def command_dirs(cwd: Path | None = None, home: Path | None = None) -> list[tuple[str, Path]]:
    """(scope, folder), project first so a project command wins over a user one with the same name."""
    return [("project", (cwd or Path.cwd()) / ".dostuff" / "commands"),
            ("user", (home or Path.home()) / ".dostuff" / "commands")]


def _description(text: str) -> str:
    """The frontmatter's description, else the body's first non-empty line."""
    match = _FRONTMATTER_RE.match(text)
    if match and (desc := _DESCRIPTION_RE.search(match.group(1))):
        return desc.group(1).strip().strip("\"'")
    body = _FRONTMATTER_RE.sub("", text, count=1)
    return next((line.strip().lstrip("#").strip() for line in body.splitlines() if line.strip()), "")


def discover_commands(cwd: Path | None = None, home: Path | None = None) -> list[dict[str, Any]]:
    """Every `*.md` in the command folders as {name, description, location, scope, kind: "command"}."""
    commands: list[dict[str, Any]] = []
    for scope, folder in command_dirs(cwd, home):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            commands.append({"name": path.stem, "description": _description(text), "location": str(path),
                             "scope": scope, "kind": "command"})
    return commands


def command_prompt(command: dict[str, Any], args: str) -> str:
    """The user message for `/<name> args`. Raises OSError if the file can't be read."""
    body = _FRONTMATTER_RE.sub("", Path(command["location"]).read_text(encoding="utf-8"), count=1).strip()
    if PLACEHOLDER in body:
        return body.replace(PLACEHOLDER, args)
    return f"{body}\n\nArguments: {args}" if args else body
