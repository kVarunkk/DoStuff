"""`# note` and /memory: the AGENTS.md files the TUI writes to (read back by load_project_instructions)."""
from pathlib import Path
from typing import Literal

from dostuff.helpers.agent.load_project_instructions import INSTRUCTION_FILES

MemoryScope = Literal["project", "global"]


def memory_file(scope: MemoryScope, cwd: Path | None = None, home: Path | None = None) -> Path:
    """The project's (cwd) or the global (~/.dostuff) instruction file: an existing AGENTS.md or
    CLAUDE.md (the one load_project_instructions reads), else AGENTS.md."""
    folder = (home or Path.home()) / ".dostuff" if scope == "global" else (cwd or Path.cwd())
    return next((folder / name for name in INSTRUCTION_FILES if (folder / name).is_file()), folder / INSTRUCTION_FILES[0])


def quick_memory_text(text: str) -> str | None:
    """The note in a `# note` prompt (one line), else None. `#` alone or a multi-line text (e.g. a
    pasted Markdown document) isn't one."""
    stripped = text.strip()
    if not stripped.startswith("#") or "\n" in stripped:
        return None
    note = stripped.lstrip("#").strip()
    return note or None


def append_memory(path: Path, note: str) -> None:
    """Add `- note` as a new line at the end of the file (created with its folder if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    sep = "" if not existing or existing.endswith("\n") else "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{sep}- {note}\n")
