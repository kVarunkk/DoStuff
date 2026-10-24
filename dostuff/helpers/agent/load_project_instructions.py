from pathlib import Path

INSTRUCTION_FILES = ("AGENTS.md", "CLAUDE.md")  # first one found in a directory wins


def instruction_files(cwd: Path | None = None, home: Path | None = None) -> list[Path]:
    """Instruction files that apply in `cwd`, most general first.

    The global one (~/.dostuff/AGENTS.md or CLAUDE.md), then one per directory from the
    outermost parent of `cwd` down to `cwd` itself, so the closest file is read last and
    takes precedence where they disagree.
    """
    cwd = (cwd or Path.cwd()).resolve()
    home = home or Path.home()
    found: list[Path] = []
    for directory in [home / ".dostuff"] + list(reversed(cwd.parents)) + [cwd]:
        for name in INSTRUCTION_FILES:
            path = directory / name
            if path.is_file():
                if path.resolve() not in {p.resolve() for p in found}:
                    found.append(path)
                break
    return found


def load_project_instructions(cwd: Path | None = None, home: Path | None = None) -> str:
    """Every applicable AGENTS.md / CLAUDE.md, each under a heading with its path.

    With a single file its content is returned as-is.
    """
    sections = [(path, _read_md(path)) for path in instruction_files(cwd, home)]
    sections = [(path, text) for path, text in sections if text]
    if len(sections) == 1:
        return sections[0][1]
    return "\n\n".join(f"### From {path}\n{text}" for path, text in sections)


def _read_md(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
