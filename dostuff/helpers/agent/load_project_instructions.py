import os
from pathlib import Path


def load_project_instructions() -> str:
    """Load AGENTS.md / CLAUDE.md from project (Pi-style cascade).
    
    Searches CWD and parent directories for AGENTS.md (or CLAUDE.md fallback),
    then checks global ~/.dostuff/ as final fallback. Returns content or empty string.
    """
    # Cascade: walk up from CWD looking for AGENTS.md, then CLAUDE.md
    cwd = Path.cwd()
    for directory in [cwd] + list(cwd.parents):
        agents_path = directory / "AGENTS.md"
        if agents_path.exists():
            return _read_md(agents_path)
        claude_path = directory / "CLAUDE.md"
        if claude_path.exists():
            return _read_md(claude_path)
    
    # Global fallback in ~/.dostuff/
    global_agents = Path.home() / ".dostuff" / "AGENTS.md"
    if global_agents.exists():
        return _read_md(global_agents)
    global_claude = Path.home() / ".dostuff" / "CLAUDE.md"
    if global_claude.exists():
        return _read_md(global_claude)
    
    return ""


def _read_md(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
