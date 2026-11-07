from pathlib import Path
from typing import Iterable

from dostuff.helpers.tools.project_root import project_root
from dostuff.lib.exceptions import PathAccessDenied, PathOutsideProject

# Discovered skill dirs: file tools may read these, and only these, outside the project root. Replaced, never
# merged, on each discover_skills() run, so a removed skill stops being readable.
_skill_roots: frozenset[Path] = frozenset()


def skill_roots() -> frozenset[Path]:
    return _skill_roots


def set_skill_roots(paths: Iterable[str | Path]) -> None:
    global _skill_roots
    _skill_roots = frozenset(Path(p).resolve() for p in paths)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def dostuff_home() -> Path:
    """`~/.dostuff`: readable like the project's `.dostuff`, so the agent can inspect what it is told about."""
    return (Path.home() / ".dostuff").resolve()


def is_denied_for_read(path: Path) -> bool:
    """Any `.env*` file, anything in a `.dostuff/data` dir (past sessions and prompts, from every project, and
    saved OAuth client secrets), and an MCP config (`mcp_config.json` in a `.dostuff` dir, or the legacy one at
    the project root): its env / header values are tokens, which `read_settings` redacts."""
    parts = [part.lower() for part in path.parts]
    if path.name.startswith(".env") or (".dostuff", "data") in zip(parts, parts[1:]):
        return True
    if path.name.lower() != "mcp_config.json":
        return False
    in_dostuff = any(part.lower() == ".dostuff" for part in path.parts)
    return in_dostuff or path.parent == project_root().resolve()  # mcp_config_sources()'s legacy location


def resolve_safe_path(path: str, root: str | Path | None = None, *, for_read: bool = False) -> Path:
    """Resolve `path` inside the project root (the session working dir: the cwd, or a subagent's worktree).

    Relative paths are joined to the root; absolute or `..` paths that escape it are
    rejected — never silently re-rooted. With `for_read=True`, a discovered skill dir and
    `~/.dostuff` are readable too, and secrets and MCP tokens are rejected.
    """
    base_root = Path(root if root is not None else project_root()).resolve()
    target = Path(path.strip()).expanduser()
    if not target.is_absolute():
        target = base_root / target
    target = target.resolve()

    readable_outside = False
    if for_read:
        readable_outside = _is_within(target, dostuff_home()) or any(
            _is_within(target, skill_root) for skill_root in _skill_roots
        )
    if not _is_within(target, base_root) and not readable_outside:
        raise PathOutsideProject(
            f"Access denied: path '{path}' resolves outside the project root ({base_root})."
        )
    if for_read and is_denied_for_read(target):
        raise PathAccessDenied(
            f"Access denied: '{path}' is a .env file, an MCP config or dostuff's own data, which may hold secrets "
            "and cannot be read by file tools."
        )
    return target
