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


def is_denied_for_read(path: Path) -> bool:
    """Any `.env*` file, even in a skill dir, and `~/.dostuff/**` except a discovered skill dir below
    `~/.dostuff/skills`: file tools never read these."""
    if path.name.startswith(".env"):
        return True
    dostuff_home = (Path.home() / ".dostuff").resolve()
    if not _is_within(path, dostuff_home):
        return False
    skills_home = dostuff_home / "skills"
    return not any(skills_home in root.parents and _is_within(path, root) for root in _skill_roots)


def resolve_safe_path(path: str, root: str | Path | None = None, *, for_read: bool = False) -> Path:
    """Resolve `path` inside the project root (the session working dir: the cwd, or a subagent's worktree).

    Relative paths are joined to the root; absolute or `..` paths that escape it are
    rejected — never silently re-rooted. With `for_read=True`, a discovered skill dir is
    readable too, and secrets and harness data are rejected.
    """
    base_root = Path(root if root is not None else project_root()).resolve()
    target = Path(path.strip()).expanduser()
    if not target.is_absolute():
        target = base_root / target
    target = target.resolve()

    in_skill_dir = for_read and any(_is_within(target, root) for root in _skill_roots)
    if not _is_within(target, base_root) and not in_skill_dir:
        raise PathOutsideProject(
            f"Access denied: path '{path}' resolves outside the project root ({base_root})."
        )
    if for_read and is_denied_for_read(target):
        raise PathAccessDenied(f"Access denied: '{path}' may contain secrets and cannot be read by file tools.")
    return target
