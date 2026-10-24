from pathlib import Path
from dostuff.helpers.tools.project_root import project_root
from dostuff.lib.exceptions import PathAccessDenied, PathOutsideProject


def is_denied_for_read(path: Path) -> bool:
    """`~/.dostuff/**` and any `.env*` file are never readable by file tools."""
    dostuff_home = (Path.home() / ".dostuff").resolve()
    if path == dostuff_home or dostuff_home in path.parents:
        return True
    return path.name.startswith(".env")


def resolve_safe_path(path: str, root: str | Path | None = None, *, for_read: bool = False) -> Path:
    """Resolve `path` inside the project root (the session working dir: the cwd, or a subagent's worktree).

    Relative paths are joined to the root; absolute or `..` paths that escape it are
    rejected — never silently re-rooted. With `for_read=True`, secrets and harness data
    are rejected as well.
    """
    base_root = Path(root if root is not None else project_root()).resolve()
    target = Path(path.strip()).expanduser()
    if not target.is_absolute():
        target = base_root / target
    target = target.resolve()

    if target != base_root and base_root not in target.parents:
        raise PathOutsideProject(
            f"Access denied: path '{path}' resolves outside the project root ({base_root})."
        )
    if for_read and is_denied_for_read(target):
        raise PathAccessDenied(f"Access denied: '{path}' may contain secrets and cannot be read by file tools.")
    return target
