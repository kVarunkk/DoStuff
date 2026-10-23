"""Directory walking shared by the grep and glob tools: sandboxed, with default excludes."""
import fnmatch
import os
from pathlib import Path
from typing import Iterator

from dostuff.helpers.tools.resolve_safe_path import is_denied_for_read

# Directory names never descended into (gitignore-style defaults)
EXCLUDED_DIR_PATTERNS = (".git", ".hg", ".svn", "venv*", ".venv*", "node_modules", "__pycache__",
                         "build", "dist", ".mypy_cache", ".pytest_cache", ".tox", "*.egg-info")


def is_excluded_dir(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDED_DIR_PATTERNS)


def iter_files(root: Path) -> Iterator[Path]:
    """Every readable file under `root` (or `root` itself if it is a file), skipping excluded dirs."""
    if root.is_file():
        if not is_denied_for_read(root):
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not is_excluded_dir(d))
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if not is_denied_for_read(path):
                yield path


def display_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()
