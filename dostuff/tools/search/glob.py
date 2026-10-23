import fnmatch
from pathlib import Path

from dostuff.helpers.tools.resolve_safe_path import resolve_safe_path
from dostuff.tools.search.walk import display_path, iter_files

_MAX_RESULTS = 500


def glob(pattern: str, path: str = ".") -> str:
    """Find files by name pattern, newest first.

    Skips .git, virtualenvs, node_modules, __pycache__, build and dist directories.

    Args:
        pattern: Glob pattern matched against the path relative to `path`, e.g. '**/*.py',
            'src/*.ts' or 'README*'. A pattern without '/' matches file names at any depth.
        path: Directory to search, relative to the project root. Defaults to ".".

    Returns:
        Matching file paths relative to the project root, one per line.
    """
    project_root = Path.cwd().resolve()
    root = resolve_safe_path(path, for_read=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    pattern = pattern.replace("\\", "/")
    # fnmatch's '*' already crosses '/', so '**/' only needs to also allow zero directories
    alternatives = {pattern, pattern.replace("**/", "")} if "**/" in pattern else {pattern}

    found: list[tuple[float, str]] = []
    for file_path in iter_files(root):
        rel_to_root = file_path.relative_to(root).as_posix()
        target = rel_to_root if "/" in pattern else file_path.name
        if any(fnmatch.fnmatch(target, alt) for alt in alternatives):
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                continue
            found.append((mtime, display_path(file_path, project_root)))

    if not found:
        return f"No files matching {pattern!r} in {path}."
    found.sort(key=lambda item: item[0], reverse=True)
    lines = [rel for _, rel in found[:_MAX_RESULTS]]
    if len(found) > _MAX_RESULTS:
        lines.append(f"[{len(found) - _MAX_RESULTS} more not shown; narrow the pattern]")
    return "\n".join(lines)
