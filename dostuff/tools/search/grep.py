import fnmatch
import re
from pathlib import Path

from dostuff.helpers.tools.resolve_safe_path import resolve_safe_path
from dostuff.tools.search.walk import display_path, iter_files

_MAX_FILE_BYTES = 2_000_000
_MAX_LINE_CHARS = 300


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(8192)
    except OSError:
        return True


def grep(pattern: str, path: str = ".", glob: str | None = None, ignore_case: bool = False, max_results: int = 200) -> str:
    """Search file contents for a regular expression (Python `re` syntax).

    Skips .git, virtualenvs, node_modules, __pycache__, build and dist directories,
    binary files, and files the sandbox does not allow reading.

    Args:
        pattern: Regular expression to search for, e.g. 'def \\w+_tool' or 'TODO'.
        path: File or directory to search, relative to the project root. Defaults to ".".
        glob: Only search files whose name (or relative path) matches this pattern, e.g. '*.py'.
        ignore_case: Match case-insensitively.
        max_results: Maximum number of matching lines to return.

    Returns:
        Matching lines as 'path:line: text', one per line, or a message if nothing matched.
    """
    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        raise ValueError(f"Invalid regular expression {pattern!r}: {e}") from e
    project_root = Path.cwd().resolve()
    root = resolve_safe_path(path, for_read=True)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    limit = max(int(max_results), 1)

    matches: list[str] = []
    files_with_matches = 0
    truncated = False
    for file_path in iter_files(root):
        rel = display_path(file_path, project_root)
        if glob and not (fnmatch.fnmatch(file_path.name, glob) or fnmatch.fnmatch(rel, glob)):
            continue
        try:
            if file_path.stat().st_size > _MAX_FILE_BYTES or _is_binary(file_path):
                continue
            with open(file_path, encoding="utf-8", errors="replace") as f:
                found = False
                for number, line in enumerate(f, start=1):
                    if regex.search(line):
                        found = True
                        text = line.rstrip("\r\n")
                        if len(text) > _MAX_LINE_CHARS:
                            text = text[:_MAX_LINE_CHARS] + " …"
                        matches.append(f"{rel}:{number}: {text}")
                        if len(matches) >= limit:
                            truncated = True
                            break
                files_with_matches += found
        except OSError:
            continue
        if truncated:
            break

    if not matches:
        return f"No matches for {pattern!r} in {path}."
    summary = f"[{len(matches)} match(es) in {files_with_matches} file(s)"
    summary += f"; stopped at max_results={limit}]" if truncated else "]"
    return "\n".join(matches + [summary])
