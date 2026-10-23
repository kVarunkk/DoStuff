from dostuff.helpers.tools.resolve_safe_path import resolve_safe_path

DEFAULT_READ_LIMIT = 2000
_MAX_LINE_CHARS = 2000


def read_file(path: str, offset: int = 1, limit: int = DEFAULT_READ_LIMIT) -> str:
    """Read a text file, returning numbered lines (like `cat -n`).

    Args:
        path: Path relative to the project root.
        offset: 1-based line number to start reading from.
        limit: Maximum number of lines to return.

    Returns:
        The requested lines prefixed with their line numbers. When only part of the
        file is shown, a final note gives the total line count and the next offset.
    """
    target_path = resolve_safe_path(path, for_read=True)

    if not target_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not target_path.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {path}")

    lines = target_path.read_text(encoding="utf-8", errors="replace").replace("\xa0", " ").splitlines()
    total = len(lines)
    if total == 0:
        return "(empty file)"
    start = max(int(offset), 1)
    count = max(int(limit), 1)
    if start > total:
        return f"(offset {start} is past the end of the file; it has {total} lines)"
    end = min(start - 1 + count, total)
    width = len(str(end))
    out = []
    for number in range(start, end + 1):
        line = lines[number - 1]
        if len(line) > _MAX_LINE_CHARS:
            line = line[:_MAX_LINE_CHARS] + " [line truncated]"
        out.append(f"{number:>{width}}\t{line}")
    if start > 1 or end < total:
        note = f"[showing lines {start}-{end} of {total}"
        out.append(note + (f"; continue with offset={end + 1}]" if end < total else "]"))
    return "\n".join(out)
