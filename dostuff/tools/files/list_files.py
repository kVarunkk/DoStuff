from pathlib import Path

def list_files(directory: str = ".") -> str:
    """Lists files and subdirectories inside a given directory.

    Use this before reading or writing files to see what already exists, or to check
    whether a specific file is present before creating/overwriting it.

    Args:
        directory: Path relative to current directory or absolute. Defaults to ".".

    Returns:
        A newline-separated list of entries, with directories marked by a trailing '/'.
        Returns "(empty directory)" if there are no entries.

    Raises:
        NotADirectoryError: If the given path exists but is not a directory.
        FileNotFoundError: If the given directory does not exist.
    """
    safe_path = Path(directory).resolve()

    if not safe_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not safe_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    entries = sorted(safe_path.iterdir(), key=lambda p: p.name)
    if not entries:
        return "(empty directory)"

    lines = [f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries]
    return "\n".join(lines)