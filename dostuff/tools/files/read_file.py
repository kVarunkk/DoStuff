from pathlib import Path

def read_file(path: str) -> str:
    """Read a file.

    Args:
        path: Path relative to current directory or absolute.

    Returns:
        The file contents decoded as UTF-8.
    """
    target_path = Path(path).resolve()

    if not target_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not target_path.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {path}")

    content = target_path.read_text(encoding="utf-8", errors="replace")
    return content.replace("\xa0", " ")