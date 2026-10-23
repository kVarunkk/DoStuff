from dostuff.helpers.tools.diff import report_change
from dostuff.helpers.tools.resolve_safe_path import resolve_safe_path


def write_file(path: str, content: str) -> str:
    """Writes content to a file inside the project directory. Creates the file
    if it does not exist, otherwise replaces its contents. Prefer edit_file for
    changing part of an existing file.

    Args:
        path: Path relative to the project root, e.g. './blog_post.md'
            or 'skills/my-skill/SKILL.md'.
        content: The text content to write to the file.

    Returns:
        A confirmation with a unified diff of the change.
    """
    safe_path = resolve_safe_path(path)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    previous_content = safe_path.read_text(encoding="utf-8", errors="replace") if safe_path.is_file() else ""
    with open(safe_path, "w", encoding="utf-8", newline="") as f:  # write exactly what was given
        f.write(content)
    return report_change("File written", path, previous_content, content)
