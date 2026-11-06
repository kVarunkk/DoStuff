from dostuff.helpers.tools.diff import report_change
from dostuff.helpers.tools.resolve_safe_path import resolve_safe_path

CRLF = "\r\n"
LF = "\n"


class EditFailed(ValueError):
    """An edit that can't be applied. `brief` is the short form for the approval preview."""

    def __init__(self, message: str, brief: str):
        super().__init__(message)
        self.brief = brief


def apply_edit(content: str, old_str: str, new_str: str, replace_all: bool, path: str) -> tuple[str, int]:
    """One replacement on LF-normalised `content`: (updated, replacements made). Raises EditFailed."""
    if not old_str:
        raise EditFailed("old_str must not be empty; use write_file to create a file.", "old_str is empty")
    if old_str == new_str:
        raise EditFailed("old_str and new_str are identical; nothing to change.", "old_str and new_str are identical")
    old_str = old_str.replace(CRLF, LF)
    new_str = new_str.replace(CRLF, LF)

    count = content.count(old_str)
    if count == 0:
        raise EditFailed(f"old_str not found in {path}. Read the file and copy the text exactly.", "old_str not found")
    if count > 1 and not replace_all:
        raise EditFailed(
            f"old_str occurs {count} times in {path}. Add surrounding lines to make it unique, "
            "or set replace_all=true.",
            f"old_str occurs {count} times",
        )
    if replace_all:
        return content.replace(old_str, new_str), count
    return content.replace(old_str, new_str, 1), 1


def edit_file(path: str, old_str: str, new_str: str, replace_all: bool = False) -> str:
    """Replaces an exact string in an existing file. Prefer this over write_file
    for changing part of a file.

    Args:
        path: Path relative to the project root, e.g. 'src/app.py'.
        old_str: Exact text to replace, including whitespace and indentation. Must
            occur exactly once unless replace_all is true; include surrounding lines
            to make it unique.
        new_str: Replacement text.
        replace_all: Replace every occurrence of old_str instead of requiring exactly one.

    Returns:
        A confirmation with a unified diff of the change.
    """
    safe_path = resolve_safe_path(path)
    if not safe_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    # Match on LF-normalised text, then write back with the file's own line endings
    with open(safe_path, encoding="utf-8", newline="") as f:
        raw = f.read()
    crlf = CRLF in raw
    content = raw.replace(CRLF, LF)

    updated, count = apply_edit(content, old_str, new_str, replace_all, path)
    with open(safe_path, "w", encoding="utf-8", newline="") as f:
        f.write(updated.replace(LF, CRLF) if crlf else updated)
    return report_change(f"File edited ({count} replacement(s))", path, content, updated)
