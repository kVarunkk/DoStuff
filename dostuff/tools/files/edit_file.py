from dostuff.helpers.tools.diff import report_change
from dostuff.helpers.tools.resolve_safe_path import resolve_safe_path

CRLF = "\r\n"
LF = "\n"


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
    if not old_str:
        raise ValueError("old_str must not be empty; use write_file to create a file.")
    if old_str == new_str:
        raise ValueError("old_str and new_str are identical; nothing to change.")

    # Match on LF-normalised text, then write back with the file's own line endings
    with open(safe_path, encoding="utf-8", newline="") as f:
        raw = f.read()
    crlf = CRLF in raw
    content = raw.replace(CRLF, LF)
    old_str = old_str.replace(CRLF, LF)
    new_str = new_str.replace(CRLF, LF)

    count = content.count(old_str)
    if count == 0:
        raise ValueError(f"old_str not found in {path}. Read the file and copy the text exactly.")
    if count > 1 and not replace_all:
        raise ValueError(
            f"old_str occurs {count} times in {path}. Add surrounding lines to make it unique, "
            "or set replace_all=true."
        )

    updated = content.replace(old_str, new_str) if replace_all else content.replace(old_str, new_str, 1)
    with open(safe_path, "w", encoding="utf-8", newline="") as f:
        f.write(updated.replace(LF, CRLF) if crlf else updated)
    return report_change(f"File edited ({count if replace_all else 1} replacement(s))", path, content, updated)
