from pathlib import Path

from dostuff.helpers.agent import attachments
from dostuff.helpers.agent.attachments import classify_attachment
from dostuff.helpers.tools.resolve_safe_path import resolve_safe_path

DEFAULT_READ_LIMIT = 2000
_MAX_LINE_CHARS = 2000


def read_file(path: str, offset: int = 1, limit: int = DEFAULT_READ_LIMIT) -> str:
    """Read a file, returning numbered lines (like `cat -n`).

    Images (png, jpeg, gif, webp) are shown to you in the next message when this model can view images; so are PDFs
    when it can read them, otherwise you get their text. Word, Excel and PowerPoint files (.docx, .xlsx, .pptx) are
    converted to text, and offset/limit apply to that text.

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

    kind = classify_attachment(target_path)
    if kind is not None:
        return _read_attachment(target_path, path.replace("\\", "/"), kind, offset, limit)
    legacy = attachments.legacy_office_reason(target_path)
    if legacy:
        raise ValueError(f"{path}: {legacy}")
    lines = target_path.read_text(encoding="utf-8", errors="replace").replace("\xa0", " ").splitlines()
    return number_lines(lines, offset, limit)


def _read_attachment(target: Path, shown: str, kind: str, offset: int, limit: int) -> str:
    """An image or a natively read PDF goes to the model in the follow-up step (REQ-5.2); everything else is
    converted to text (REQ-5.1)."""
    from dostuff.helpers.agent import tool_attachments
    from dostuff.lib import model as model_lib  # lazy: litellm
    if kind == "image":
        return tool_attachments.attach(target.read_bytes(), target.name, shown)
    if kind == "pdf" and model_lib.model_supports("pdf"):
        data = target.read_bytes()
        if attachments.pdf_problem(data) is None:
            return tool_attachments.attach(data, target.name, shown)
    from dostuff.helpers.agent.convert_document import convert_document  # lazy: the [docs] libraries
    return number_lines(convert_document(target, kind).splitlines(), offset, limit)


def number_lines(lines: list[str], offset: int = 1, limit: int = DEFAULT_READ_LIMIT) -> str:
    """`limit` lines from line `offset` on, prefixed with their numbers; a final note when only part is shown."""
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
