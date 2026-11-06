from typing import Any

# pydantic on Python < 3.12 rejects typing.TypedDict when building tool schemas
from typing_extensions import Annotated, NotRequired, TypedDict

from pydantic import Field

from dostuff.helpers.tools.diff import report_change
from dostuff.helpers.tools.resolve_safe_path import resolve_safe_path
from dostuff.tools.files.edit_file import CRLF, LF, EditFailed, apply_edit

_SHAPE = "must be an object with old_str and new_str"


class EditOp(TypedDict):
    old_str: Annotated[str, Field(description="Exact text to replace, as the file reads after the edits before this one.")]
    new_str: Annotated[str, Field(description="Replacement text.")]
    replace_all: NotRequired[Annotated[bool, Field(description="Replace every occurrence instead of requiring exactly one.")]]


def check_edits(edits: Any) -> None:
    """Raise ValueError unless `edits` has the schema's shape: nothing validates a tool call's arguments before
    the tool runs, and a weak tool-caller can send a JSON string or an entry with a key missing."""
    if not isinstance(edits, list):
        raise ValueError(f"edits must be a list of objects with old_str and new_str, got {type(edits).__name__}.")
    if not edits:
        raise ValueError("edits must not be empty.")
    for position, edit in enumerate(edits, start=1):
        label = f"edit {position} of {len(edits)}"
        if not isinstance(edit, dict):
            raise ValueError(f"{label}: {_SHAPE}, got {type(edit).__name__}.")
        if not isinstance(edit.get("old_str"), str) or not isinstance(edit.get("new_str"), str):
            raise ValueError(f"{label}: {_SHAPE}.")


def apply_edits(content: str, edits: list[EditOp], path: str) -> tuple[str, int]:
    """`edits` applied in order to LF-normalised `content`: (updated, replacements made). Pure, so the approval
    preview runs the same code as the tool. Raises EditFailed naming the edit that could not be applied."""
    total = 0
    for position, edit in enumerate(edits, start=1):
        try:
            content, count = apply_edit(content, edit["old_str"], edit["new_str"], bool(edit.get("replace_all")), path)
        except EditFailed as e:
            label = f"edit {position} of {len(edits)}"
            raise EditFailed(f"{label}: {e}", f"{label}: {e.brief}") from e
        total += count
    return content, total


def multi_edit(path: str, edits: list[EditOp]) -> str:
    """Makes several exact-string replacements in one file, in one call. Prefer this over repeated
    edit_file calls when changing one file in several places. All or nothing: if any edit fails, the
    file is left untouched.

    Args:
        path: Path relative to the project root, e.g. 'src/app.py'.
        edits: The replacements, applied in order: each one sees the result of the ones before it.
            Each old_str must occur exactly once unless that edit sets replace_all.

    Returns:
        A confirmation with a unified diff of the whole change.
    """
    safe_path = resolve_safe_path(path)
    if not safe_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    check_edits(edits)

    # Match on LF-normalised text, then write back with the file's own line endings
    with open(safe_path, encoding="utf-8", newline="") as f:
        raw = f.read()
    crlf = CRLF in raw
    content = raw.replace(CRLF, LF)

    updated, count = apply_edits(content, edits, path)
    with open(safe_path, "w", encoding="utf-8", newline="") as f:
        f.write(updated.replace(LF, CRLF) if crlf else updated)
    return report_change(f"File edited ({len(edits)} edits, {count} replacement(s))", path, content, updated)
