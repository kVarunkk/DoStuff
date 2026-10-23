"""Diffs for file-mutating tools: plain text for the model, coloured for the UI."""
import difflib

from dostuff.helpers.ui.emit import emit

_MAX_MODEL_DIFF_LINES = 200


def unified_diff(old: str, new: str, path: str) -> list[str]:
    return list(difflib.unified_diff(
        old.splitlines(), new.splitlines(), fromfile=path, tofile=path, lineterm="",
    ))


def diff_stats(diff_lines: list[str]) -> tuple[int, int]:
    added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    return added, removed


def plain_diff(diff_lines: list[str]) -> str:
    """The diff as returned to the model (no ANSI codes), capped in length."""
    if len(diff_lines) > _MAX_MODEL_DIFF_LINES:
        hidden = len(diff_lines) - _MAX_MODEL_DIFF_LINES
        diff_lines = diff_lines[:_MAX_MODEL_DIFF_LINES] + [f"... {hidden} more diff lines"]
    return "\n".join(diff_lines)


def colored_diff(diff_lines: list[str]) -> str:
    """Line-numbered diff with ANSI backgrounds, for display only."""
    colored_lines = []
    old_line = new_line = 0
    for s in diff_lines:
        if s.startswith("@@"):
            parts = s.split()
            try:
                old_line = int(parts[1].split(",")[0].lstrip("-"))
                new_line = int(parts[2].split(",")[0].lstrip("+"))
            except (IndexError, ValueError):
                pass
            continue
        if s.startswith("---") or s.startswith("+++") or s.startswith("\\ No newline at end of file"):
            continue
        if s.startswith("-"):
            colored_lines.append(f"  {old_line:>3}    \033[48;5;52m{s}\033[0m")
            old_line += 1
        elif s.startswith("+"):
            colored_lines.append(f"  {new_line:>3}    \033[48;5;22m{s}\033[0m")
            new_line += 1
        else:
            colored_lines.append(f"  {new_line:>3}    {s}")
            old_line += 1
            new_line += 1
    return "\n".join(colored_lines)


def report_change(verb: str, path: str, old: str, new: str) -> str:
    """Emit the coloured diff to the UI and return the plain-text tool result for the model."""
    diff_lines = unified_diff(old, new, path)
    result = f"{verb}: {path}"
    if not diff_lines:
        return f"{result} (no changes)"
    added, removed = diff_stats(diff_lines)
    emit(f"{path} (+{added} / -{removed})\n{colored_diff(diff_lines)}", msg_type="diff")
    return f"{result} (+{added} / -{removed} lines)\n{plain_diff(diff_lines)}"
