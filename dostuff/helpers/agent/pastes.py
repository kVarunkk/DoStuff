"""Large pastes in the TUI input: collapsed to a placeholder on screen, spilled to a file for the model.

A paste over `ui.paste_collapse_lines` shows as `[Pasted text #1 +243 lines]` in the input. On submit
the placeholder is swapped back for the text — inline when it's small, otherwise the text is written
to `.dostuff/pastes/` in the project and the model gets the path plus a short preview, and reads the
rest with read_file (offset/limit) or grep. Files older than `loop.paste_retention_days` are swept at
startup; with 0 they're deleted when the TUI exits instead.
"""
import re
import time
from pathlib import Path

from dostuff.helpers.tools.project_root import project_root

PASTE_DIR = Path(".dostuff") / "pastes"
PREVIEW_LINES = 20
_PREVIEW_LINE_CHARS = 500
COLLAPSE_CHARS = 2000  # a long single-line paste (minified JSON, ...) collapses too
PLACEHOLDER_RE = re.compile(r"\[Pasted text #(\d+) \+\d+ lines?\]")


def _line_count(text: str) -> int:
    return text.count("\n") + 1


def placeholder(number: int, text: str) -> str:
    lines = _line_count(text)
    return f"[Pasted text #{number} +{lines} line{'s' if lines != 1 else ''}]"


class PasteStore:
    """This process's collapsed pastes, by number; numbers keep counting across /clear."""

    def __init__(self) -> None:
        self._texts: dict[int, str] = {}
        self._files: dict[tuple[str, int], Path] = {}  # (session id, number) → file written for it

    def should_collapse(self, text: str, collapse_lines: int) -> bool:
        return collapse_lines > 0 and (_line_count(text) > collapse_lines or len(text) > COLLAPSE_CHARS)

    def add(self, text: str) -> str:
        """Keep `text` and return the placeholder to put in the input instead."""
        number = len(self._texts) + 1
        self._texts[number] = text
        return placeholder(number, text)

    def inline(self, text: str) -> str:
        """Every known placeholder swapped back for its text (shell commands, approval feedback)."""
        return PLACEHOLDER_RE.sub(lambda m: self._texts.get(int(m.group(1)), m.group(0)), text)

    def resolve(self, text: str, session_id: str, file_lines: int, file_chars: int) -> str:
        """Placeholders → text inline, or a file reference when over `file_lines` lines / `file_chars` chars.

        `file_lines` 0 keeps every paste inline. A placeholder this process doesn't know (recalled from
        an earlier run's history, or typed by hand) is left as it is.
        """
        def swap(match: re.Match[str]) -> str:
            number = int(match.group(1))
            body = self._texts.get(number)
            if body is None:
                return match.group(0)
            if file_lines <= 0 or (_line_count(body) <= file_lines and len(body) <= file_chars):
                return body
            return self._file_reference(number, body, session_id)
        return PLACEHOLDER_RE.sub(swap, text)

    def _file_reference(self, number: int, body: str, session_id: str) -> str:
        root = project_root()
        path = self._files.get((session_id, number))
        if path is None or not path.is_file():
            folder = root / PASTE_DIR
            folder.mkdir(parents=True, exist_ok=True)
            ignore = folder / ".gitignore"
            if not ignore.exists():
                ignore.write_text("*\n", encoding="utf-8")  # keep pastes out of git without touching the repo's .gitignore
            path = folder / f"{session_id[:8]}-{number}.txt"
            path.write_text(body, encoding="utf-8", newline="")
            self._files[(session_id, number)] = path
        try:
            shown = path.relative_to(root).as_posix()
        except ValueError:  # cwd changed since it was written (worktree)
            shown = path.as_posix()
        lines = body.splitlines()
        preview = "\n".join(
            line if len(line) <= _PREVIEW_LINE_CHARS else line[:_PREVIEW_LINE_CHARS] + " [line truncated]"
            for line in lines[:PREVIEW_LINES]
        )
        more = f"first {PREVIEW_LINES} of {len(lines)} lines" if len(lines) > PREVIEW_LINES else "all lines"
        return (
            f'<pasted_content file="{shown}" lines="{len(lines)}" chars="{len(body)}">\n'
            f"The user pasted {len(lines):,} lines. To save context they were saved to {shown} instead of "
            "being included here: read the parts you need with read_file (offset/limit) or search it with "
            f"grep, rather than reading the whole file at once. Preview ({more}):\n"
            f"{preview}\n"
            "</pasted_content>"
        )

    def delete_files(self) -> int:
        """Delete the files this process wrote; returns how many went."""
        removed = 0
        for path in self._files.values():
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        self._files.clear()
        return removed


def sweep_old_pastes(retention_days: float, root: Path | None = None) -> int:
    """Delete paste files older than `retention_days` (0 or less: none); returns how many went."""
    folder = (root or project_root()) / PASTE_DIR
    if retention_days <= 0 or not folder.is_dir():
        return 0
    cutoff = time.time() - retention_days * 86_400
    removed = 0
    for path in folder.glob("*.txt"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            pass
    return removed
