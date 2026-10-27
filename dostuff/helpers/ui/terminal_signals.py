"""Escape sequences the TUI writes to the terminal itself: run status in the tab/window title
(OSC 0) and turn notifications (bell / OSC 9). Pure functions; the app writes them through its driver.
"""
from __future__ import annotations

import re
from typing import Literal

TitleState = Literal["idle", "running", "waiting", "done"]
TITLE_MARKS: dict[str, str] = {"idle": "", "running": "⏳", "waiting": "✋", "done": "✓"}
# XTWINOPS: save / restore the terminal's own title (xterm, VTE, kitty, WezTerm …; others ignore them)
PUSH_TITLE = "\x1b[22;0t"
POP_TITLE = "\x1b[23;0t"
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _clean(text: str) -> str:
    """No control characters: a BEL or ESC in a title or message would end the sequence early."""
    return _CONTROL_CHARS.sub(" ", text).strip()


def format_title(state: TitleState, label: str) -> str:
    """`⏳ dostuff · my-project`; no mark while idle."""
    mark = TITLE_MARKS.get(state, "")
    text = f"dostuff · {label}" if label else "dostuff"
    return f"{mark} {text}" if mark else text


def title_sequence(title: str) -> str:
    return f"\x1b]0;{_clean(title)}\x07"


def restore_title_sequence() -> str:
    """Leave the title as it was: an empty title (most terminals fall back to their default),
    then pop the title saved by PUSH_TITLE where that's supported."""
    return title_sequence("") + POP_TITLE


def notification_sequence(kind: str, message: str) -> str:
    """`bell` → BEL; `osc9` → OSC 9 desktop notification; anything else → nothing."""
    if kind == "bell":
        return "\x07"
    if kind == "osc9":
        return f"\x1b]9;{_clean(message)[:200]}\x07"
    return ""


def should_notify(kind: str, elapsed: float, after_seconds: float, focused: bool) -> bool:
    """Notify only when it's on, and the terminal is in the background or the wait was long."""
    return kind in ("bell", "osc9") and (not focused or elapsed >= after_seconds)
