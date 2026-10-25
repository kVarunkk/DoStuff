"""Inline ghost text for slash commands in the TUI input (Tab accepts it)."""

from collections.abc import Iterable

from dostuff.helpers.agent.constants import COMMAND_DESCRIPTIONS, COMMAND_FLAGS


def slash_suggestion(text: str, highlighted: str | None = None, commands: Iterable[str] = COMMAND_DESCRIPTIONS) -> str:
    """What to show after the cursor while a slash command is typed, or "".

    While the name is typed: the rest of `highlighted` (the "/" menu's pick), else of the first
    of `commands` it starts (the built-ins, plus skill commands in the TUI). After the name and a
    space: the command's flag (e.g. /model --active), completing a partly typed one, unless it's
    already there.
    """
    if not text.startswith("/") or "\n" in text:
        return ""
    name, space, arg = text.partition(" ")
    name = name.lower()
    if not space:
        match = highlighted if highlighted and highlighted.startswith(name) else next(
            (c for c in commands if c.startswith(name)), None)
        return match[len(name):] if match else ""
    flag = COMMAND_FLAGS.get(name)
    if not flag or flag in arg.split():
        return ""
    last = arg.rsplit(" ", 1)[-1]
    if not last:
        return flag
    return flag[len(last):] if last.startswith("-") and flag.startswith(last) else ""
