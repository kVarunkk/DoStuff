"""The `ask_user` answer grammar (spec D5): one formatter, one parser, shared by the tool and the TUI.

Success looks like:

    Asked the user 2 questions; they answered:
    1. Which database should I use?
       → Postgres
    2. Which auth methods?
       → JWT, OAuth

Import-light on purpose: this module is read on the TUI's paint path (CON-010 spirit), so it imports
nothing beyond the standard library, and the tool imports it, never the other way round.
"""
import re

# The one arrow that separates a question from its answer; the panel and the parser share it
ASK_ANSWER_ARROW = "   → "
ASK_QUESTIONS_LINE = "Asked the user {n} question{s}; they answered:"
# `<n>. ` opens a pair; nothing else does, so a question or answer holding `→` is safe
_NUMBERED = re.compile(r"^\d+\. ")

# The two non-answers (D5); the parser returns None for both, so they render as ordinary tool results
CANCELLED_ASK_RESULT = (
    "The user cancelled the questions; no answers.\n"
    "Use your best judgement and say what you assumed."
)
UNAVAILABLE_ASK_RESULT = (
    "No user is available in this run, so the questions were not shown.\n"
    "Ask nothing further; state your assumption and continue."
)


class NoUserAvailable(Exception):
    """No user can answer in this run (headless without a terminal, a subagent): the adapter's way to say
    "nobody to ask", distinct from `None` = cancelled (spec D7/D13)."""


def one_line(text: object) -> str:
    """Whitespace-collapsed to a single line: the grammar depends on it (D4, D5)."""
    return " ".join(str(text).split())


def pick_options(text: str, options: list[dict]) -> str | None:
    """`"2"` or `"1,3"` (or `"1 3"`) → the matching option labels joined in **option order** (D5), or None
    when the text is not a list of numbers (`"my own thing"` is a free answer, and so is `"9"` out of range).

    Shared by the TUI adapter and the headless one, so both read a number the same way (D13).
    """
    parts = [part for part in re.split(r"[,\s]+", str(text).strip()) if part]
    if not parts or not all(part.isdigit() for part in parts):
        return None
    indexes = {int(part) for part in parts}
    if any(not 1 <= i <= len(options) for i in indexes):
        return None
    labels = [one_line(option.get("label", "")) for i, option in enumerate(options, 1) if i in indexes]
    return ", ".join(label for label in labels if label) or None


def format_answers(pairs: list[tuple[str, str]]) -> str:
    """The tool's result text for `pairs` (question, answer) — D5, one pair per numbered block."""
    n = len(pairs)
    lines = [ASK_QUESTIONS_LINE.format(n=n, s="" if n == 1 else "s")]
    for i, (question, answer) in enumerate(pairs, 1):
        lines.append(f"{i}. {one_line(question)}")
        lines.append(f"{ASK_ANSWER_ARROW}{one_line(answer)}")
    return "\n".join(lines)


def parse_answers(text: str) -> list[tuple[str, str]] | None:
    """The (question, answer) pairs in `format_answers`' output, or None when `text` is not an answers block
    (a cancelled, refused or unavailable result, or any other tool's output), so it renders as a normal result."""
    lines = str(text).splitlines()
    if not lines or not re.fullmatch(r"Asked the user \d+ questions?; they answered:", lines[0].strip()):
        return None
    pairs: list[tuple[str, str]] = []
    question: str | None = None
    for line in lines[1:]:
        numbered = _NUMBERED.match(line)
        if numbered:
            if question is not None:  # a new pair opens; the previous one had no answer line
                return None
            question = line[numbered.end():]
        elif line.startswith(ASK_ANSWER_ARROW) and question is not None:
            pairs.append((question, line[len(ASK_ANSWER_ARROW):]))
            question = None
        elif line.strip():
            return None  # anything else in the middle: not our grammar
    if question is not None or not pairs:
        return None
    return pairs
