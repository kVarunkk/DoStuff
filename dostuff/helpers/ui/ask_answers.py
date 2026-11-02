"""The `ask_user` answer grammar (spec D5): one formatter, one parser, shared by the tool and the TUI.

Import-light on purpose: this module is read on the TUI's paint path (CON-010 spirit), so it imports
nothing beyond the standard library, and the tool imports it, never the other way round.
"""


class NoUserAvailable(Exception):
    """No user can answer in this run (headless without a terminal, a subagent): the adapter's way to say
    "nobody to ask", distinct from `None` = cancelled (spec D7/D13)."""


def format_answers(pairs: list[tuple[str, str]]) -> str:
    raise NotImplementedError


def parse_answers(text: str) -> list[tuple[str, str]] | None:
    raise NotImplementedError
