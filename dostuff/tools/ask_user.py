"""The `ask_user` tool: the model asks 1-4 questions and the user answers them (spec D1 - D6).

It is an ordinary tool that reaches the UI through a harness-injected `adapter` (D1, like `edit_config`
and `agent`). Two adapters implement `ask_user(questions)`:

    TuiAdapter       the panel (D6)
    HeadlessAdapter  one line per question on the terminal, or "no user" without one (D13)

The tool returns a fixed, parseable text (D5); `helpers/ui/ask_answers.py` owns that grammar and the
TUI renders it with the same parser, so there is one copy of the data.
"""
from typing import Any

# pydantic on Python < 3.12 rejects typing.TypedDict when building tool schemas
from typing_extensions import Annotated, NotRequired, TypedDict

from pydantic import Field

# Limits (D4). The tool refuses past them rather than truncating a question away.
ASK_MAX_QUESTIONS = 4
ASK_MAX_OPTIONS = 4
ASK_HEADER_CHARS = 12
ASK_LABEL_CHARS = 60
ASK_DESCRIPTION_CHARS = 200


class AskOption(TypedDict):
    label: Annotated[str, Field(description="What the user picks, 1-5 words.")]
    description: NotRequired[Annotated[str, Field(description="The trade-off, one line.")]]


class AskQuestion(TypedDict):
    question: Annotated[str, Field(description="The question, one sentence, ending with '?'.")]
    header: NotRequired[Annotated[str, Field(description="Short tab label, up to 12 chars.")]]
    options: NotRequired[Annotated[list[AskOption], Field(description="2-4 choices; omit for a free-text answer.")]]
    multi_select: NotRequired[Annotated[bool, Field(description="Allow several picks; needs options. Default false.")]]


async def ask_user(questions: list[AskQuestion], adapter: Any = None) -> str:
    """Ask the user a question (or up to 4 at once) with `ask_user`, and get their answer back.

    Use it when the answer changes what you build and you can't find it in the repo: a choice between
    libraries, a product decision, a name, a trade-off the user has opinions about. Don't use it to
    confirm a plan, to ask permission (the approval prompt does that), or for anything grep and
    read_file can answer. Ask everything you need in one call instead of several turns. Give 2-4
    concrete options with the trade-off, one line each, or leave `options` out for a free-text answer.
    A call the user cancels or a refusal can be retried.

    Args:
        questions: The questions, 1 to 4 of them.
    """
    raise NotImplementedError
