"""The `ask_user` tool: the model asks 1-4 questions and the user answers them (spec D1 - D6).

It is an ordinary tool that reaches the UI through a harness-injected `adapter` (D1, like `edit_config`
and `agent`). Two adapters implement `ask_user(questions)`:

    TuiAdapter       the panel (D6)
    HeadlessAdapter  one line per question on the terminal, or "no user" without one (D13)

The tool returns a fixed, parseable text (D5); `helpers/ui/ask_answers.py` owns that grammar and the
TUI renders it with the same parser, so there is one copy of the data.
"""
import json
from typing import Any

# pydantic on Python < 3.12 rejects typing.TypedDict when building tool schemas
from typing_extensions import Annotated, NotRequired, TypedDict

from pydantic import Field

from dostuff.helpers.ui.ask_answers import (
    CANCELLED_ASK_RESULT, UNAVAILABLE_ASK_RESULT, NoUserAvailable, format_answers, one_line,
)

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

    Args:
        questions: The questions, 1 to 4 of them.

    Returns:
        The answers, one numbered question per line. A refusal (`error: …`) names the fault; fix it
        and call again.
    """
    cleaned = _clean_questions(questions)
    refusal = _refusal(cleaned, questions)
    if refusal:
        return refusal
    if adapter is None or not callable(getattr(adapter, "ask_user", None)):
        return UNAVAILABLE_ASK_RESULT  # a unit test or a run with nobody to ask (REQ-1.4)
    try:
        answers = await adapter.ask_user(cleaned)
    except NoUserAvailable:
        return UNAVAILABLE_ASK_RESULT
    except Exception as e:  # the tool never lets an adapter exception escape into the loop
        return f"error: ask_user could not ask: {e}"
    if not isinstance(answers, (list, tuple)) or len(answers) != len(cleaned):
        return CANCELLED_ASK_RESULT  # answered nothing usable: the adapter's own "cancelled" is None
    return format_answers([(q["question"], _answer(pair)) for q, pair in zip(cleaned, answers)])


# ── Cleaning (D4) ─────────────────────────────────────────────────────────────
# Bad input is a normal case, not an exception: the arguments are model-authored. Cosmetic problems
# are clamped away so the call stays useful, structural ones are refused with an `error:` result the
# model can fix and retry.

def _clamp(text: object, limit: int, ellipsis: bool = True) -> str:
    """Whitespace-collapsed and cut to `limit` chars; with `ellipsis` one char of the budget goes to `…`,
    so the user can see the text was cut."""
    value = one_line(text)
    if len(value) <= limit:
        return value
    return value[:limit - 1] + "…" if ellipsis else value[:limit]


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _clean_options(raw: object) -> list[dict[str, str]]:
    """The options that have a label; anything else (no label, not a dict) is dropped."""
    if not isinstance(raw, list):
        return []
    cleaned: list[dict[str, str]] = []
    for option in raw:
        if not isinstance(option, dict) or not _has_text(option.get("label")):
            continue
        entry = {"label": _clamp(option["label"], ASK_LABEL_CHARS)}
        if _has_text(option.get("description")):
            entry["description"] = _clamp(option["description"], ASK_DESCRIPTION_CHARS)
        cleaned.append(entry)
    return cleaned


def _raw_questions(raw: object) -> list[Any]:
    """`questions` as a list, whatever shape the model sent. It may arrive as a JSON string (a fake model,
    an old transcript); a non-list is an empty list, so the "no questions" refusal covers it instead of
    raising a `TypeError`."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return raw if isinstance(raw, list) else []


def _has_question_text(question: object) -> bool:
    return isinstance(question, dict) and _has_text(question.get("question"))


def _clean_questions(raw: object) -> list[dict[str, Any]]:
    """The questions with the cosmetic fields clamped and the empty ones dropped (D4)."""
    kept = [q for q in _raw_questions(raw) if _has_question_text(q)]
    cleaned: list[dict[str, Any]] = []
    for i, question in enumerate(kept, 1):  # Q<n> counts the dropped questions out
        entry: dict[str, Any] = {
            "question": one_line(question["question"]),
            "header": _clamp(question["header"], ASK_HEADER_CHARS, ellipsis=False)
            if _has_text(question.get("header")) else f"Q{i}",
            "options": _clean_options(question.get("options")),
            "multi_select": question.get("multi_select") is True,
        }
        cleaned.append(entry)
    return cleaned


# ── Refusing (D4) ─────────────────────────────────────────────────────────────

def _refusal(cleaned: list[dict[str, Any]], raw: object) -> str | None:
    """The first structural fault as an `error:` result the model can fix and retry, or None.

    D4's order: how many questions were sent, then each question's options, then duplicate questions.
    The count is of the raw list, so a call that sent 5 (two of them empty) is told 5 rather than 3.
    """
    sent = _raw_questions(raw)
    if not cleaned:
        return "error: ask_user needs at least one question with text."
    if len(sent) > ASK_MAX_QUESTIONS:
        return (f"error: ask_user takes 1-{ASK_MAX_QUESTIONS} questions, got {len(sent)}. "
                "Split them across calls.")
    for question, cleaned_question in zip([q for q in sent if _has_question_text(q)], cleaned):
        if question.get("options") is not None and not isinstance(question["options"], list):
            return (f'error: question "{cleaned_question["question"]}" has options that are not a list of '
                    "{label, description} objects.")
    for q in cleaned:
        text, options = q["question"], q["options"]
        if len(options) > ASK_MAX_OPTIONS:
            return f'error: question "{text}" has {len(options)} options; the limit is {ASK_MAX_OPTIONS}.'
        if len(options) == 1:
            return (f'error: question "{text}" has one option. Give it 2-{ASK_MAX_OPTIONS}, '
                    "or leave options out for a free-text answer.")
        labels = [option["label"] for option in options]
        duplicates = [label for label in labels if labels.count(label) > 1]
        if duplicates:
            return f'error: question "{text}" repeats the option "{duplicates[0]}".'
        if q["multi_select"] and not options:
            return f'error: question "{text}" is multi_select but has no options.'
    texts = [q["question"] for q in cleaned]
    repeated = [text for text in texts if texts.count(text) > 1]
    if repeated:
        return f'error: ask_user repeats the question "{repeated[0]}".'
    return None


def _answer(pair: object) -> str:
    """The answer half of one adapter pair; a bare string answers its own question."""
    if isinstance(pair, (tuple, list)) and len(pair) >= 2:
        return one_line(pair[1])
    return one_line(pair)
