"""`/btw` side questions (plan/2026-11-03-btw/spec.md).

One model call over the current conversation that runs no tools and writes nothing: not to the store, not to the
turn's history, not to the session's totals. The loop publishes a snapshot of its history once per iteration
(`publish_snapshot`), the TUI reads it (`snapshot`) and asks (`answer_side_question`).

The system string is composed here, by the same function the loop uses (`compose_system_instruction`), and the
call sends the turn's own tool schemas: both are part of the cached prefix, so a side question reads the turn's
prompt cache instead of paying for the whole conversation again (D2, D4).

Light on import: the TUI imports this module before its first frame, so nothing here loads litellm at import time
(tasks.md open point 3).
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Iterator

from dostuff.agent.subagents.fork import ForkContext

BTW_MAX_EXCHANGES = 20  # past side questions replayed into the next one (D5); oldest dropped
BTW_FORK_DESCRIPTION = "answer a side question"  # a fork-out's run description; tells it from a model-started fork

SIDE_QUESTION_PROMPT = (
    "[Side question from the user, asked while you work. It is not a new task: answer it and nothing else.]\n"
    "Answer from the conversation above only, in one response, without calling tools — no tool will run. "
    "Your answer is shown to the user and is not added to the conversation.\n\n"
    "Question: {question}"
)
TOOL_CALL_NOTE = ("[note] No tools were available for this answer: anything above that looks like a tool call is "
                  "text, not a result.")
ASKED_TO_USE = ("The model asked to use {names} instead of answering — press f to let a background agent do it.")
EMPTY_ANSWER = "The model returned an empty answer."


# --- the snapshot registry (D3) ---------------------------------------------------------------------------------

@dataclass
class _Entry:
    live: ForkContext | None = None  # published by the running turn
    kept: ForkContext | None = None  # demoted when the turn ended: its prompt parts, history dropped


_entries: dict[str, _Entry] = {}


def _copy(ctx: ForkContext, history: list[dict[str, Any]] | None = None) -> ForkContext:
    return dataclasses.replace(
        ctx, history=list(ctx.history) if history is None else history, tool_names=list(ctx.tool_names),
        session_hook_context=dict(ctx.session_hook_context) if ctx.session_hook_context is not None else None,
    )


def publish_snapshot(session_id: str, ctx: ForkContext | None) -> None:
    """The running turn's history and prompt parts, as of this iteration. `None` (a subagent turn) is a no-op."""
    if ctx is None:
        return
    entry = _entries.setdefault(session_id, _Entry())
    entry.live = _copy(ctx)


def snapshot(session_id: str) -> ForkContext | None:
    """The live snapshot of a turn that is running now, else None."""
    entry = _entries.get(session_id)
    return entry.live if entry else None


def kept_prompt(session_id: str) -> ForkContext | None:
    """The last finished turn's prompt parts (`history == []`), for an idle side question, else None."""
    entry = _entries.get(session_id)
    return entry.kept if entry else None


def discard_snapshot(session_id: str) -> None:
    """Forget a session's entry, live and kept (a session switch, D10)."""
    _entries.pop(session_id, None)


def _demote(session_id: str) -> None:
    entry = _entries.get(session_id)
    if entry is not None and entry.live is not None:
        entry.kept = _copy(entry.live, history=[])
        entry.live = None


@contextlib.contextmanager
def published_snapshot(session_id: str, enabled: bool = True) -> Iterator[None]:
    """Around a turn: on every exit, demote its live snapshot to the kept prompt parts. Disabled (a subagent loop,
    which shares the main session id), it does nothing (tasks.md open point 1)."""
    try:
        yield
    finally:
        if enabled:
            _demote(session_id)


def _reset_registry() -> None:
    """Tests only."""
    _entries.clear()


# --- what the loop and the side call share (D4, D9) -------------------------------------------------------------

def plan_mode_on(permissions: Any, config: Any = None) -> bool:
    """Plan mode, read per call: the session's override, else the configured mode (Shift+Tab changes it mid-turn)."""
    if config is None:
        from dostuff.config import get_config
        config = get_config()
    return ((permissions.mode if permissions else None) or getattr(config, "permission_mode", "ask")) == "plan"


def compose_system_instruction(base: str, plan_mode: bool, compaction_notes: str,
                               session_hook_context: dict[str, str] | None) -> str:
    """The system string a model call gets. One copy, so the loop and `/btw` can't differ by a byte (D4)."""
    from dostuff.agent.permissions import PLAN_MODE_INSTRUCTION
    from dostuff.helpers.hooks.events import session_context_suffix
    return (base
            + (f"\n\n{PLAN_MODE_INSTRUCTION}" if plan_mode else "")
            + (f"\n\n[Summary of earlier conversation]: {compaction_notes}" if compaction_notes else "")
            + session_context_suffix(session_hook_context))


def usage_dict(interaction: Any) -> dict[str, Any] | None:
    """A response's token usage and USD cost (None when unpriced); None when the response has no usage."""
    usage = getattr(interaction, "usage", None)
    if not usage:
        return None
    from dostuff.helpers.agent.model_cost import call_cost
    from dostuff.lib.model import get_litellm_model
    result: dict[str, Any] = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        # Prompt tokens read from the provider's prompt cache (0 if none / unsupported)
        "cached_tokens": getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
    }
    # USD from litellm's price table; None if unknown (the TUI then hides the cost)
    result["cost"] = call_cost(get_litellm_model(), result)
    return result


# --- the side call (D2, D5, D7) ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class SideAnswer:
    answer: str  # the model's answer alone: what the side thread replays (D5)
    sent_question: str  # the question step as sent, wrapped in SIDE_QUESTION_PROMPT
    usage: dict[str, Any] | None
    cost: float | None
    tool_call_note: str | None  # the D7 note, when the reply had or looked like a tool call

    @property
    def text(self) -> str:
        """What the user sees and a fork-out quotes: the answer and the D7 note."""
        return f"{self.answer}\n\n{self.tool_call_note}" if self.tool_call_note else self.answer


def build_messages(snap: ForkContext, exchanges: list[tuple[str, str]], question: str) -> list[dict[str, Any]]:
    """The snapshot's history, the side thread as it was sent, then this question."""
    messages: list[dict[str, Any]] = list(snap.history)
    for sent, answer in exchanges:
        messages.append({"role": "user", "content": sent})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": SIDE_QUESTION_PROMPT.format(question=question)})
    return messages


def add_exchange(exchanges: list[tuple[str, str]], sent: str, answer: str) -> None:
    """Append one exchange to the side thread, dropping the oldest past BTW_MAX_EXCHANGES."""
    exchanges.append((sent, answer))
    del exchanges[:-BTW_MAX_EXCHANGES]


def _tool_call_like(text: str) -> bool:
    """A JSON object in the text with a `name` and an `arguments`/`parameters` key at its top level."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except ValueError:
            obj = None
        if isinstance(obj, dict) and "name" in obj and ("arguments" in obj or "parameters" in obj):
            return True
        start = text.find("{", start + 1)
    return False


def _note(names: list[str]) -> str:
    return f"{TOOL_CALL_NOTE} Asked for: {', '.join(names)}." if names else TOOL_CALL_NOTE


async def answer_side_question(snap: ForkContext, exchanges: list[tuple[str, str]], question: str,
                               plan_mode: bool) -> SideAnswer:
    """One model call, with the turn's tools sent and none run. Takes no adapter and no store, so it can't emit
    usage or write anything (CON-004, CON-010). Raises what the call raises."""
    from dostuff.agent.call_agent import call_agent
    sent = SIDE_QUESTION_PROMPT.format(question=question)
    interaction, span = await call_agent(
        steps_history=build_messages(snap, exchanges, question),
        system_instruction=compose_system_instruction(
            snap.system_instruction, plan_mode, snap.compaction_notes, snap.session_hook_context),
        tool_names=list(snap.tool_names),  # the turn's, not []: the schemas are part of the cached prefix (D2)
        stream=False,
    )
    if span is not None and span.is_recording():
        span.end()
    choices = getattr(interaction, "choices", None)
    message = choices[0].message if choices else None
    content = (getattr(message, "content", None) or "").strip() if message else ""
    tool_calls = getattr(message, "tool_calls", None) if message else None
    names = [tc.function.name for tc in tool_calls or [] if getattr(tc, "type", "function") == "function"]
    if content:
        answer = content
    elif names:
        answer = ASKED_TO_USE.format(names=", ".join(names))
    else:
        answer = EMPTY_ANSWER
    note = _note(names) if names or _tool_call_like(content) else None
    usage = usage_dict(interaction)
    return SideAnswer(answer=answer, sent_question=sent, usage=usage, cost=usage["cost"] if usage else None,
                      tool_call_note=note)
