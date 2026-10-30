"""Turn-level hook helpers used by the TUI, headless and loop() (spec D1, REQ-2.2–REQ-2.5).

The callers fire UserPromptSubmit and the startup/resume/clear SessionStart themselves: they append the
user step before calling loop(), so a hook inside loop() would be too late to keep a prompt out of history.
"""
import re
from typing import Any

from dostuff.helpers.hooks.dispatch import report_outcome, run_hooks
from dostuff.helpers.hooks.load_hooks import SESSION_START_SOURCES
from dostuff.helpers.hooks.run_hook import payload_for

BLOCKED_PROMPT_MESSAGE = "Prompt blocked by a UserPromptSubmit hook: {reason}"
STOP_NOTE = "[Stop hook {name}] {reason}"  # the user step a Stop block appends (REQ-2.3)
_STOP_NOTE_RE = re.compile(r"\[Stop hook (.+?)\] (.*)", re.DOTALL)
STOP_LIMIT_MESSAGE = ("{event} hook '{name}' blocked again after {limit} continuation(s) "
                      "(hooks.stop_block_limit); ending the turn.")


def parse_stop_note(content: Any) -> tuple[str, str] | None:
    """(hook name, reason) if a stored user step is a Stop-hook continuation, else None (REQ-2.8)."""
    if not isinstance(content, str):
        return None
    match = _STOP_NOTE_RE.fullmatch(content)
    return (match[1], match[2]) if match else None


def blocked_prompt_message(reason: str) -> str:
    """What the TUI shows and headless prints when a prompt is rejected, so both read the same."""
    return BLOCKED_PROMPT_MESSAGE.format(reason=reason or "no reason given")


def stop_event_for(turn_type: str) -> str:
    return "SubagentStop" if turn_type == "subagent_loop" else "Stop"


async def user_prompt_submit(
    prompt: str, *, session_id: str, turn_id: str, turn_type: str = "interactive_loop", adapter: Any = None,
) -> tuple[str | None, str]:
    """(blocked reason or None, extra context) for a prompt about to be appended as the user step.

    A block (exit 2, `decision: block`, or a failed `fail_closed` hook) rejects the prompt; the context
    belongs in that turn's system instruction only and is never persisted (REQ-2.2).
    """
    payload = payload_for("UserPromptSubmit", session_id=session_id, turn_id=turn_id, turn_type=turn_type,
                          prompt=prompt)
    outcome = await run_hooks("UserPromptSubmit", payload, adapter=adapter)
    await report_outcome("UserPromptSubmit", outcome, adapter=adapter, show_block=False)
    if outcome.cancelled:
        return "cancelled by user", ""
    if outcome.decision == "deny":
        return outcome.reason or f"hook '{outcome.decided_by}'", ""
    return None, outcome.context


def with_prompt_context(instructions: str, context: str) -> str:
    """The turn's system instruction plus UserPromptSubmit context."""
    return f"{instructions}\n\n[Context from UserPromptSubmit hooks]\n{context}" if context else instructions


async def session_start(
    source: str, holder: dict[str, str], *, session_id: str, turn_id: str = "",
    turn_type: str = "interactive_loop", adapter: Any = None,
) -> None:
    """Run SessionStart hooks for `source` and store their context under holder[source] (REQ-2.4).

    Each source replaces only its own key; the holder is session state that loop() adds to every model call.
    """
    payload = payload_for("SessionStart", session_id=session_id, turn_id=turn_id, turn_type=turn_type,
                          source=source)
    outcome = await run_hooks("SessionStart", payload, adapter=adapter)
    await report_outcome("SessionStart", outcome, adapter=adapter, target=source)
    if outcome.context:
        holder[source] = outcome.context
    else:
        holder.pop(source, None)


def session_context_suffix(holder: dict[str, str] | None) -> str:
    """The SessionStart context as it is appended to the system instruction."""
    if not holder:
        return ""
    joined = "\n".join(holder[s] for s in SESSION_START_SOURCES if holder.get(s))
    return f"\n\n[Session context from SessionStart hooks]\n{joined}" if joined else ""
