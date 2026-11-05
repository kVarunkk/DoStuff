"""The turn-context step (plan/2026-11-05-cache-stable-context/spec.md).

Per-turn context (environment, recalled facts and episodes, UserPromptSubmit context) and the plan-mode text ride
the history as hidden `role:"user"` steps, because anything volatile in the system prompt invalidates the provider's
whole cached prefix. The model reads them; a user never sees them. Every reader of a history skips them through
`is_turn_context`, so the marker lives here and nowhere else (D7).

Light on import: the TUI loads this before its first frame.
"""
from typing import Any

TURN_CONTEXT_MARKER = "turn_context"
# `step_data` is JSON text, so the title queries match the marker with this. A prompt quoting the same words can't
# false-match: its quotes are escaped in `step_data` (D8). A bare pattern, for binding to `NOT LIKE ?`.
HIDDEN_LIKE_PATTERN = f'%"hidden": "{TURN_CONTEXT_MARKER}"%'
PROMPT_CONTEXT_HEADER = "[Context from UserPromptSubmit hooks]"


def is_turn_context(step: object) -> bool:
    """True for a harness-authored context step: a bundle or a plan step (`meta.hidden == "turn_context"`)."""
    if not isinstance(step, dict):
        return False
    meta = step.get("meta")
    return isinstance(meta, dict) and meta.get("hidden") == TURN_CONTEXT_MARKER


def _wrap(text: str) -> str:
    return f"<turn-context>\n{text}\n</turn-context>"


def compose_turn_context(environment: str = "", facts: str = "", episodes: str = "", prompt_context: str = "") -> str:
    """The bundle: the sections that have text, in a fixed order; "" when none has (D4). Does no I/O."""
    sections = [
        environment,
        f"<relevant_user_facts>\n{facts}\n</relevant_user_facts>" if facts else "",
        f"<past_episodes>\n{episodes}\n</past_episodes>" if episodes else "",
        f"{PROMPT_CONTEXT_HEADER}\n{prompt_context}" if prompt_context else "",
    ]
    body = "\n\n".join(s for s in sections if s)
    return _wrap(body) if body else ""


def make_context_step(bundle: str) -> dict[str, Any]:
    """The hidden step that carries a bundle."""
    return {"role": "user", "content": bundle, "meta": {"hidden": TURN_CONTEXT_MARKER}}


def make_plan_mode_step(plan_mode: bool, text: str) -> dict[str, Any]:
    """The hidden step that tells the model plan mode changed (D11). `text` is passed in so this module stays light."""
    return {"role": "user", "content": _wrap(text), "meta": {"hidden": TURN_CONTEXT_MARKER, "plan_mode": plan_mode}}


def delivered_plan_mode(history: list) -> bool:
    """The newest plan step's state; False when the history has none."""
    for step in reversed(history):
        if is_turn_context(step) and "plan_mode" in step["meta"]:
            return bool(step["meta"]["plan_mode"])
    return False
