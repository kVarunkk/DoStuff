import json
from litellm import acompletion, ModelResponse
from dostuff.lib.model import completion_kwargs

# Share of the context window one summary request may use (prompt + notes + chunk)
SUMMARY_REQUEST_SHARE = 0.5
_SUMMARY_PROMPT_TOKENS = 400  # rough size of the instructions below


def estimate_tokens(step: dict) -> int:
    """Rough token estimate for a single step dict. Not exact — real tokenization
    isn't a fixed chars-per-token ratio — but good enough for deciding a trim boundary.
    """
    text = json.dumps(step, default=str)
    return len(text) // 4


def _exchanges(steps_history: list[dict]) -> list[list[dict]]:
    """Group steps into units that must stay together: an assistant tool-call step plus its
    tool results is one unit; every other step is its own unit."""
    units: list[list[dict]] = []
    for step in steps_history:
        if step.get("role") == "tool" and units and (units[-1][0].get("tool_calls") or units[-1][0].get("role") == "tool"):
            units[-1].append(step)
        else:
            units.append([step])
    return units


def split_by_token_budget(
    steps_history: list[dict], keep_token_budget: int
) -> tuple[list[dict], list[dict]]:
    """Split into (old_steps, recent_steps): recent_steps is the longest suffix of whole
    exchanges within keep_token_budget (at least one exchange). If everything is over budget,
    the oldest complete exchange is always dropped, so compaction makes progress.

    recent_steps never starts with a tool result.
    """
    units = _exchanges(steps_history)
    unit_tokens = [sum(estimate_tokens(s) for s in unit) for unit in units]
    total = sum(unit_tokens)

    kept_units = 0
    running_total = 0
    for tokens in reversed(unit_tokens):
        if kept_units and running_total + tokens > keep_token_budget:
            break
        kept_units += 1
        running_total += tokens
    if kept_units == len(units) and total > keep_token_budget and len(units) > 1:
        kept_units -= 1

    recent_units = units[len(units) - kept_units:]
    recent_steps = [step for unit in recent_units for step in unit]
    while recent_steps and recent_steps[0].get("role") == "tool":
        recent_steps.pop(0)
    split_index = len(steps_history) - len(recent_steps)
    return steps_history[:split_index], steps_history[split_index:]


def _chunk_steps(steps: list[dict], chunk_token_budget: int) -> list[str]:
    """Serialised chunks of steps, each roughly within chunk_token_budget."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    max_chars = chunk_token_budget * 4
    for step in steps:
        text = json.dumps(step, default=str)
        if len(text) > max_chars:  # one huge step (e.g. a big tool result): keep head and tail
            half = max_chars // 2
            text = text[:half] + f" …[{len(text) - max_chars} chars omitted]… " + text[-half:]
        tokens = len(text) // 4
        if current and current_tokens + tokens > chunk_token_budget:
            chunks.append("[" + ",".join(current) + "]")
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += tokens
    if current:
        chunks.append("[" + ",".join(current) + "]")
    return chunks


async def _summarize(chunk: str, existing_notes: str) -> str:
    summary_prompt = (
        "Produce a structured session summary with exactly these sections:\n"
        "- Goal\n- Constraints & Preferences\n- Progress (Done / In Progress / Blocked)\n- Key Decisions\n- Next Steps\n- Critical Context (file paths, session IDs, DB paths)\n"
        "- Turn Context (original request, early progress, suffix notes)\n"
        "Use bullet lists under each. Keep concise. Preserve names, dates, metrics, session IDs, and file paths.\n\n"
        f"Existing compaction notes (merge/update these):\n{existing_notes}\n\n"
        f"History steps to summarize:\n{chunk}"
    )
    response = await acompletion(
        **completion_kwargs(),
        messages=[{"role": "user", "content": summary_prompt}],
    )
    if isinstance(response, ModelResponse):
        return response.choices[0].message.content or ""
    return ""


async def compact_context(steps_history: list[dict], keep_token_budget: int, existing_notes: str = "", context_window: int | None = None) -> tuple[list[dict], str]:
    """Produces a compacted view of steps_history for sending to the model, without
    mutating the original list.

    Older steps are summarised in chunks so each summary request stays under half the
    context window; each chunk's summary is merged into the running notes.

    Returns:
        (recent_steps, summary_text) — recent_steps is a suffix of steps_history to keep
        sending verbatim; summary_text replaces the old compaction notes (empty string if no
        compaction was needed).
    """
    old_steps, recent_steps = split_by_token_budget(steps_history, keep_token_budget)
    if not old_steps:
        return steps_history, ""

    if context_window is None:
        from dostuff.helpers.agent.get_model_token_limit import get_model_token_limit
        context_window = get_model_token_limit()
    request_budget = int(context_window * SUMMARY_REQUEST_SHARE)
    notes = existing_notes
    for chunk in _chunk_steps(old_steps, max(request_budget - _SUMMARY_PROMPT_TOKENS - len(notes) // 4, 1000)):
        notes = await _summarize(chunk, notes) or notes
    if not notes or notes == existing_notes:
        return steps_history, ""
    return recent_steps, notes
