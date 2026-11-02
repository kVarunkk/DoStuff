"""Tool-output helpers shared by the loop and the TUI, kept free of heavy imports (litellm, tools)."""

import json
from typing import Callable, Mapping

EMPTY_RESPONSE_NUDGE = "Continue, or say you are done."

# Per-tool call labels, names-only on purpose: this module is on the TUI's pre-first-frame path, and
# importing TOOL_MAP here once cost 27 s of startup (NOTES.md 2026-11-01, CON-010). Each entry is a
# small pure function over the parsed arguments, never a ToolSpec hook.
CallLabel = Callable[[dict], str]


def truncate_tool_output(text: str, max_chars: int) -> str:
    """Keep the head (60 %) and tail (40 %) of an over-long tool result (design D5)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    omitted = len(text) - head - tail
    marker = (
        f"\n[… {omitted} chars truncated of {len(text)} total; "
        "use read_file with offset/limit or grep to see more …]\n"
    )
    return text[:head] + marker + text[-tail:]


def _clip(text: str, limit: int = 60) -> str:
    """One line, cut to `limit` chars with `…`; the single `…` rule for every label."""
    value = " ".join(text.split())
    return value if len(value) <= limit else value[:limit - 1] + "…"


def ask_user_call_label(args: dict) -> str:
    """`"Which database should I use?" +2 more`: the first question and how many others there are.

    Falls back to the generic shape's inner text when there is nothing usable to show."""
    questions = args.get("questions")
    if not isinstance(questions, list):
        return ""
    texts = [q for q in (" ".join(str(item.get("question", "")).split()) if isinstance(item, dict) else ""
                         for item in questions) if q]
    if not texts:
        return ""
    first = f'"{_clip(texts[0])}"'
    return first if len(texts) == 1 else f"{first} +{len(texts) - 1} more"


# The tools whose arguments read badly as JSON; a later one adds one line here (ask-user spec D9)
CALL_LABELS: dict[str, CallLabel] = {"ask_user": ask_user_call_label}


def format_call_label(name: str, args: object, value_chars: int = 60, max_chars: int = 200,
                      labels: Mapping[str, CallLabel] | None = None) -> str:
    """`write_file(path="a.py", content="import os…")`: one line, each long value clipped.

    `args` is the parsed dict or the raw JSON string stored in history; a file's whole
    `content` would otherwise flood the conversation. A tool in `labels` (default `CALL_LABELS`;
    pass `{}` to disable) supplies its own inner text.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            pass
    label_fn = (CALL_LABELS if labels is None else labels).get(name)
    if label_fn is not None and isinstance(args, dict):
        custom = label_fn(args)
        if custom:
            label = f"{name}({custom})"
            return label if len(label) <= max_chars else label[:max_chars - 2] + "…)"
    if isinstance(args, dict):
        parts = []
        for key, value in args.items():
            shown = json.dumps(value, ensure_ascii=False)
            if len(shown) > value_chars:
                shown = shown[:value_chars] + "…"
            parts.append(f"{key}={shown}")
        inner = ", ".join(parts)
    else:
        inner = " ".join(str(args).split())
    label = f"{name}({inner})"
    return label if len(label) <= max_chars else label[:max_chars - 2] + "…)"
