"""Tool-output helpers shared by the loop and the TUI, kept free of heavy imports (litellm)."""

import json

EMPTY_RESPONSE_NUDGE = "Continue, or say you are done."


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


def format_call_label(name: str, args: object, value_chars: int = 60, max_chars: int = 200) -> str:
    """`write_file(path="a.py", content="import os…")`: one line, each long value clipped.

    `args` is the parsed dict or the raw JSON string stored in history; a file's whole
    `content` would otherwise flood the conversation.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            pass
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
