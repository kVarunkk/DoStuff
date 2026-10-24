"""Tool-output helpers shared by the loop and the TUI, kept free of heavy imports (litellm)."""

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
