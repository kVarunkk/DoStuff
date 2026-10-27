"""/context: an estimate of what fills the context window, by part, with a bar."""
import json
from typing import Any, Callable

from rich.text import Text

# The MCP tools the model is given; each server's own tools are found through them, not sent
MCP_META_TOOLS = frozenset({"search_mcp_tools", "get_mcp_tool_details", "call_mcp_tool"})
BAR_WIDTH = 40
PART_STYLES = {
    "System prompt": "#8a8a8a",
    "Tools": "#c678dd",
    "MCP tools": "#d7769f",
    "Project instructions": "#e5c07b",
    "Summary of earlier messages": "#56b6c2",
    "Messages": "#8ab4f8",
}


def count_tokens(text: str, model: str) -> int:
    """litellm's tokenizer for `model`; about 4 characters a token if that fails."""
    if not text:
        return 0
    try:
        import litellm
        return int(litellm.token_counter(model=model, text=text))
    except Exception:
        return (len(text) + 3) // 4


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    parts = [content if isinstance(content, str) else json.dumps(content, ensure_ascii=False) if content else ""]
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))
    return "\n".join(p for p in parts if p)


def context_parts(system_prompt: str, project_instructions: str, tool_schemas: dict[str, dict[str, Any]],
                  messages: list[dict[str, Any]], summary: str, count: Callable[[str], int]) -> list[tuple[str, int]]:
    """(part, tokens) for each part of the next request; `system_prompt` includes the project
    instructions, which are counted on their own."""
    if project_instructions and project_instructions in system_prompt:
        system_prompt = system_prompt.replace(project_instructions, "", 1)
    tools = [s for name, s in tool_schemas.items() if name not in MCP_META_TOOLS]
    mcp = [s for name, s in tool_schemas.items() if name in MCP_META_TOOLS]
    return [
        ("System prompt", count(system_prompt)),
        ("Tools", count(json.dumps(tools)) if tools else 0),
        ("MCP tools", count(json.dumps(mcp)) if mcp else 0),
        ("Project instructions", count(project_instructions)),
        ("Summary of earlier messages", count(summary)),
        ("Messages", sum(count(_message_text(m)) for m in messages)),
    ]


def _fmt(n: int) -> str:
    return str(n) if n < 1000 else f"{n / 1000:.1f}k"


def render_context(parts: list[tuple[str, int]], limit: int, model: str, last_request: int = 0) -> Text:
    """A bar (one colour per part, then free space) and a line per part with tokens and %."""
    used = sum(n for _, n in parts)
    total = max(limit, used, 1)
    text = Text(f"Context: ~{_fmt(used)} / {_fmt(limit)} tokens ({used / total * 100:.1f}%)  ·  {model}\n")
    cells = 0
    for name, n in parts:
        width = round(n / total * BAR_WIDTH) if n else 0
        width = max(width, 1) if n else 0
        width = min(width, BAR_WIDTH - cells)
        text.append("█" * width, style=PART_STYLES.get(name, ""))
        cells += width
    text.append("░" * (BAR_WIDTH - cells), style="#4a4a4a")
    text.append("\n")
    label_width = max(len(name) for name, _ in parts) + 2
    for name, n in parts:
        text.append("  ■ ", style=PART_STYLES.get(name, ""))
        text.append(f"{name.ljust(label_width)}{_fmt(n):>7}  {n / total * 100:5.1f}%\n")
    text.append("  □ ", style="#4a4a4a")
    text.append(f"{'Free'.ljust(label_width)}{_fmt(max(limit - used, 0)):>7}  {max(limit - used, 0) / total * 100:5.1f}%\n")
    note = "Estimated with the model's tokenizer; recalled memories and the environment block vary per message."
    if last_request:
        note += f"\nThe last request used {_fmt(last_request)} input tokens (reported by the provider)."
    text.append(note, style="dim")
    return text
