"""Render a stored session as Markdown (`dostuff export`, `/export`)."""
import json
from typing import Any

from dostuff.helpers.memory.shell_step import parse_shell_step

EMPTY_RESPONSE_NUDGE = "Continue, or say you are done."  # loop.EMPTY_RESPONSE_NUDGE; not worth importing litellm for


def _fence(text: str, lang: str = "") -> str:
    """A code block whose fence is longer than any backtick run inside the text."""
    longest, run = 0, 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{text.rstrip()}\n{fence}"


def _pretty_args(raw: Any) -> str:
    if isinstance(raw, str):
        try:
            return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return raw
    return json.dumps(raw, indent=2, ensure_ascii=False)


def session_to_markdown(steps: list[dict[str, Any]], meta: dict[str, Any] | None, session_id: str) -> str:
    """The whole audit trail: prompts, answers, tool calls with their arguments, and tool results."""
    meta = meta or {}
    title = meta.get("title") or f"Session {session_id}"
    lines = [f"# {title}", ""]
    for label, value in (
        ("Session", f"`{session_id}`"),
        ("Directory", f"`{meta['working_dir']}`" if meta.get("working_dir") else ""),
        ("Last used", meta.get("last_used") or ""),
        ("Tokens", f"{meta.get('prompt_tokens') or 0:,} in / {meta.get('completion_tokens') or 0:,} out" if meta.get("total_tokens") else ""),
    ):
        if value:
            lines.append(f"- **{label}:** {value}")
    lines.append("")

    for step in steps:
        role = step.get("role")
        content = step.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if role == "user":
            if content == EMPTY_RESPONSE_NUDGE:
                continue
            shell = parse_shell_step(content)
            if shell is not None:
                lines += ["## Shell", "", _fence(f"$ {shell[0]}\n{shell[1]}", "console"), ""]
            else:
                lines += ["## User", "", content, ""]
        elif role == "assistant":
            calls = step.get("tool_calls") or []
            if content or not calls:
                lines += ["## Assistant", "", content or "_(no text)_", ""]
            for tc in calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                lines += [f"**Tool call:** `{fn.get('name', '?')}`", "", _fence(_pretty_args(fn.get("arguments", "")), "json"), ""]
        elif role == "tool":
            lines += [f"**Result** (`{step.get('name', '')}`):", "", _fence(content), ""]
        elif content:
            lines += [f"## {str(role).title()}", "", content, ""]
    return "\n".join(lines).rstrip() + "\n"
