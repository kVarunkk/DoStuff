import json
from typing import Literal

def format_transcript(
    steps_history: list[dict],
    type: Literal["memory_update", "skill_update"],
    max_result_chars: int = 1000,
) -> str:
    """Extracts and formats text messages, tool calls, and tool results from steps history.

    - type='memory_update': Formats only conversational text (User / Assistant).
    - type='skill_update': Formats text, nested tool calls, and tool results.
    """
    lines = []

    # Map OpenAI roles to human-readable labels
    role_labels = {
        "user": "User",
        "assistant": "Model",
        "tool": "Function Result",
    }

    for step in steps_history:
        role = step.get("role", "")
        
        # Skip steps that don't match the standard role schema mapping
        if role not in role_labels:
            continue

        label = role_labels[role]

        # -------------------------------------------------------------------------
        # Case A: User Message Block
        # Schema: {"role": "user", "content": "..."}
        # -------------------------------------------------------------------------
        if role == "user":
            content = step.get("content")
            if content and isinstance(content, str):
                lines.append(f"{label}: {content.strip()}")
            continue

        # -------------------------------------------------------------------------
        # Case B: Assistant Message Block (May contain plain text and/or tool calls)
        # Schema: {"role": "assistant", "content": "...", "tool_calls": [...]}
        # -------------------------------------------------------------------------
        if role == "assistant":
            content = step.get("content")
            tool_calls = step.get("tool_calls", [])

            # 1. Append conversational text if present
            if content and isinstance(content, str):
                lines.append(f"{label}: {content.strip()}")

            # 2. Append tool calls if this is a skill update run
            if type == "skill_update" and tool_calls and isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    func_data = tc.get("function", {})
                    fn_name = func_data.get("name", "unknown_function")
                    args = func_data.get("arguments", {})
                    
                    args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                    lines.append(f"Function Call ({fn_name}): {args_str}")
            continue

        # -------------------------------------------------------------------------
        # Case C: Tool Result Block
        # Schema: {"role": "tool", "name": "...", "tool_call_id": "...", "content": "..."}
        # -------------------------------------------------------------------------
        if role == "tool" and type == "skill_update":
            fn_name = step.get("name", "unknown_function")
            raw_result = step.get("content", "")

            # Convert dictionary payloads or list results back to clean strings
            if isinstance(raw_result, (dict, list)):
                result_str = json.dumps(raw_result)
            else:
                result_str = str(raw_result)

            # Truncate long tool outputs to preserve token context windows
            if len(result_str) > max_result_chars:
                truncated_count = len(result_str) - max_result_chars
                result_str = (
                    result_str[:max_result_chars]
                    + f"... [Truncated {truncated_count} chars]"
                )

            lines.append(f"{label} ({fn_name}): {result_str}")
            continue

    return "\n".join(lines)
