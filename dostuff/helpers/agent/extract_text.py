from typing import Any

def extract_text(step: Any) -> str | None:
    content = getattr(step, "content", None)
    if not content:
        return None

    for item in content:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text:
            return text

    return None