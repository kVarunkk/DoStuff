"""USD cost of a model call, from litellm's price table."""
from typing import Any


def call_cost(model: str, usage: dict[str, Any]) -> float | None:
    """Cost of one call; None when the model's price is unknown or zero (e.g. Ollama)."""
    import litellm  # lazily: the TUI imports format_cost before litellm has loaded
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cache_read_input_tokens=int(usage.get("cached_tokens") or 0),
        )
    except Exception:
        return None
    total = float(prompt_cost) + float(completion_cost)
    return total if total > 0 else None


def format_cost(usd: float) -> str:
    return f"${usd:.2f}" if usd >= 0.01 else f"${usd:.4f}"
