"""Active model / provider, read from the Config singleton at call time (never frozen at import)."""
from typing import Any

from dostuff.config import get_config


def get_model() -> str:
    """litellm model string, "provider/model" (env MODEL overrides config.yaml)."""
    return get_config().model_name


def get_provider() -> str:
    return get_config().provider


def uses_explicit_prompt_caching() -> bool:
    """True for Claude models, which cache only at `cache_control` breakpoints.

    OpenAI, DeepSeek and Gemini 2.5+ cache repeated prefixes on their own; adding breakpoints
    for Gemini would make litellm create billed context caches instead.
    """
    if not get_config().prompt_caching:
        return False
    provider = get_provider()
    if provider == "anthropic":
        return True
    return provider in ("bedrock", "vertex_ai", "openrouter") and "claude" in get_model().lower()


def completion_kwargs() -> dict[str, Any]:
    """Common kwargs for every litellm call: the model, plus `api_key` when `api_key_env` is set."""
    kwargs: dict[str, Any] = {"model": get_model()}
    api_key = get_config().api_key()
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs
