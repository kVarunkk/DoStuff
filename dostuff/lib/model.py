"""Active model / provider, read from the Config singleton at call time (never frozen at import)."""
import uuid
from contextvars import ContextVar
from typing import Any

from dostuff.config import get_config
from dostuff.helpers.models.opencode import OPENCODE_PROVIDERS, is_opencode, litellm_route, opencode_kwargs

# The running session, for providers that want a per-conversation id (OpenCode Go); set by the loop
current_session_id: ContextVar[str] = ContextVar("current_session_id", default=uuid.uuid4().hex)


def get_model() -> str:
    """litellm model string, "provider/model" (env MODEL overrides config.yaml)."""
    return get_config().model_name


def get_provider() -> str:
    return get_config().provider


def _opencode_id() -> str:
    """The model id without the `opencode/` / `opencode-go/` prefix."""
    model, provider = get_model(), get_provider()
    return model[len(provider) + 1:] if model.startswith(provider + "/") else model


def get_litellm_model() -> str:
    """The model litellm is called with: get_model(), except for gateways routed through another
    provider (OpenCode), so litellm's model info (context window, prices) applies."""
    if is_opencode(get_provider()):
        return litellm_route(get_provider(), _opencode_id())[0]
    return get_model()


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
    return provider in ("bedrock", "vertex_ai", "openrouter", *OPENCODE_PROVIDERS) and "claude" in get_model().lower()


def completion_kwargs() -> dict[str, Any]:
    """Common kwargs for every litellm call: the model, plus `api_key` when `api_key_env` is set."""
    api_key = get_config().api_key()
    provider = get_provider()
    if is_opencode(provider):
        from dostuff import __version__
        return opencode_kwargs(provider, _opencode_id(), api_key, current_session_id.get(), __version__)
    kwargs: dict[str, Any] = {"model": get_model()}
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs
