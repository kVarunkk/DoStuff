"""Active model / provider, read from the Config singleton at call time (never frozen at import)."""
import uuid
from contextvars import ContextVar
from typing import Any

from dostuff.config import get_config
from dostuff.helpers.models.opencode import OPENCODE_PROVIDERS, is_opencode, litellm_route, opencode_kwargs

# The running session, for providers that want a per-conversation id (OpenCode Go); set by the loop
current_session_id: ContextVar[str] = ContextVar("current_session_id", default=uuid.uuid4().hex)
# A subagent run's own model (subagents spec D15), set inside the run's task; None = the main model.
# Context variables follow tasks and asyncio.to_thread, so the whole run (and only the run) uses it.
current_model: ContextVar[str | None] = ContextVar("current_model", default=None)


def _run_model() -> str | None:
    """The running subagent's model when it differs from the main one."""
    override = current_model.get()
    return override if override and override != get_config().model_name else None


def get_model() -> str:
    """litellm model string, "provider/model" (a subagent's own model, else env MODEL, else config.yaml)."""
    return _run_model() or get_config().model_name


def get_provider() -> str:
    override = _run_model()
    if override is None:
        return get_config().provider
    return override.split("/", 1)[0] if "/" in override else get_config().provider


def is_main_provider() -> bool:
    """False inside a subagent whose model is on another provider (its key isn't `api_key_env`'s)."""
    return get_provider() == get_config().provider


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
    """Common kwargs for every litellm call: the model, plus `api_key` when `api_key_env` is set and the
    model is on the main provider (another provider's key comes from its usual environment variable)."""
    api_key = get_config().api_key() if is_main_provider() else None
    provider = get_provider()
    if is_opencode(provider):
        from dostuff import __version__
        return opencode_kwargs(provider, _opencode_id(), api_key, current_session_id.get(), __version__)
    kwargs: dict[str, Any] = {"model": get_model()}
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs
