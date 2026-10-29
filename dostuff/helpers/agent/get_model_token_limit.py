import litellm
from dostuff.config import get_config
from dostuff.helpers.models.opencode import context_window, is_opencode
from dostuff.lib.model import get_litellm_model, get_model

FALLBACK_CONTEXT_WINDOW = 128_000
_warned_models: set[str] = set()


def _litellm_limit(model: str) -> int | None:
    """litellm's `max_input_tokens` for `model`, or None when it doesn't map it."""
    try:
        limit = litellm.get_model_info(model=model).get("max_input_tokens")
    except Exception:
        return None
    return int(limit) if limit else None


def get_model_token_limit() -> int:
    """Context window for the active model: `loop.context_window`, else the OpenCode table, else
    litellm's model info, else 128k.

    OpenCode ids are checked against `helpers/models/opencode._CONTEXT_WINDOWS` first: litellm only
    knows their upstream `openai/` / `anthropic/` route for a few of them, and the same id can have a
    different window on each gateway.
    """
    configured = get_config().context_window
    if configured > 0:
        return configured
    opencode = is_opencode(get_config().provider)
    if opencode:
        model_id = get_model().split("/", 1)[-1]
        known = context_window(get_config().provider, model_id)
        if known:
            return known
    model = get_litellm_model()
    limit = _litellm_limit(model)
    if limit:
        return limit
    if opencode:
        # litellm keys some ids only without the route prefix, e.g. `deepseek-v4-pro` (1M) exists
        # while `openai/deepseek-v4-pro` doesn't.
        limit = _litellm_limit(get_model().split("/", 1)[-1])
        if limit:
            return limit
    if model not in _warned_models:
        _warned_models.add(model)
        from dostuff.helpers.ui.emit import emit
        emit(
            f"Unknown context window for '{model}'; assuming {FALLBACK_CONTEXT_WINDOW:,} tokens. "
            "Set loop.context_window in config.yaml to override.",
            msg_type="system",
        )
    return FALLBACK_CONTEXT_WINDOW
