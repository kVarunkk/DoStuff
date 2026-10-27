import litellm
from dostuff.config import get_config
from dostuff.lib.model import get_litellm_model

FALLBACK_CONTEXT_WINDOW = 128_000
_warned_models: set[str] = set()


def get_model_token_limit() -> int:
    """Context window for the active model: `loop.context_window`, else litellm's model info, else 128k."""
    configured = get_config().context_window
    if configured > 0:
        return configured
    model = get_litellm_model()
    try:
        limit = litellm.get_model_info(model=model).get("max_input_tokens")
        if limit:
            return int(limit)
    except Exception:
        pass
    if model not in _warned_models:
        _warned_models.add(model)
        from dostuff.helpers.ui.emit import emit
        emit(
            f"Unknown context window for '{model}'; assuming {FALLBACK_CONTEXT_WINDOW:,} tokens. "
            "Set loop.context_window in config.yaml to override.",
            msg_type="system",
        )
    return FALLBACK_CONTEXT_WINDOW
