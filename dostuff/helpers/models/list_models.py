"""Known chat models per provider, as litellm "provider/model" names, for the model pickers."""
import os
import re

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"  # public, no key needed
FETCH_TIMEOUT_S = 5.0
# litellm lists these with mode "chat", but they aren't models to talk to
_NOT_CHAT = ("guard", "moderation", "embed", "tts", "transcribe", "whisper", "realtime", "audio",
             "image", "computer-use", "search-preview", "container")
# Dated snapshot suffixes: -2024-08-06, -20251001, -0613, -09-2025
_DATED = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8}|\d{4}|\d{2}-\d{4})$")


class ModelListError(Exception):
    """The list couldn't be fetched (Ollama not running, offline for OpenRouter)."""


def provider_models(provider: str) -> list[str]:
    """Sorted model names for `provider`; [] when nothing is known. May raise ModelListError."""
    if provider == "ollama":
        return _ollama_models()
    if provider == "openrouter":
        return _openrouter_models()
    return _litellm_models(provider)


def _litellm_models(provider: str) -> list[str]:
    """litellm's catalog, chat models only, without snapshots whose undated alias is listed."""
    import litellm
    names = set()
    for raw in litellm.models_by_provider.get(provider, []):
        full = raw if raw.startswith(provider + "/") else f"{provider}/{raw}"
        info = litellm.model_cost.get(raw) or litellm.model_cost.get(full) or {}
        if info.get("mode") == "chat" and not any(word in full.lower() for word in _NOT_CHAT):
            names.add(full)
    return sorted(n for n in names if not ((m := _DATED.search(n)) and n[:m.start()] in names))


def _get_json(url: str, what: str) -> dict:
    import httpx
    try:
        response = httpx.get(url, timeout=FETCH_TIMEOUT_S)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as e:
        raise ModelListError(f"couldn't fetch {what} ({e})") from e
    return data if isinstance(data, dict) else {}


def _ollama_models() -> list[str]:
    """Models pulled into the local Ollama server."""
    base = (os.environ.get("OLLAMA_API_BASE") or "http://localhost:11434").rstrip("/")
    data = _get_json(f"{base}/api/tags", f"models from Ollama at {base}; is it running?")
    return sorted(f"ollama/{m['name']}" for m in data.get("models") or [] if isinstance(m, dict) and m.get("name"))


def _openrouter_models() -> list[str]:
    """OpenRouter's catalog, models that output text."""
    data = _get_json(OPENROUTER_MODELS_URL, "the OpenRouter model list")
    return sorted(
        f"openrouter/{m['id']}" for m in data.get("data") or []
        if isinstance(m, dict) and m.get("id")
        and "text" in ((m.get("architecture") or {}).get("output_modalities") or ["text"])
    )
