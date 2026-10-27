"""OpenCode Zen (`opencode/<id>`) and OpenCode Go (`opencode-go/<id>`) models.

litellm has no OpenCode provider. Each gateway serves a model family on the endpoint of its
upstream API (https://opencode.ai/docs/zen/), so a model is sent through litellm's openai,
anthropic or gemini route with the gateway as api_base.
"""
import os
from typing import Any, Literal

OPENCODE_PROVIDERS = {
    "opencode": "https://opencode.ai/inference",
    "opencode-go": "https://opencode.ai/inference/go",
}
OPENCODE_API_KEY_ENV = "OPENCODE_API_KEY"  # one console key works for Zen and Go
# Served on the Anthropic Messages endpoint besides Claude; they differ between the gateways
_ANTHROPIC_IDS = {
    "opencode": {"qwen3.8-flash", "qwen3.6-plus", "qwen3.5-plus"},
    "opencode-go": {"qwen3.8-max", "qwen3.8-flash", "qwen3.7-plus", "minimax-m3"},
}
# In /v1/models but not chat models
_NOT_CHAT = ("jev-", "test")

Endpoint = Literal["chat", "responses", "anthropic", "google"]


def is_opencode(provider: str) -> bool:
    return provider in OPENCODE_PROVIDERS


def endpoint_for(provider: str, model_id: str) -> Endpoint:
    """Which upstream API the gateway serves `model_id` on."""
    if model_id.startswith("claude-") or model_id in _ANTHROPIC_IDS[provider]:
        return "anthropic"
    if model_id.startswith(("gpt-", "grok-", "muse-")):
        return "responses"
    if model_id.startswith("gemini-"):
        return "google"
    return "chat"


def litellm_route(provider: str, model_id: str) -> tuple[str, str]:
    """(litellm model, api_base) for an OpenCode model id."""
    base = OPENCODE_PROVIDERS[provider]
    endpoint = endpoint_for(provider, model_id)
    if endpoint == "anthropic":
        return f"anthropic/{model_id}", f"{base}/anthropic"  # litellm appends /v1/messages
    if endpoint == "google":
        return f"gemini/{model_id}", f"{base}/google/v1beta"
    if endpoint == "responses":
        return f"openai/responses/{model_id}", f"{base}/openai/v1"
    return f"openai/{model_id}", f"{base}/openai/v1"


def opencode_kwargs(provider: str, model_id: str, api_key: str | None, session_id: str, version: str) -> dict[str, Any]:
    """litellm kwargs for an OpenCode model. Go asks clients to name themselves and send a
    stable per-conversation id; Zen gets the same headers."""
    model, api_base = litellm_route(provider, model_id)
    return {
        "model": model,
        "api_base": api_base,
        # Never fall back to OPENAI_API_KEY / ANTHROPIC_API_KEY: those would go to opencode.ai
        "api_key": api_key or os.environ.get(OPENCODE_API_KEY_ENV) or "missing-opencode-key",
        "extra_headers": {"User-Agent": f"dostuff/{version}", "x-opencode-session": session_id},
    }


def models_url(provider: str) -> str:
    return f"{OPENCODE_PROVIDERS[provider]}/v1/models"


def chat_model_ids(data: dict) -> list[str]:
    """Model ids from a /v1/models response, without the non-chat ones."""
    return [
        m["id"] for m in data.get("data") or []
        if isinstance(m, dict) and m.get("id") and not str(m["id"]).startswith(_NOT_CHAT)
    ]
