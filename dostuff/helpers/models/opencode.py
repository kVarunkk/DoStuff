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
# Input-token limit per model, from models.dev's `opencode` / `opencode-go` entries
# (https://models.dev — the catalog OpenCode's own clients size their context with). The gateways
# don't publish limits in /v1/models, litellm doesn't map their ids, and the same model can have a
# different window on each gateway, so this table is the only reliable source. Refresh it when
# `provider_models()` lists an id that is missing here (those fall back to litellm, then 128k).
_CONTEXT_WINDOWS = {
    "opencode": {
        "beam-free": 1_048_576,
        "big-pickle": 160_000,
        "claude-fable-5": 1_000_000,
        "claude-fable-5-1": 1_000_000,
        "claude-haiku-4-5": 200_000,
        "claude-haiku-5-5": 1_000_000,
        "claude-opus-4-5": 200_000,
        "claude-opus-4-6": 1_000_000,
        "claude-opus-4-7": 1_000_000,
        "claude-opus-4-8": 1_000_000,
        "claude-opus-5": 1_000_000,
        "claude-opus-5-5": 1_000_000,
        "claude-sonnet-4-5": 1_000_000,
        "claude-sonnet-4-6": 1_000_000,
        "claude-sonnet-5": 1_000_000,
        "claude-sonnet-5-5": 1_000_000,
        "deepseek-v4-pro": 1_000_000,
        "deepseek-v4.1-flash": 1_000_000,
        "deepseek-v4.1-flash-free": 200_000,
        "flint-40b-free": 262_144,
        "gemini-3-flash": 1_048_576,
        "gemini-3.1-pro": 1_048_576,
        "gemini-3.5-flash": 1_048_576,
        "gemini-3.5-flash-lite": 1_048_576,
        "gemini-3.6-flash": 1_048_576,
        "gemini-3.7-flash": 1_048_576,
        "gemini-3.8-flash": 1_048_576,
        "gemini-4-argon": 1_048_576,
        "glm-5": 204_800,
        "glm-5.1": 204_800,
        "glm-5.2": 1_000_000,
        "glm-5.3": 1_000_000,
        "glm-5.3-flash": 1_000_000,
        "glm-5.3-flash-free": 200_000,
        "gpt-5": 272_000,
        "gpt-5-codex": 272_000,
        "gpt-5-nano": 272_000,
        "gpt-5.1": 272_000,
        "gpt-5.1-codex": 272_000,
        "gpt-5.1-codex-max": 272_000,
        "gpt-5.1-codex-mini": 272_000,
        "gpt-5.2": 272_000,
        "gpt-5.2-codex": 272_000,
        "gpt-5.3-codex": 272_000,
        "gpt-5.3-codex-spark": 128_000,
        "gpt-5.4": 922_000,
        "gpt-5.4-mini": 272_000,
        "gpt-5.4-nano": 272_000,
        "gpt-5.4-pro": 922_000,
        "gpt-5.5": 922_000,
        "gpt-5.5-pro": 922_000,
        "gpt-5.6-luna": 922_000,
        "gpt-5.6-sol": 922_000,
        "gpt-5.6-terra": 922_000,
        "gpt-6-astra": 922_000,
        "gpt-6-luna": 922_000,
        "gpt-6-sol": 922_000,
        "gpt-6.1-sol": 922_000,
        "grok-4.5": 500_000,
        "grok-4.6": 500_000,
        "grok-4.7": 500_000,
        "grok-build-0.1": 256_000,
        "kimi-k2.5": 262_144,
        "kimi-k2.6": 262_144,
        "kimi-k2.7-code": 262_144,
        "kimi-k3": 1_048_576,
        "mimo-v2.6-flash-free": 200_000,
        "minimax-m2.5": 204_800,
        "minimax-m2.7": 204_800,
        "minimax-m3": 512_000,
        "mistral-large-4": 524_288,
        "muse-spark-1.2": 1_048_576,
        "muse-spark-1.3": 1_048_576,
        "muse-wonder-1.0": 1_048_576,
        "nemotron-3.5-lightning-free": 262_144,
        "nemotron-3.5-super-vl-free": 1_048_576,
        "qwen3.5-plus": 262_144,
        "qwen3.6-plus": 262_144,
        "qwen3.8-flash": 1_000_000,
        "qwen3.8-max": 262_144,
    },
    "opencode-go": {
        "beam-free": 1_048_576,
        "claude-haiku-5-5": 1_000_000,
        "deepseek-v4-pro": 1_000_000,
        "deepseek-v4.1-flash": 1_000_000,
        "gemini-4-argon": 1_048_576,
        "glm-5.2": 1_000_000,
        "glm-5.3": 1_000_000,
        "glm-5.3-flash": 1_000_000,
        "gpt-5.6-luna": 922_000,
        "gpt-6-luna": 922_000,
        "grok-4.7": 500_000,
        "hy3": 192_000,
        "hy4-preview": 1_024_000,
        "iceman-contributor-free": 262_144,
        "kimi-k2.7-code": 262_144,
        "kimi-k3": 1_048_576,
        "kimi-k3.1-pro": 1_048_576,
        "longcat-2.0": 1_000_000,
        "mimo-v2.6-flash": 1_048_576,
        "mimo-v2.6-pro": 1_048_576,
        "minimax-m3": 1_000_000,
        "minimax-m3.1-flash": 1_000_000,
        "muse-spark-1.2-contributor": 1_048_576,
        "muse-spark-1.3-contributor": 1_048_576,
        "qwen3.7-plus": 1_000_000,
        "qwen3.8-flash": 1_000_000,
        "qwen3.8-max": 1_000_000,
        "step-5-preview": 1_000_000,
    },
}
# In /v1/models but not chat models
_NOT_CHAT = ("jev-", "test")

Endpoint = Literal["chat", "responses", "anthropic", "google"]


def is_opencode(provider: str) -> bool:
    return provider in OPENCODE_PROVIDERS


def context_window(provider: str, model_id: str) -> int | None:
    """Input-token limit for an OpenCode model id, or None if the table doesn't list it."""
    return _CONTEXT_WINDOWS.get(provider, {}).get(model_id)


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
