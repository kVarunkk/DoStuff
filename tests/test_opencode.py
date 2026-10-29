"""OpenCode Zen / Go models: routing through litellm's openai / anthropic / gemini providers."""
import pytest

import dostuff.helpers.agent.get_model_token_limit as token_limit_module
from dostuff import config as config_module
from dostuff.config import reset_config
from dostuff.helpers.models import list_models
from dostuff.helpers.models.opencode import chat_model_ids, endpoint_for, litellm_route
from dostuff.lib.model import completion_kwargs, current_session_id, get_litellm_model, uses_explicit_prompt_caching
from dostuff.setup_wizard import missing_api_keys


@pytest.fixture
def model(tmp_path, monkeypatch):
    """Make `name` the active model (no config.yaml, so its prefix gives the provider)."""
    monkeypatch.setattr(config_module.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)

    def use(name: str) -> None:
        monkeypatch.setenv("MODEL", name)
        reset_config()
    yield use
    reset_config()


@pytest.mark.parametrize("provider, model_id, endpoint", [
    ("opencode", "claude-sonnet-5-5", "anthropic"),
    ("opencode", "qwen3.8-flash", "anthropic"),
    ("opencode", "qwen3.8-max", "chat"),
    ("opencode-go", "qwen3.8-max", "anthropic"),
    ("opencode-go", "minimax-m3", "anthropic"),
    ("opencode", "minimax-m3", "chat"),
    ("opencode", "gpt-5.5", "responses"),
    ("opencode", "grok-4.7", "responses"),
    ("opencode", "gemini-3.5-flash", "google"),
    ("opencode", "big-pickle", "chat"),
])
def test_endpoint_for(provider, model_id, endpoint):
    assert endpoint_for(provider, model_id) == endpoint


def test_litellm_route():
    assert litellm_route("opencode", "claude-opus-5-5") == ("anthropic/claude-opus-5-5", "https://opencode.ai/inference/anthropic")
    assert litellm_route("opencode", "gpt-5.5") == ("openai/responses/gpt-5.5", "https://opencode.ai/inference/openai/v1")
    assert litellm_route("opencode-go", "kimi-k3") == ("openai/kimi-k3", "https://opencode.ai/inference/go/openai/v1")
    assert litellm_route("opencode", "gemini-3.5-flash") == ("gemini/gemini-3.5-flash", "https://opencode.ai/inference/google/v1beta")


def test_completion_kwargs_use_opencode_key_and_headers(model, monkeypatch):
    model("opencode-go/kimi-k3")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENCODE_API_KEY", "oc-key")
    current_session_id.set("sess-1")
    kwargs = completion_kwargs()
    assert kwargs["model"] == "openai/kimi-k3" and kwargs["api_base"].endswith("/go/openai/v1")
    assert kwargs["api_key"] == "oc-key"
    assert kwargs["extra_headers"]["x-opencode-session"] == "sess-1"
    assert kwargs["extra_headers"]["User-Agent"].startswith("dostuff/")
    assert get_litellm_model() == "openai/kimi-k3"


def test_prompt_caching_only_for_claude(model):
    model("opencode/claude-sonnet-5-5")
    assert uses_explicit_prompt_caching()
    model("opencode-go/minimax-m3")  # Anthropic endpoint, not Claude
    assert not uses_explicit_prompt_caching()


def test_missing_api_keys(monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    assert missing_api_keys("opencode/gpt-5.5") == ["OPENCODE_API_KEY"]
    monkeypatch.setenv("OPENCODE_API_KEY", "oc-key")
    assert missing_api_keys("opencode-go/glm-5.3") == []


def test_model_list(monkeypatch):
    data = {"data": [{"id": "kimi-k3"}, {"id": "jev-1.13"}, {"id": "test"}, {"id": "claude-haiku-5-5"}]}
    assert chat_model_ids(data) == ["kimi-k3", "claude-haiku-5-5"]
    monkeypatch.setattr(list_models, "_get_json", lambda url, what: data if url.endswith("/go/v1/models") else {})
    assert list_models.provider_models("opencode-go") == ["opencode-go/claude-haiku-5-5", "opencode-go/kimi-k3"]


def test_context_window_table(model):
    """OpenCode gateways publish no context limit, so the table in helpers/models/opencode.py is used."""
    t = token_limit_module
    for name, window in [
        ("opencode-go/deepseek-v4.1-flash", 1_000_000),  # the model that used to be assumed 128k
        ("opencode/deepseek-v4.1-flash", 1_000_000),
        ("opencode/minimax-m3", 512_000),  # same id, different window per gateway
        ("opencode-go/minimax-m3", 1_000_000),
        ("opencode/qwen3.8-max", 262_144),
        ("opencode-go/qwen3.8-max", 1_000_000),
    ]:
        model(name)
        assert t.get_model_token_limit() == window, name


def test_context_window_bare_litellm_fallback(model, monkeypatch):
    """An id missing from the table still resolves through litellm without the route prefix."""
    t = token_limit_module
    monkeypatch.setattr(t, "context_window", lambda provider, model_id: None)
    model("opencode-go/deepseek-v4-pro")
    assert t._litellm_limit("openai/deepseek-v4-pro") is None  # the routed name is unknown to litellm
    assert t._litellm_limit("deepseek-v4-pro") == 1_000_000  # the bare id is not
    assert t.get_model_token_limit() == 1_000_000


def test_context_window_configured_wins(model, monkeypatch):
    model("opencode-go/deepseek-v4.1-flash")
    monkeypatch.setattr(token_limit_module.get_config(), "context_window", 32_000)
    assert token_limit_module.get_model_token_limit() == 32_000


def test_context_window_unknown_warns_once(model, monkeypatch):
    t = token_limit_module
    emitted = []
    monkeypatch.setattr("dostuff.helpers.ui.emit.emit", lambda message, msg_type="system": emitted.append(message))
    monkeypatch.setattr(t, "_warned_models", set())
    model("opencode-go/not-a-real-model")
    assert t.get_model_token_limit() == t.FALLBACK_CONTEXT_WINDOW
    assert t.get_model_token_limit() == t.FALLBACK_CONTEXT_WINDOW
    assert len(emitted) == 1 and "128,000" in emitted[0]
