"""MCP keys stay in `.env`: `${NAME}` in a server's `headers` and `env` is filled in when it connects."""
import asyncio
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dostuff.cli import app
from dostuff.config import reset_config
from dostuff.helpers.config_secrets import find_secrets
from dostuff.helpers.mcp.server_config import expand_env_references, key_warnings, missing_env_references
from dostuff.helpers.ui.doctor import doctor_checks
from dostuff.lib.mcp.mcp_client import MCPClient
from dostuff.tools import settings
from dostuff.tools.mcp.call_mcp_tool import _impl_call_mcp_tool

SERVER = Path(__file__).with_name("mcp_echo_server.py")
HTTP_CONFIG = {"url": "https://x/mcp", "headers": {"Authorization": "Bearer ${SEARCH_API_KEY}"}}


class NullRegistry:
    async def register_tools(self, server_name, tools):
        pass


def client() -> MCPClient:
    return MCPClient(registration_store=None, tool_registry_store=NullRegistry())  # type: ignore[arg-type]


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".dostuff").mkdir(parents=True)
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(proj)
    monkeypatch.setenv("SEARCH_API_KEY", "")  # so loading .env in a test is undone after it
    monkeypatch.delenv("SEARCH_API_KEY")
    reset_config()
    yield home
    reset_config()


def test_expand_fills_in_each_reference(monkeypatch):
    monkeypatch.setenv("SEARCH_API_KEY", "abc")
    assert expand_env_references({"Authorization": "Bearer ${SEARCH_API_KEY}", "X-Plain": "$SEARCH_API_KEY", "Retries": 3}) == {
        "Authorization": "Bearer abc", "X-Plain": "$SEARCH_API_KEY", "Retries": 3}
    assert expand_env_references(None) == {}


@pytest.mark.parametrize("value", [None, ""])
def test_an_unset_or_empty_variable_is_missing(monkeypatch, value):
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    if value is not None:
        monkeypatch.setenv("SEARCH_API_KEY", value)
    assert missing_env_references(HTTP_CONFIG) == ["SEARCH_API_KEY"]


def test_a_key_value_is_not_expanded_twice(monkeypatch):
    monkeypatch.setenv("SEARCH_API_KEY", "${OTHER}")
    monkeypatch.setenv("OTHER", "leak")
    assert expand_env_references({"a": "${SEARCH_API_KEY}"}) == {"a": "${OTHER}"}


def test_missing_variable_stops_the_connection_and_names_it(monkeypatch):
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    mcp = client()
    with pytest.raises(ValueError, match=r"SEARCH_API_KEY not set: add it to \.dostuff/\.env"):
        asyncio.run(mcp.connect_to_server("search", transport="http", **HTTP_CONFIG))
    assert "search" not in mcp.server_meta and not mcp._connections


def test_http_server_gets_the_real_header(monkeypatch):
    monkeypatch.setenv("SEARCH_API_KEY", "abc")
    sent = {}
    mcp = client()

    async def serve(server_name, transport, url, req_headers, auth, command, args, env, ready, stop):
        sent.update(req_headers)
        ready.set_exception(RuntimeError("no server here"))

    monkeypatch.setattr(mcp, "_serve", serve)
    with pytest.raises(RuntimeError):
        asyncio.run(mcp.connect_to_server("search", transport="http", **HTTP_CONFIG))
    assert sent == {"Authorization": "Bearer abc"}


def test_stored_config_keeps_the_name_not_the_key(monkeypatch):
    monkeypatch.setenv("SEARCH_API_KEY", "abc")
    mcp = client()

    async def serve(server_name, transport, url, req_headers, auth, command, args, env, ready, stop):
        assert mcp.server_meta[server_name]["headers"] == HTTP_CONFIG["headers"]
        ready.set_exception(RuntimeError("no server here"))

    monkeypatch.setattr(mcp, "_serve", serve)
    with pytest.raises(RuntimeError):
        asyncio.run(mcp.connect_to_server("search", transport="http", **HTTP_CONFIG))


def test_stdio_server_receives_the_value_under_the_name_it_expects(monkeypatch):
    pytest.importorskip("mcp.server.mcpserver")
    monkeypatch.setenv("MY_WORK_KEY", "abc")

    async def scenario():
        mcp = client()
        try:
            await asyncio.wait_for(mcp.connect_to_server(
                "echo", command=sys.executable, args=[str(SERVER)], env={"SERVER_TOKEN": "${MY_WORK_KEY}"}), timeout=60)
            assert await _impl_call_mcp_tool("echo__read_env", {"name": "SERVER_TOKEN"}, mcp) == "abc"
        finally:
            await mcp.cleanup()

    asyncio.run(scenario())


@pytest.mark.parametrize("cfg", [
    {"headers": {"Authorization": "Bearer ${KEY}"}},
    {"headers": {"X-Api-Key": "${KEY}"}},
    {"env": {"GITHUB_TOKEN": "${KEY}"}},
    {"env": {"MAX_TOKENS": "4096"}},
])
def test_a_reference_is_not_a_secret(cfg):
    assert find_secrets({"s": cfg}) == []


@pytest.mark.parametrize("cfg, where", [
    ({"headers": {"Authorization": "Bearer abc123"}}, "s.headers.Authorization"),
    ({"headers": {"X-Api-Key": "abc123"}}, "s.headers.X-Api-Key"),
    ({"env": {"GITHUB_TOKEN": "abc123"}}, "s.env.GITHUB_TOKEN"),
    ({"args": ["--key", "sk-" + "a" * 30]}, "s.args[1]"),
])
def test_plain_keys_are_named_and_never_shown(cfg, where):
    (warning,) = key_warnings({"s": cfg})
    assert where in warning and "abc123" not in warning and "aaaa" not in warning


def test_read_settings_shows_the_name_and_hides_a_key(home):
    settings.set_mcp_server("search", url="https://x/mcp", headers={"Authorization": "Bearer ${SEARCH_API_KEY}", "X-Key": "abc123"})
    shown = settings.read_settings()
    assert "Bearer ${SEARCH_API_KEY}" in shown and "abc123" not in shown


def test_set_mcp_server_warns_about_an_unset_variable_and_a_plain_key(home):
    result = settings.set_mcp_server("search", url="https://x/mcp", headers={"Authorization": "Bearer ${SEARCH_API_KEY}"})
    assert "SEARCH_API_KEY not set" in result
    result = settings.set_mcp_server("search", url="https://x/mcp", headers={"Authorization": "Bearer abc123"})
    assert "search.headers.Authorization in the MCP config looks like a key" in result
    assert "abc123" not in result


def test_config_editor_warns(home):
    text = json.dumps({"mcpServers": {"search": HTTP_CONFIG}})
    assert any("SEARCH_API_KEY not set" in w for w in settings.check_settings_text("mcp", text))


def test_doctor_reports_a_variable_until_it_is_set(home, monkeypatch):
    (home / ".dostuff" / "mcp_config.json").write_text(json.dumps({"mcpServers": {"search": HTTP_CONFIG}}), encoding="utf-8")
    assert any("SEARCH_API_KEY not set" in c.label and c.ok is False for c in doctor_checks())
    monkeypatch.setenv("SEARCH_API_KEY", "abc")
    assert not any("SEARCH_API_KEY" in c.label for c in doctor_checks())


def test_mcp_add_keeps_the_reference_and_warns(home):
    result = CliRunner().invoke(app, ["mcp", "add", "search", "--url", "https://x/mcp", "-H", "Authorization: Bearer ${SEARCH_API_KEY}"])
    assert result.exit_code == 0 and "SEARCH_API_KEY not set" in result.output
    saved = json.loads((home / ".dostuff" / "mcp_config.json").read_text())["mcpServers"]["search"]
    assert saved["headers"] == {"Authorization": "Bearer ${SEARCH_API_KEY}"}


def test_mcp_add_sees_a_variable_in_env_file(home):
    (home / ".dostuff" / ".env").write_text("SEARCH_API_KEY=abc\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["mcp", "add", "search", "--url", "https://x/mcp", "-H", "Authorization: Bearer ${SEARCH_API_KEY}"])
    assert result.exit_code == 0 and "not set" not in result.output
