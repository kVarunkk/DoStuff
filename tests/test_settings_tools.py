"""Settings tools: the agent changes config.yaml and the MCP servers; writes always ask."""
import asyncio
import json
from pathlib import Path

import pytest

from dostuff.agent.permissions import PermissionConfig, SessionPermissions, check
from dostuff.agent.run_tool import authorize_tool
from dostuff.tools import settings


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".dostuff").mkdir(parents=True)
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(proj)
    from dostuff import config as config_module
    config_module.reset_config()
    yield home
    config_module.reset_config()


class Recorder:
    def __init__(self):
        self.events = []

    def emit(self, event_type, data):
        self.events.append((event_type, data))


def test_edit_config_keeps_comments_validates_and_reloads(home):
    from dostuff.config import get_config
    path = home / ".dostuff" / "config.yaml"
    path.write_text("# my notes\nloop:\n  max_iterations: 100  # cap\n", encoding="utf-8")
    assert get_config().max_iterations == 100
    adapter = Recorder()

    result = settings.edit_config("max_iterations: 100", "max_iterations: 40", adapter=adapter)
    assert "+  max_iterations: 40  # cap" in result
    assert path.read_text(encoding="utf-8") == "# my notes\nloop:\n  max_iterations: 40  # cap\n"
    assert get_config().max_iterations == 40  # applied without a restart
    assert adapter.events == [("settings_changed", "config")]

    # Empty old_str appends a section; unknown status bar items come back as warnings
    result = settings.edit_config("", "ui:\n  status_bar: [mode, cost, timer, bogus]")
    assert "unknown ui.status_bar item 'bogus'" in result and "'timer'" not in result
    assert get_config().status_bar_items == ["mode", "cost"]

    with pytest.raises(ValueError, match="invalid YAML"):
        settings.edit_config("loop:", "loop: [")
    with pytest.raises(ValueError, match="occurs 0 times"):
        settings.edit_config("nope", "x")
    assert "max_iterations: 40" in path.read_text(encoding="utf-8")  # untouched by the failed edits

    # Project scope creates ./.dostuff/config.yaml
    settings.edit_config("", "memory:\n  episodic_scope: user", scope="project")
    assert get_config().episodic_scope == "user"


def test_mcp_servers_add_update_remove_with_redacted_secrets(home):
    path = home / ".dostuff" / "mcp_config.json"
    adapter = Recorder()
    settings.set_mcp_server("github", command=["uvx", "mcp-server-github"], env={"GITHUB_TOKEN": "secret"}, adapter=adapter)
    assert json.loads(path.read_text())["mcpServers"]["github"] == {
        "transport": "stdio", "command": "uvx", "args": ["mcp-server-github"], "env": {"GITHUB_TOKEN": "secret"}}
    assert adapter.events == [("settings_changed", "mcp")]

    shown = settings.read_settings()
    assert "secret" not in shown and '"GITHUB_TOKEN": "<redacted>"' in shown
    assert "Reference: every config.yaml option" in shown

    # Passing the redacted value back keeps the stored secret
    result = settings.set_mcp_server("github", command=["uvx", "mcp-server-github", "--v2"], env={"GITHUB_TOKEN": "<redacted>"})
    assert "secret" not in result
    assert json.loads(path.read_text())["mcpServers"]["github"]["env"] == {"GITHUB_TOKEN": "secret"}
    with pytest.raises(ValueError, match="no value is stored"):
        settings.set_mcp_server("github", command=["x"], env={"OTHER": "<redacted>"})
    with pytest.raises(ValueError, match="not both"):
        settings.set_mcp_server("bad", command=["x"], url="https://x")

    settings.remove_mcp_server("github")
    assert json.loads(path.read_text())["mcpServers"] == {}
    with pytest.raises(ValueError, match="No server 'github'"):
        settings.remove_mcp_server("github")


def test_settings_changes_ask_in_every_mode_without_always(home):
    auto = PermissionConfig(mode="auto")
    decision = check("edit_config", "settings", {"old_str": "a", "new_str": "b"}, None, auto)
    assert decision.action == "ask" and decision.approval_keys == []
    # write_file / edit_file into a .dostuff folder can't get around it
    assert check("edit_file", "write", {"path": ".dostuff/config.yaml"}, None, auto).action == "ask"
    assert check("edit_file", "write", {"path": "src/app.py"}, None, auto).action == "allow"
    assert check("read_settings", "read", {}, None, auto).action == "allow"

    (home / ".dostuff" / "config.yaml").write_text("loop:\n  max_iterations: 100\n", encoding="utf-8")
    prompts = []

    class Asker:
        async def ask(self, message, always_label=""):
            prompts.append((message, always_label))
            return "a"  # "always" isn't offered; answering it still only allows this call

    session = SessionPermissions(mode="auto")
    args = {"old_str": "100", "new_str": "50"}
    assert asyncio.run(authorize_tool("edit_config", args, adapter=Asker(), permissions=session)) is None
    message, always_label = prompts[0]
    assert "-  max_iterations: 100" in message and "+  max_iterations: 50" in message and always_label == ""
    assert session.approved == set()
    asyncio.run(authorize_tool("set_mcp_server", {"name": "git", "command": ["uvx", "mcp-server-git"]},
                               adapter=Asker(), permissions=session))
    assert '"runs": "uvx mcp-server-git"' in prompts[1][0]


def test_settings_tools_are_not_given_to_subagents():
    from dostuff.tools.delegate_to_subagent import resolve_subagent_tools
    allowed, dropped = resolve_subagent_tools(["read_file", "edit_config", "set_mcp_server", "read_settings"])
    assert allowed == ["read_file"]
    assert set(dropped) == {"edit_config", "set_mcp_server", "read_settings"}


def test_save_settings_file_validates_keeps_crlf_and_applies(home):
    from dostuff.config import get_config
    path = home / ".dostuff" / "config.yaml"
    path.write_bytes(b"loop:\r\n  max_iterations: 5\r\n")
    recorder = Recorder()

    with pytest.raises(ValueError, match="Invalid YAML"):
        settings.save_settings_file("config", "user", "loop: [x\n", recorder)
    with pytest.raises(ValueError, match="mapping"):
        settings.save_settings_file("config", "user", "- a\n", recorder)
    assert path.read_bytes() == b"loop:\r\n  max_iterations: 5\r\n" and recorder.events == []

    summary = settings.save_settings_file("config", "user", "loop:\n  max_iterations: 9\nbogus: 1", recorder)
    assert path.read_bytes() == b"loop:\r\n  max_iterations: 9\r\nbogus: 1\r\n"
    assert "(+2 / -1 lines)" in summary and "unknown top-level key 'bogus'" in summary
    assert recorder.events == [("settings_changed", "config")] and get_config().max_iterations == 9


def test_save_mcp_settings_file(home):
    with pytest.raises(ValueError, match="Invalid JSON"):
        settings.save_settings_file("mcp", "project", "{", None)
    with pytest.raises(ValueError, match="mcpServers"):
        settings.save_settings_file("mcp", "project", '{"mcpServers": []}', None)
    summary = settings.save_settings_file("mcp", "project", '{"mcpServers": {"x": {"env": {"T": "secret"}}}}', None)
    assert "neither a command nor a url" in summary and "secret" not in summary
    saved = json.loads((Path.cwd() / ".dostuff" / "mcp_config.json").read_text(encoding="utf-8"))
    assert saved["mcpServers"]["x"]["env"]["T"] == "secret"


def test_new_settings_text_is_valid_and_empty():
    import yaml
    assert yaml.safe_load(settings.new_settings_text("config", "user")) is None
    assert settings.check_settings_text("mcp", settings.new_settings_text("mcp", "user")) == []
