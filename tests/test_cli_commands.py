"""CLI: resume, sessions (list/rm/prune/rename), export, mcp add/list/remove, headless -p."""
import asyncio
import datetime
import json
import os
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".dostuff").mkdir(parents=True)
    proj = tmp_path / "proj"
    other = tmp_path / "other"
    proj.mkdir()
    other.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from dostuff import config as config_module
    config_module.reset_config()
    from dostuff.lib.memory.session_store import SQLiteSessionStore

    async def seed():
        store = SQLiteSessionStore(db_path=str(home / ".dostuff" / "data" / "sessions.db"))
        (home / ".dostuff" / "data").mkdir(parents=True, exist_ok=True)
        os.chdir(proj)
        for sid, prompt in (("aaa111", "write a tiger file"), ("aab222", "fix the tests")):
            await store.append(sid, {"role": "user", "content": prompt})
            await store.append(sid, {"role": "assistant", "content": f"done: {prompt}",
                                     "tool_calls": [{"function": {"name": "write_file", "arguments": '{"path": "t.txt"}'}}]})
            await store.append(sid, {"role": "tool", "name": "write_file", "content": "wrote t.txt"})
        await store.save_session_meta("empty1")  # what older versions left on every launch
        os.chdir(other)
        await store.append("bbb333", {"role": "user", "content": "elsewhere"})
        await store.close()
    asyncio.run(seed())
    db = home / ".dostuff" / "data" / "sessions.db"
    conn = sqlite3.connect(db)
    old = (datetime.datetime.now() - datetime.timedelta(days=40)).isoformat()
    conn.execute("UPDATE sessions SET last_used = ? WHERE session_id = 'bbb333'", (old,))
    conn.execute("UPDATE sessions SET last_used = ? WHERE session_id = 'aaa111'", ((datetime.datetime.now() - datetime.timedelta(hours=1)).isoformat(),))
    conn.commit()
    conn.close()
    monkeypatch.chdir(proj)
    from dostuff.cli import app
    yield {"app": app, "home": home, "proj": proj, "other": other}
    config_module.reset_config()


def _ids(output):
    return [line.split()[0] for line in output.splitlines() if line[:3] in ("aaa", "aab", "bbb", "emp")]


def test_sessions_lists_this_directory(env):
    result = runner.invoke(env["app"], ["sessions"])
    assert result.exit_code == 0, result.output
    assert _ids(result.output) == ["aab222", "aaa111"]  # newest first, no empty or other-dir sessions
    assert "fix the tests" in result.output

    result = runner.invoke(env["app"], ["sessions", "--all"])
    assert set(_ids(result.output)) == {"aab222", "aaa111", "bbb333"}
    result = runner.invoke(env["app"], ["sessions", "-n", "1"])
    assert _ids(result.output) == ["aab222"] and "1 more" in result.output

    os.chdir(env["home"])
    result = runner.invoke(env["app"], ["sessions"])
    assert "No sessions in" in result.output


def test_resume_picks_latest_here_or_by_prefix(env, monkeypatch):
    from dostuff import cli
    launched = []
    monkeypatch.setattr(cli, "_launch", lambda sid: launched.append(sid))
    assert runner.invoke(env["app"], ["resume"]).exit_code == 0
    assert runner.invoke(env["app"], ["resume", "bbb"]).exit_code == 0
    assert launched == ["aab222", "bbb333"]

    result = runner.invoke(env["app"], ["resume", "aa"])  # ambiguous prefix
    assert result.exit_code == 1 and "matches 2 sessions" in result.output
    os.chdir(env["home"])
    result = runner.invoke(env["app"], ["resume"])
    assert result.exit_code == 1 and "No sessions in" in result.output


def test_rename_rm_and_prune(env):
    app = env["app"]
    assert runner.invoke(app, ["sessions", "rename", "aaa", "Tiger", "work"]).exit_code == 0
    assert "Tiger work" in runner.invoke(app, ["sessions"]).output

    checkpoints = env["home"] / ".dostuff" / "data" / "checkpoints" / "aab222"
    checkpoints.mkdir(parents=True)
    result = runner.invoke(app, ["sessions", "rm", "aab"], input="n\n")
    assert result.exit_code == 1 and "aab222" in runner.invoke(app, ["sessions"]).output
    result = runner.invoke(app, ["sessions", "rm", "aab", "-y"])
    assert result.exit_code == 0 and "Deleted 1" in result.output
    assert "aab222" not in runner.invoke(app, ["sessions", "--all"]).output
    assert not checkpoints.exists()

    result = runner.invoke(app, ["sessions", "prune", "--dry-run"])
    assert "1 session(s) to delete: 1 empty" in result.output and "empty1" in result.output
    result = runner.invoke(app, ["sessions", "prune", "--older-than", "30d", "-y"])
    assert result.exit_code == 0 and "Deleted 2" in result.output
    assert set(_ids(runner.invoke(app, ["sessions", "--all"]).output)) == {"aaa111"}
    assert runner.invoke(app, ["sessions", "prune"]).output.strip() == "Nothing to prune."
    assert runner.invoke(app, ["sessions", "prune", "--older-than", "soon"]).exit_code != 0


def test_export(env):
    result = runner.invoke(env["app"], ["export", "aaa", "-o", "-"])
    assert result.exit_code == 0, result.output
    md = result.output
    assert md.startswith("# Session aaa111")
    assert "## User\n\nwrite a tiger file" in md and "**Tool call:** `write_file`" in md
    assert '"path": "t.txt"' in md and "wrote t.txt" in md

    result = runner.invoke(env["app"], ["export"])  # latest here, to a file
    assert result.exit_code == 0
    assert (env["proj"] / "dostuff-session-aab222.md").read_text(encoding="utf-8").startswith("# Session aab222")


def test_mcp_add_list_remove(env):
    app = env["app"]
    result = runner.invoke(app, ["mcp", "add", "github", "-e", "GITHUB_TOKEN=abc", "--", "uvx", "mcp-server-github", "--verbose"])
    assert result.exit_code == 0, result.output
    data = json.loads((env["home"] / ".dostuff" / "mcp_config.json").read_text())
    assert data["mcpServers"]["github"] == {"transport": "stdio", "command": "uvx", "args": ["mcp-server-github", "--verbose"], "env": {"GITHUB_TOKEN": "abc"}}

    assert runner.invoke(app, ["mcp", "add", "github", "--", "npx", "x"]).exit_code == 1  # exists
    assert runner.invoke(app, ["mcp", "add", "both", "--url", "https://x", "--", "npx"]).exit_code != 0
    result = runner.invoke(app, ["mcp", "add", "remote", "--url", "https://mcp.example.com", "-H", "Authorization: Bearer t", "--project"])
    assert result.exit_code == 0, result.output
    proj_data = json.loads((env["proj"] / ".dostuff" / "mcp_config.json").read_text())
    assert proj_data["mcpServers"]["remote"] == {"transport": "http", "url": "https://mcp.example.com", "headers": {"Authorization": "Bearer t"}}

    # The project file is merged over the global one
    from dostuff.helpers.mcp.load_mcp_config import load_mcp_config
    assert set(load_mcp_config()) == {"github", "remote"}
    listing = runner.invoke(app, ["mcp", "list"]).output
    assert "github" in listing and "uvx mcp-server-github" in listing and "https://mcp.example.com" in listing

    assert runner.invoke(app, ["mcp", "remove", "remote"]).exit_code == 1  # it's in the project file
    assert runner.invoke(app, ["mcp", "remove", "remote", "--project"]).exit_code == 0
    assert runner.invoke(app, ["mcp", "remove", "github"]).exit_code == 0
    assert "No MCP servers configured" in runner.invoke(app, ["mcp", "list"]).output


def test_models_add_and_list(env, monkeypatch):
    import yaml
    from dostuff import config as config_module
    app = env["app"]
    cfg = env["home"] / ".dostuff" / "config.yaml"
    monkeypatch.delenv("MY_GROQ", raising=False)
    # No config yet: created from the template, with this model as the only (active) entry
    result = runner.invoke(app, ["models", "add", "anthropic/claude-sonnet-5-5"])
    assert result.exit_code == 0 and "now the default model" in result.output, result.output
    assert "permissions:" in cfg.read_text(encoding="utf-8")

    result = runner.invoke(app, ["models", "add", "llama-3.3-70b", "-P", "groq", "--api-key-env", "MY_GROQ"])
    assert "Added 'groq/llama-3.3-70b'" in result.output and "Switch to it with /model" in result.output
    assert "MY_GROQ=..." in result.output
    assert runner.invoke(app, ["models", "add", "groq/llama-3.3-70b"]).exit_code == 1  # already listed
    assert runner.invoke(app, ["models", "add", "gpt-4o"]).exit_code != 0  # no provider
    result = runner.invoke(app, ["models", "add", "groq/llama-3.3-70b", "--active"])
    assert "already in" in result.output and "now the default model" in result.output
    models = yaml.safe_load(cfg.read_text(encoding="utf-8"))["models"]
    assert [(m["name"], m["active"]) for m in models] == [("anthropic/claude-sonnet-5-5", False), ("groq/llama-3.3-70b", True)]
    assert models[1]["api_key_env"] == "MY_GROQ"

    config_module.reset_config()
    listing = runner.invoke(app, ["models", "list"]).output
    assert "groq/llama-3.3-70b" in listing and "default" in listing and "MODEL=openai/gpt-4o-mini is set" in listing

    # --project writes ./.dostuff/config.yaml, whose list replaces the global one here
    result = runner.invoke(app, ["models", "add", "openai/gpt-4o", "--project"])
    assert result.exit_code == 0 and "replaces the one in ~/.dostuff/config.yaml" in result.output
    assert yaml.safe_load((env["proj"] / ".dostuff" / "config.yaml").read_text(encoding="utf-8"))["models"] == [
        {"name": "openai/gpt-4o", "provider": "openai", "active": True}]


def test_models_add_without_a_name_picks_from_the_list(env, monkeypatch):
    import yaml
    from dostuff import setup_wizard
    from dostuff.helpers.models import list_models
    monkeypatch.setattr(list_models, "provider_models",
                        lambda p: ["anthropic/claude-haiku-5-5", "anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5"])
    app = env["app"]
    assert runner.invoke(app, ["models", "add"]).exit_code != 0  # no terminal for the picker
    monkeypatch.setattr(setup_wizard, "can_prompt", lambda: True)
    result = runner.invoke(app, ["models", "add"], input="2\nopus\n1\n")  # provider, filter, pick
    assert result.exit_code == 0, result.output
    assert "1 matching" in result.output and "Added 'anthropic/claude-opus-5-5'" in result.output
    result = runner.invoke(app, ["models", "add", "-P", "anthropic"], input="my-model\ny\n")  # not listed: confirm
    assert "Added 'anthropic/my-model'" in result.output
    names = [m["name"] for m in yaml.safe_load((env["home"] / ".dostuff" / "config.yaml").read_text(encoding="utf-8"))["models"]]
    assert names == ["anthropic/claude-opus-5-5", "anthropic/my-model"]


def test_provider_models_lists(monkeypatch):
    import httpx
    import litellm
    from dostuff.helpers.models import list_models
    monkeypatch.setattr(litellm, "models_by_provider", {"openai": [
        "gpt-4o", "gpt-4o-2024-08-06", "openai/gpt-5", "text-embedding-3-small", "gpt-4o-realtime-preview", "mini-2024-01-01"]})
    monkeypatch.setattr(litellm, "model_cost", {
        "gpt-4o": {"mode": "chat"}, "gpt-4o-2024-08-06": {"mode": "chat"}, "openai/gpt-5": {"mode": "chat"},
        "text-embedding-3-small": {"mode": "embedding"}, "gpt-4o-realtime-preview": {"mode": "chat"},
        "mini-2024-01-01": {"mode": "chat"}})
    # The dated gpt-4o snapshot goes (its alias is listed); one with no alias stays
    assert list_models.provider_models("openai") == ["openai/gpt-4o", "openai/gpt-5", "openai/mini-2024-01-01"]

    def fake_get(url, timeout):
        if "openrouter" in url:
            body = {"data": [{"id": "z/text", "architecture": {"output_modalities": ["text"]}},
                             {"id": "a/img", "architecture": {"output_modalities": ["image"]}}, {"id": "b/plain"}]}
        else:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", fake_get)
    assert list_models.provider_models("openrouter") == ["openrouter/b/plain", "openrouter/z/text"]
    with pytest.raises(list_models.ModelListError, match="is it running"):
        list_models.provider_models("ollama")


def test_add_model_entry_keeps_comments_and_layout(tmp_path):
    import yaml
    from dostuff.helpers.models.add_model_entry import ModelConfigError, add_model_entry
    path = tmp_path / "config.yaml"
    path.write_text(
        "# top comment\nmodels:\n    - name: a/one  # first\n      provider: a\n      active: true\n\n"
        "loop:\n  max_iterations: 5  # keep me\n", encoding="utf-8")
    assert add_model_entry(path, "b/two", "b", active=True) is True
    text = path.read_text(encoding="utf-8")
    assert "# top comment" in text and "# first" in text and "# keep me" in text
    assert '    - name: "b/two"' in text  # matches the existing indentation
    data = yaml.safe_load(text)
    assert [(m["name"], m["active"]) for m in data["models"]] == [("a/one", False), ("b/two", True)]
    assert add_model_entry(path, "a/one", "a", active=True) is False  # existing: only made active
    assert [m["active"] for m in yaml.safe_load(path.read_text(encoding="utf-8"))["models"]] == [True, False]
    with pytest.raises(ModelConfigError):
        add_model_entry(path, "a/one", "a")

    # Flow style can't be edited line by line: the block is rewritten, the rest is kept
    path.write_text("models: []\nloop:\n  max_iterations: 5\n", encoding="utf-8")
    add_model_entry(path, "c/three", "c")
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "models": [{"name": "c/three", "provider": "c", "active": False}], "loop": {"max_iterations": 5}}


@pytest.fixture
def headless(env, monkeypatch):
    from dostuff.helpers.mcp import load_mcp_config as mcp_module
    from dostuff.lib.memory.episodic_memory_store import EpisodicMemoryStore
    from dostuff.lib.memory.semantic_memory_store import SemanticMemoryStore
    from dostuff.agent import loop as loop_module

    async def no_memories(self, *args, **kwargs):
        return []
    monkeypatch.setattr(SemanticMemoryStore, "query", no_memories)
    monkeypatch.setattr(EpisodicMemoryStore, "query", no_memories)
    monkeypatch.setattr(mcp_module, "load_mcp_config", lambda: {})
    calls = []

    async def fake_loop(session_id, turn_id, user_text, instructions, mcp_client, working_history, turn_type,
                        current_session_history, steps_history, store, adapter=None, permissions=None, **kwargs):
        calls.append({"text": user_text, "history": list(working_history), "mode": permissions.mode})
        reply = await adapter.ask("Allow write_file?")
        adapter.emit("tool_call", "write_file(path=a.txt)")
        adapter.emit("usage", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        text = f"answer ({reply.answer})"
        await store.append(session_id, {"role": "assistant", "content": text})
        return text
    monkeypatch.setattr(loop_module, "loop", fake_loop)
    return calls


def test_print_mode(env, headless):
    app = env["app"]
    result = runner.invoke(app, ["-p", "say hi"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "answer (no)"  # no terminal: the approval is denied
    assert "● write_file" in result.stderr and "were denied" in result.stderr

    result = runner.invoke(app, ["-p", "summarize", "--json", "-q", "--permission-mode", "auto"], input="file contents")
    assert result.exit_code == 0, result.output
    out = json.loads(result.stdout)
    assert out["result"] == "answer (no)" and out["usage"]["total_tokens"] == 15
    assert headless[-1]["mode"] == "auto"
    assert headless[-1]["text"] == "summarize\n\n<stdin>\nfile contents\n</stdin>"
    assert "●" not in result.stderr and "session" not in result.stderr  # --quiet: only the denial warning

    # --session continues a stored session with its history
    result = runner.invoke(app, ["-p", "and then?", "--session", out["session_id"]])
    assert result.exit_code == 0
    assert [s["role"] for s in headless[-1]["history"]] == ["user", "assistant", "user"]

    assert runner.invoke(app, ["-p", "x", "--session", "nope"]).exit_code == 1
    assert runner.invoke(app, ["-p", "x", "--permission-mode", "yolo"]).exit_code == 2
    assert runner.invoke(app, ["--json"]).exit_code == 2
