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
    monkeypatch.chdir(tmp_path)  # before the os.chdir calls below, so teardown returns to the original cwd
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


def test_sessions_include_this_repos_worktrees(env):
    db = env["home"] / ".dostuff" / "data" / "sessions.db"
    conn = sqlite3.connect(db)
    wt = env["proj"] / ".dostuff" / "worktrees" / "fork-bbb3"
    conn.execute("UPDATE sessions SET working_dir = ? WHERE session_id = 'bbb333'", (str(wt),))
    conn.commit()
    conn.close()
    result = runner.invoke(env["app"], ["sessions"])
    assert result.exit_code == 0, result.output
    assert _ids(result.output) == ["aab222", "aaa111", "bbb333"]
    assert "[worktree fork-bbb3] elsewhere" in result.output


def test_resume_picks_latest_here_or_by_prefix(env, monkeypatch):
    from dostuff import cli
    launched = []
    monkeypatch.setattr(cli, "_launch", lambda sid, worktree=None: launched.append(sid))
    assert runner.invoke(env["app"], ["resume"]).exit_code == 0
    assert runner.invoke(env["app"], ["resume", "bbb"]).exit_code == 0
    assert launched == ["aab222", "bbb333"]

    result = runner.invoke(env["app"], ["resume", "aa"])  # ambiguous prefix
    assert result.exit_code == 1 and "matches 2 sessions" in result.output
    os.chdir(env["home"])
    result = runner.invoke(env["app"], ["resume"])
    assert result.exit_code == 1 and "No sessions in" in result.output


def test_continue_resumes_latest_here(env, monkeypatch):
    from dostuff import cli
    launched = []
    monkeypatch.setattr(cli, "_launch", lambda sid, worktree=None: launched.append(sid))
    assert runner.invoke(env["app"], ["-c"]).exit_code == 0
    assert runner.invoke(env["app"], ["--continue"]).exit_code == 0
    assert launched == ["aab222", "aab222"]
    result = runner.invoke(env["app"], ["-c", "--session", "aaa111"])
    assert result.exit_code == 2 and "can't be used together" in result.output
    os.chdir(env["home"])
    result = runner.invoke(env["app"], ["-c"])
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
        calls.append({"text": user_text, "history": list(working_history), "mode": permissions.mode,
                      "allow": permissions.allow, "max_iterations": kwargs.get("max_iterations")})
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


def test_print_mode_flags(env, headless):
    """--max-turns, --allowed-tools and --output-format stream-json."""
    app = env["app"]
    result = runner.invoke(app, ["-p", "go", "-q", "--max-turns", "3", "--allowed-tools", "write_file,npm test",
                                 "--allowed-tools", "web_fetch:docs.python.org", "--output-format", "stream-json"])
    assert result.exit_code == 0, result.output
    assert headless[-1]["max_iterations"] == 3
    assert headless[-1]["allow"] == ["write_file", "npm test", "web_fetch:docs.python.org"]
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert [e["type"] for e in events] == ["init", "tool_call", "usage", "result"]
    assert events[0]["session_id"] == events[-1]["session_id"] and events[1]["call"] == "write_file(path=a.txt)"
    assert events[-1]["result"] == "answer (no)" and events[-1]["is_error"] is False

    result = runner.invoke(app, ["-p", "go", "-q", "--output-format", "json"])
    assert json.loads(result.stdout)["result"] == "answer (no)"
    assert runner.invoke(app, ["-p", "go", "--output-format", "xml"]).exit_code == 2
    assert runner.invoke(app, ["-p", "go", "--json", "--output-format", "text"]).exit_code == 2
    assert runner.invoke(app, ["--max-turns", "2"]).exit_code == 2


# --- hooks (plan/2026-10-29-hooks, T3.1 / T3.4) ------------------------------------------------

def _write_hooks(home, hooks: dict) -> None:
    import yaml
    from dostuff import config as config_module
    (home / ".dostuff" / "config.yaml").write_text(yaml.safe_dump({"hooks": hooks}), encoding="utf-8")
    config_module.reset_config()


def test_headless_prompt_blocked_by_hook(env, headless, hook_cmd):
    blocker = hook_cmd("import sys; sys.stderr.write('not in this repo please'); sys.exit(2)")
    _write_hooks(env["home"], {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": blocker}]}]})
    result = runner.invoke(env["app"], ["-p", "push it", "--json"])
    assert result.exit_code == 2, result.output
    assert "Prompt blocked by a UserPromptSubmit hook: not in this repo please" in result.stderr
    assert headless == []  # the loop never ran
    session_id = json.loads(result.stdout)["session_id"]

    from dostuff.lib.memory.session_store import SQLiteSessionStore

    async def stored():
        store = SQLiteSessionStore(db_path=str(env["home"] / ".dostuff" / "data" / "sessions.db"))
        try:
            return await store.load(session_id)
        finally:
            await store.close()
    assert asyncio.run(stored()) == []


def test_headless_session_start_sources(env, headless, hook_cmd, tmp_path):
    log = tmp_path / "sources.log"
    logger = hook_cmd(f"""
        import json, sys
        source = json.load(sys.stdin)["source"]
        open({str(log)!r}, "a").write(source + "\\n")
        print("ctx-" + source)
    """)
    _write_hooks(env["home"], {"SessionStart": [{"hooks": [{"type": "command", "command": logger}]}]})
    result = runner.invoke(env["app"], ["-p", "go", "--json", "-q"])
    assert result.exit_code == 0, result.output
    assert log.read_text().split() == ["startup"]
    session_id = json.loads(result.stdout)["session_id"]

    result = runner.invoke(env["app"], ["-p", "and then?", "-q", "--session", session_id])
    assert result.exit_code == 0, result.output
    assert log.read_text().split() == ["startup", "resume"]


def test_headless_project_hooks(env, headless, hook_cmd, tmp_path, monkeypatch):
    import yaml
    from dostuff import config as config_module
    from dostuff.helpers.hooks import trust
    log = tmp_path / "ran.log"
    logger = hook_cmd(f"open({str(log)!r}, 'a').write('ran' + chr(10))")
    project_file = env["proj"] / ".dostuff" / "config.yaml"
    project_file.parent.mkdir(exist_ok=True)

    def project_hooks(**group):
        project_file.write_text(yaml.safe_dump({"hooks": {"UserPromptSubmit": [
            {**group, "hooks": [{"type": "command", "name": "guard", "command": logger}]}]}}), encoding="utf-8")
        config_module.reset_config()

    def runs():
        return len(log.read_text().splitlines()) if log.exists() else 0

    def run(*flags):
        monkeypatch.setattr(trust, "_decisions", {})  # a fresh process each time
        return runner.invoke(env["app"], ["-p", "go", *flags])

    # untrusted, not a guardrail: skipped with one warning, the turn still runs
    project_hooks()
    result = run()
    assert result.exit_code == 0, result.output
    assert runs() == 0 and len(headless) == 1
    assert result.stderr.count("project hooks") == 1 and "--trust-project-hooks" in result.stderr

    # a trust record from an earlier TUI session is honoured
    trust.remember(trust.trust_key(trust.project_entries(config_module.get_config().hooks)))
    assert run().exit_code == 0 and runs() == 1
    (env["home"] / ".dostuff" / "trusted_hooks.json").unlink()

    # --trust-project-hooks runs them for this run only
    assert run("--trust-project-hooks").exit_code == 0 and runs() == 2
    assert not (env["home"] / ".dostuff" / "trusted_hooks.json").exists()

    # an untrusted guardrail: refuse to start rather than fail open
    project_hooks(fail_closed=True)
    result = run()
    assert result.exit_code == 2, result.output
    assert len(headless) == 3 and runs() == 2
    assert "guard" in result.stderr and "--trust-project-hooks" in result.stderr
    assert run("--trust-project-hooks").exit_code == 0 and runs() == 3

    # CI forces color and a narrow terminal; Rich then splits the option name with escape codes
    help_text = runner.invoke(env["app"], ["--help"], env={"NO_COLOR": "1", "FORCE_COLOR": None,
                                                          "TERM": "dumb", "COLUMNS": "200"}).output
    assert "--trust-project-hooks" in help_text


# --- background shell T5.4: headless kills its jobs (plan/2026-10-31-background-shell) ---------

@pytest.fixture
def headless_jobs(headless, registry, monkeypatch, _py):
    """The fake loop starts a background sleeper the way the real one would; `raise_after` makes it
    raise once the job is running. Returns the list of started jobs."""
    from dostuff.agent import loop as loop_module
    from dostuff.tools.bash import jobs as jobs_module
    from dostuff.tools.bash.bash_command import bash_command
    monkeypatch.setattr(jobs_module, "_registry", registry)
    sleeper = _py("import time; time.sleep(30)")
    started: list = []
    state = {"raise_after": None}

    async def fake_loop(session_id, turn_id, user_text, instructions, mcp_client, working_history, turn_type,
                        current_session_history, steps_history, store, adapter=None, **kwargs):
        await bash_command(sleeper, run_in_background=True, background_id="srv", owner=session_id)
        job = registry.get(session_id, "srv")
        started.append(job)
        adapter.emit("job", {"id": job.id, "status": job.status, "code": job.exit_code})
        if state["raise_after"] is not None:
            raise state["raise_after"]
        return "answer"
    monkeypatch.setattr(loop_module, "loop", fake_loop)
    return started, state


def test_headless_kills_jobs(env, headless_jobs, _wait_dead):
    app = env["app"]
    started, _ = headless_jobs
    result = runner.invoke(app, ["-p", "serve"])
    assert result.exit_code == 0, result.output
    job = started[-1]
    assert _wait_dead(job.pid, 5) and job.status == "killed"
    assert f'⚙ stopped background job "srv" (pid {job.pid})' in result.stderr

    result = runner.invoke(app, ["-p", "serve", "-q", "--output-format", "stream-json"])
    events = [json.loads(line) for line in result.stdout.splitlines()]
    jobs = [e for e in events if e["type"] == "job"]
    assert [(e["id"], e["status"]) for e in jobs] == [("srv", "running"), ("srv", "killed")]
    assert "stopped background job" not in result.stderr
    assert _wait_dead(started[-1].pid, 5)


def test_headless_ctrl_c_kills_jobs(env, headless_jobs, _wait_dead):
    from dostuff.headless import run_headless
    started, state = headless_jobs
    state["raise_after"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(run_headless("serve", interactive=False, quiet=True))
    assert _wait_dead(started[-1].pid, 5) and started[-1].status == "killed"


# --- subagents rebuild T2.7: parallel runs never ask on the terminal at the same time ----------------

def test_headless_ask_lock(monkeypatch):
    import asyncio
    import builtins
    import threading
    import time

    from dostuff.headless import HeadlessAdapter
    log: list[str] = []
    lock = threading.Lock()

    def fake_input(prompt=""):
        with lock:
            log.append("enter")
        time.sleep(0.1)
        with lock:
            log.append("exit")
        return "y"
    monkeypatch.setattr(builtins, "input", fake_input)
    adapter = HeadlessAdapter(interactive=True, quiet=True)

    async def go():
        return await asyncio.gather(adapter.ask("[agent: a] Run x?"), adapter.ask("[agent: b] Run y?"))
    replies = asyncio.run(go())
    assert [r.answer for r in replies] == ["yes", "yes"]
    assert log == ["enter", "exit", "enter", "exit"]


# --- subagents rebuild T7.3–T7.5: headless agents (plan/2026-11-01-subagents/spec.md D17) ----------

@pytest.fixture
def headless_agents(env, agents, fake_model, monkeypatch):
    """`run_headless` with the real loop and a fake model (main agent and subagents); returns the FakeModel."""
    from dostuff.helpers.mcp import load_mcp_config as mcp_module
    from dostuff.lib.memory.episodic_memory_store import EpisodicMemoryStore
    from dostuff.lib.memory.semantic_memory_store import SemanticMemoryStore

    async def no_memories(self, *args, **kwargs):
        return []
    monkeypatch.setattr(SemanticMemoryStore, "query", no_memories)
    monkeypatch.setattr(EpisodicMemoryStore, "query", no_memories)
    monkeypatch.setattr(mcp_module, "load_mcp_config", lambda: {})
    return fake_model


def _agent_call(marker: str, description: str, call_id: str, background: bool = False):
    from conftest import fake_tool_call
    return fake_tool_call("agent", {"description": description, "prompt": f"[{marker}] {description}",
                                    "subagent_type": "general-purpose", "run_in_background": background}, call_id)


def _run_headless(prompt: str, **kw) -> int:
    from dostuff.headless import run_headless
    kw.setdefault("interactive", False)
    return asyncio.run(run_headless(prompt, **kw))


def test_headless_parallel_agents(env, headless_agents, agents, capsys):
    from conftest import FakeStep, fake_response, fake_tool_call
    fm = headless_agents
    fm.script("[M]", fake_response(tool_calls=[_agent_call(m, f"job {m}", f"c{m}") for m in "ABC"]),
              fake_response("all three done"))
    fm.script("[A]", fake_response(tool_calls=[fake_tool_call("write_file", {"path": "x.txt", "content": "x"})]),
              FakeStep(fake_response("report A"), delay=0.6))
    for m in "BC":
        fm.script(f"[{m}]", FakeStep(fake_response(f"report {m}"), delay=0.6))
    code = _run_headless("[M] fan out")
    out, err = capsys.readouterr()
    assert code == 0 and out.strip() == "all three done"
    runs = agents.runs()
    assert len(runs) == 3 and max(r.started for r in runs) < min(r.ended for r in runs)  # they overlapped
    # the approval prompt (denied: no terminal) names the agent
    assert "✗ Denied (no terminal to ask): [agent: job A" in err

    fm.script("[M]", fake_response(tool_calls=[_agent_call(m, f"job {m}", f"c{m}") for m in "BC"]),
              fake_response("two done"))
    code = _run_headless("[M] again", quiet=True, output_format="stream-json")
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    agent_events = [e for e in events if e["type"] == "agent"]
    assert sorted((e["description"], e["status"]) for e in agent_events) == [
        ("job B", "completed"), ("job B", "running"), ("job C", "completed"), ("job C", "running")]
    assert all(e["agent_type"] == "general-purpose" and e["background"] is False and e["id"] for e in agent_events)
    assert events[-1]["type"] == "result" and events[-1]["result"] == "two done"


@pytest.mark.parametrize("wake_on_end", [True, False])
def test_headless_waits_for_background(env, headless_agents, capsys, wake_on_end):
    from conftest import FakeStep, fake_response
    from dostuff import config as config_module
    from dostuff.helpers.agent.constants import AGENT_BACKGROUND
    if not wake_on_end:
        (env["home"] / ".dostuff" / "config.yaml").write_text("agents:\n  wake_on_end: false\n", encoding="utf-8")
        config_module.reset_config()
    fm = headless_agents
    fm.script("[M]", fake_response(tool_calls=[_agent_call("BG", "dig", "c1", background=True)]),
              fake_response("it is running"), fake_response("final: 42"))
    fm.script("[BG]", FakeStep(fake_response("bg report 42"), delay=0.3))
    code = _run_headless("[M] start it")
    out, err = capsys.readouterr()
    assert code == 0 and out.strip() == "final: 42"
    assert '⚙ agent "dig" (general-purpose) finished after' in err
    main_calls = fm.calls_for("[M]")
    assert len(main_calls) == 3
    assert "bg report 42" in str(main_calls[2]["messages"][-1]["content"])
    system = main_calls[0]["messages"][0]["content"]
    assert (AGENT_BACKGROUND.strip()[:40] in system) is wake_on_end  # D13: guidance only with the switch on


def test_headless_limit_stops_agents(env, headless_agents, agents, capsys):
    from conftest import FakeStep, fake_response
    from dostuff.helpers.agent.notices import WAKE_LIMIT
    fm = headless_agents
    for turn in range(WAKE_LIMIT + 1):  # the first turn, then one per wake-up: each starts another agent
        fm.script("[M]", fake_response(tool_calls=[_agent_call(f"B{turn}", f"bg{turn}", f"c{turn}", True)]),
                  fake_response(f"turn {turn} ends"))
    for turn in range(WAKE_LIMIT):
        fm.script(f"[B{turn}]", FakeStep(fake_response(f"report {turn}"), delay=0.1))
    fm.gate(f"[B{WAKE_LIMIT}]", fake_response("never"))  # still running when the limit is hit
    code = _run_headless("[M] keep going")
    out, err = capsys.readouterr()
    assert code == 0
    assert len(fm.calls_for("[M]")) == 2 * (WAKE_LIMIT + 1)
    last = next(r for r in agents.runs() if r.description == f"bg{WAKE_LIMIT}")
    assert last.status == "stopped"
    assert f'stopped agent "bg{WAKE_LIMIT}"' in err and f"{WAKE_LIMIT} automatic turns" in err


def test_headless_ctrl_c_stops_agents(env, headless_agents, agents, registry, _py, _wait_dead):
    from conftest import FakeStep, fake_response, fake_tool_call
    fm = headless_agents
    sleeper = _py("import time; time.sleep(30)")
    fm.script("[M]", fake_response(tool_calls=[_agent_call("K", "keeper", "c1", background=True)]),
              FakeStep(KeyboardInterrupt(), delay=1.0))  # once the agent's job is running
    fm.script("[K]", fake_response(tool_calls=[fake_tool_call(
        "bash_command", {"command": sleeper, "run_in_background": True, "background_id": "srv"})]))
    gate = fm.gate("[K]", fake_response("never"))
    with pytest.raises(KeyboardInterrupt):
        _run_headless("[M] go", permission_mode="auto", quiet=True)
    run = next(r for r in agents.runs() if r.description == "keeper")
    assert run.status == "stopped" and run.task.done()
    assert not gate.is_set()
    jobs = registry.jobs()
    assert jobs and all(_wait_dead(j.pid, 5) for j in jobs)


# --- ask-user T3.3: headless asks when there is someone to ask (REQ-3.2, D13) ---------------------

def _scripted_input(monkeypatch, answers, log=None):
    """`builtins.input` returns `answers` in order, recording each prompt (the test_headless_ask_lock shape)."""
    import builtins
    import threading
    lock = threading.Lock()

    def fake_input(prompt=""):
        with lock:
            if log is not None:
                log.append(prompt)
            answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer
    monkeypatch.setattr(builtins, "input", fake_input)


def test_headless_ask_user_interactive(monkeypatch, capsys):
    import asyncio
    from dostuff.headless import HeadlessAdapter

    written: list[str] = []
    _scripted_input(monkeypatch, ["2", "jwt, oauth", ""], written)
    adapter = HeadlessAdapter(interactive=True, quiet=False)
    questions = [
        {"question": "Which db?", "options": [{"label": "SQLite"}, {"label": "Postgres"}]},
        {"question": "Which auth?", "options": [{"label": "JWT"}, {"label": "OAuth"}], "multi_select": True},
        {"question": "Name?"},
    ]
    assert asyncio.run(adapter.ask_user(questions)) == [
        ("Which db?", "Postgres"),              # a number picks that option
        ("Which auth?", "jwt, oauth"),          # free text is taken as typed (no numbers, so no pick)
        ("Name?", "(no answer)"),               # one line per question: empty is an answer too
    ]
    err = capsys.readouterr().err + "".join(written)  # the questions go to stderr, the prompts to input()
    assert "Which db?" in err and "Postgres" in err and "OAuth" in err
    assert "1. SQLite" in err and "2. Postgres" in err  # numbered options, so a number means something


def test_headless_ask_user_picks_in_option_order(monkeypatch):
    import asyncio
    from dostuff.headless import HeadlessAdapter

    _scripted_input(monkeypatch, ["3,1"])
    adapter = HeadlessAdapter(interactive=True, quiet=True)
    question = [{"question": "Which auth?", "multi_select": True,
                 "options": [{"label": "JWT"}, {"label": "OAuth"}, {"label": "Basic"}]}]
    assert asyncio.run(adapter.ask_user(question)) == [("Which auth?", "JWT, Basic")]  # option order, not 3,1


def test_headless_ask_user_non_interactive(monkeypatch):
    import asyncio

    from dostuff.helpers.ui.ask_answers import NoUserAvailable, UNAVAILABLE_ASK_RESULT
    from dostuff.headless import HeadlessAdapter
    from dostuff.tools.ask_user import ask_user

    def no_input(prompt=""):
        raise AssertionError("must not read stdin without a terminal")
    import builtins
    monkeypatch.setattr(builtins, "input", no_input)
    adapter = HeadlessAdapter(interactive=False, quiet=True)
    try:
        asyncio.run(adapter.ask_user([{"question": "Which db?"}]))
    except NoUserAvailable:
        pass
    else:
        raise AssertionError("non-interactive ask_user must raise NoUserAvailable")
    assert adapter.denied_without_terminal == 0  # a question is not a denial
    result = asyncio.run(ask_user([{"question": "Which db?"}], adapter=adapter))
    assert result == UNAVAILABLE_ASK_RESULT


def test_headless_ask_user_lock(monkeypatch):
    import asyncio
    import builtins
    import threading
    import time

    from dostuff.headless import HeadlessAdapter
    log: list[str] = []
    lock = threading.Lock()

    def fake_input(prompt=""):
        with lock:
            log.append("enter")
        time.sleep(0.05)
        with lock:
            log.append("exit")
        return "1"
    monkeypatch.setattr(builtins, "input", fake_input)
    adapter = HeadlessAdapter(interactive=True, quiet=True)
    question = [{"question": "Which db?", "options": [{"label": "SQLite"}, {"label": "Postgres"}]}]

    async def go():
        return await asyncio.gather(adapter.ask_user(question), adapter.ask_user(question))
    assert asyncio.run(go()) == [[("Which db?", "SQLite")], [("Which db?", "SQLite")]]
    assert log == ["enter", "exit", "enter", "exit"]  # never overlapping


def test_headless_eof_cancels_question(monkeypatch):
    import asyncio
    from dostuff.headless import HeadlessAdapter

    _scripted_input(monkeypatch, [EOFError()])
    adapter = HeadlessAdapter(interactive=True, quiet=True)
    assert asyncio.run(adapter.ask_user([{"question": "Which db?"}])) is None  # cancelled, not an exception


def test_headless_ask_user_offered():
    from dostuff.headless import HeadlessAdapter  # noqa: F401  (the tool list is the same in both modes)
    from dostuff.tools.definitions import default_tool_names

    assert "ask_user" in default_tool_names()


def test_headless_ask_line_parsing():
    """The pure helper behind `ask_user`, so the numbering rules are testable without `input`."""
    from dostuff.headless import _parse_ask_line

    options = [{"label": "SQLite"}, {"label": "Postgres"}]
    assert _parse_ask_line("2", options) == "Postgres"
    assert _parse_ask_line(" 1 , 2 ", options) == "SQLite, Postgres"
    assert _parse_ask_line("9", options) == "9"            # out of range: free text
    assert _parse_ask_line("my own db", options) == "my own db"
    assert _parse_ask_line("", options) == "(no answer)"
    assert _parse_ask_line("", []) == "(no answer)"
