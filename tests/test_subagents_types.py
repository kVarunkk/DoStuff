"""Agent definitions loader, built-in types and model resolution (subagents spec)."""
from pathlib import Path

import pytest

from dostuff.agent.subagents import definitions as defs_module
from dostuff.agent.subagents.definitions import load_definitions


def _agent_file(folder: Path, filename: str, body: str = "", **front) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["---"] + [f"{k}: {v}" for k, v in front.items()] + ["---", body]
    path = folder / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def proj(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    return root


def _folders(proj: Path, home: Path) -> dict[str, Path]:
    return {
        "project-dostuff": proj / ".dostuff" / "agents",
        "project-claude": proj / ".claude" / "agents",
        "user-dostuff": home / ".dostuff" / "agents",
        "user-claude": home / ".claude" / "agents",
    }


# --- T3.1 search order and frontmatter --------------------------------------------------------------


def test_search_order(proj, agents_home):
    folders = _folders(proj, agents_home)
    paths = {key: _agent_file(folder, "rev.md", f"body from {key}", name="rev", description="d")
             for key, folder in folders.items()}
    for key in ("project-dostuff", "project-claude", "user-dostuff", "user-claude"):
        loaded = load_definitions(proj, agents_home)
        assert loaded.types["rev"].prompt == f"body from {key}"
        assert loaded.types["rev"].source == str(paths[key])
        paths[key].unlink()
    assert "rev" not in load_definitions(proj, agents_home).types

    for key, folder in folders.items():
        _agent_file(folder, "rev.md", f"body from {key}", name="rev", description="d")
    (folders["project-dostuff"] / "rev.md").unlink()
    no_claude = load_definitions(proj, agents_home, read_claude=False)
    assert no_claude.types["rev"].prompt == "body from user-dostuff"
    (folders["user-dostuff"] / "rev.md").unlink()
    assert "rev" not in load_definitions(proj, agents_home, read_claude=False).types


def test_loader_frontmatter(proj, agents_home):
    folder = proj / ".dostuff" / "agents"
    _agent_file(folder, "rev.md", "You review code.\n\nCarefully.  \n\n\n",
                name="rev", description="Reviews diffs", tools="read_file, grep", model="fast",
                max_iterations=40, color="blue", permissionMode="plan")
    loaded = load_definitions(proj, agents_home)
    rev = loaded.types["rev"]
    assert rev.description == "Reviews diffs"
    assert rev.prompt == "You review code.\n\nCarefully."
    assert rev.tools == ("read_file", "grep")
    assert rev.model == "fast"
    assert rev.max_iterations == 40
    assert loaded.warnings == []


def test_loader_keeps_builtins(proj, agents_home):
    loaded = load_definitions(proj, agents_home)
    assert loaded.types["general-purpose"].source == "built-in"


# --- T3.2 tool mapping ------------------------------------------------------------------------------


@pytest.mark.parametrize("tools, expected, dropped", [
    ("Read, Grep, Bash", ["read_file", "grep", "bash_command"], []),
    ("[Read, Grep, Bash]", ["read_file", "grep", "bash_command"], []),
    ("Edit, MultiEdit, Write", ["edit_file", "write_file"], []),
    ("Glob, LS, WebFetch, WebSearch", ["glob", "list_files", "web_fetch", "web_search"], []),
    ("mcp__github__search", ["search_mcp_tools", "get_mcp_tool_details", "call_mcp_tool"], []),
    ("read_file, Frobnicate, agent", ["read_file"], ["Frobnicate", "agent"]),
])
def test_tool_mapping(proj, agents_home, tools, expected, dropped):
    path = _agent_file(proj / ".dostuff" / "agents", "t.md", "b", name="t", description="d", tools=tools)
    loaded = load_definitions(proj, agents_home)
    assert loaded.types["t"].tools == tuple(expected)
    assert len(loaded.warnings) == len(dropped)
    for name, warning in zip(dropped, loaded.warnings):
        assert name in warning and str(path) in warning


def test_tool_mapping_resolves_shell_jobs(proj, agents_home):
    from dostuff.agent.subagents.run import resolve_subagent_tools
    _agent_file(proj / ".dostuff" / "agents", "t.md", "b", name="t", description="d", tools="Read, Bash")
    tools = load_definitions(proj, agents_home).types["t"].tools
    allowed, dropped = resolve_subagent_tools(list(tools or ()))
    assert allowed == ["read_file", "bash_command", "bash_output", "kill_shell"] and dropped == []


def test_tools_omitted_means_every_subagent_tool(proj, agents_home):
    from dostuff.agent.subagents import RESTRICTED_SUBAGENT_TOOLS
    from dostuff.agent.subagents.run import resolve_subagent_tools
    from dostuff.tools.definitions import TOOL_MAP
    _agent_file(proj / ".dostuff" / "agents", "t.md", "b", name="t", description="d")
    assert load_definitions(proj, agents_home).types["t"].tools is None
    allowed, dropped = resolve_subagent_tools(None)
    assert set(allowed) == set(TOOL_MAP) - RESTRICTED_SUBAGENT_TOOLS
    assert "write_file" in allowed and dropped == []


def test_builtin_tool_names_match_tool_map():
    from dostuff.tools.definitions import TOOL_MAP
    from dostuff.tools.names import BUILTIN_TOOL_NAMES
    assert BUILTIN_TOOL_NAMES == set(TOOL_MAP)


def test_tool_mapping_does_not_import_litellm():
    """Agent files are parsed before the TUI's first frame; importing litellm there blanks the
    screen for seconds (the TUI loads it later, in _load_backend)."""
    import subprocess
    import sys
    code = (
        "import sys\n"
        "from dostuff.agent.subagents.definitions import map_tools\n"
        "assert map_tools('read_file, Bash', 'x')[0]\n"
        "assert 'litellm' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


# --- T3.3 broken files ------------------------------------------------------------------------------


def test_broken_files_skipped(proj, agents_home):
    folder = proj / ".dostuff" / "agents"
    folder.mkdir(parents=True)
    (folder / "bad_yaml.md").write_text("---\nname: [unclosed\n---\nbody", encoding="utf-8")
    (folder / "no_front.md").write_text("just a body", encoding="utf-8")
    _agent_file(folder, "no_name.md", "b", description="d")
    _agent_file(folder, "no_desc.md", "b", name="x")
    (folder / "empty.md").write_text("", encoding="utf-8")
    (folder / "bad_utf8.md").write_bytes(b"---\nname: u\ndescription: d\n---\n\xff\xfe\xfa")
    (folder / "list.md").write_text("---\n- a\n- b\n---\nbody", encoding="utf-8")
    _agent_file(folder, "good.md", "fine", name="good", description="d")
    (folder / "notes.txt").write_text("not an agent", encoding="utf-8")

    loaded = load_definitions(proj, agents_home)
    assert loaded.types["good"].prompt == "fine"
    bad = ["bad_yaml.md", "no_front.md", "no_name.md", "no_desc.md", "empty.md", "bad_utf8.md", "list.md"]
    assert len(loaded.warnings) == len(bad)
    for filename in bad:
        assert sum(filename in w for w in loaded.warnings) == 1, filename
    assert not any("notes.txt" in w for w in loaded.warnings)


def test_unreadable_folder_is_not_an_error(proj, agents_home):
    (proj / ".dostuff").mkdir()
    (proj / ".dostuff" / "agents").write_text("a file, not a folder", encoding="utf-8")
    loaded = load_definitions(proj, agents_home)
    assert "general-purpose" in loaded.types


# --- T3.4 built-in types ----------------------------------------------------------------------------


def test_builtin_types(proj, agents_home):
    loaded = load_definitions(proj, agents_home)
    for name in ("general-purpose", "explore", "plan"):
        assert loaded.types[name].source == "built-in" and loaded.types[name].prompt
    for name in ("explore", "plan"):
        tools = set(loaded.types[name].tools or ())
        assert {"read_file", "grep", "glob", "list_files", "bash_command", "web_fetch"} <= tools
        assert not {"write_file", "edit_file"} & tools
        assert loaded.types[name].read_only_shell
    assert loaded.types["general-purpose"].tools is None and not loaded.types["general-purpose"].read_only_shell

    folder = proj / ".dostuff" / "agents"
    _agent_file(folder, "explore.md", "my explorer", name="explore", description="mine")
    path = _agent_file(folder, "fork.md", "my fork", name="fork", description="mine")
    loaded = load_definitions(proj, agents_home)
    assert loaded.types["explore"].prompt == "my explorer" and not loaded.types["explore"].read_only_shell
    assert "fork" not in loaded.types
    assert len(loaded.warnings) == 1 and str(path) in loaded.warnings[0] and "reserved" in loaded.warnings[0]


def _run_agent(prompt: str, subagent_type: str, adapter, permissions=None) -> str:
    import asyncio

    from dostuff.agent.permissions import SessionPermissions
    from dostuff.agent.run_tool import run_tool

    async def go():
        return await run_tool(
            "agent", {"description": "look", "prompt": prompt, "subagent_type": subagent_type},
            mcp_client=object(), session_id="s1", turn_id="t1", adapter=adapter,  # type: ignore[arg-type]
            permissions=permissions or SessionPermissions(mode="ask"),
        )
    return asyncio.run(go())


def test_explore_read_only_shell(agents, fake_model, agents_home, tmp_path, monkeypatch):
    from conftest import FakeAdapter, fake_response, fake_tool_call
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x").write_text("keep me")
    rm = fake_tool_call("bash_command", {"command": "rm x"})
    status = fake_tool_call("bash_command", {"command": "git status"})
    fake_model.script("[A]", fake_response(tool_calls=[rm]), fake_response(tool_calls=[status]), fake_response("ok"))
    parent = FakeAdapter(auto="yes")
    assert _run_agent("[A] look", "explore", parent).startswith("ok")
    assert parent.asks == []
    assert (tmp_path / "x").read_text() == "keep me"
    tool_steps = [m for m in fake_model.calls_for("[A]")[-1]["messages"] if m.get("role") == "tool"]
    assert "read-only" in tool_steps[0]["content"]
    assert "read-only" not in tool_steps[1]["content"]
    # outside the run the flag is off: the main agent's shell is checked as usual
    from dostuff.agent.subagents import read_only_shell
    assert read_only_shell.get() is False


def test_general_purpose_writes_checked(agents, fake_model, agents_home, tmp_path, monkeypatch):
    from conftest import FakeAdapter, fake_response, fake_tool_call
    monkeypatch.chdir(tmp_path)
    write = fake_tool_call("write_file", {"path": "out.txt", "content": "hi"})
    fake_model.script("[A]", fake_response(tool_calls=[write]), fake_response("wrote"))
    parent = FakeAdapter(auto="yes")
    assert _run_agent("[A] write", "general-purpose", parent).startswith("wrote")
    assert len(parent.asks) == 1 and (tmp_path / "out.txt").read_text() == "hi"


def test_unknown_type_lists_all(agents, fake_model, agents_home):
    from conftest import FakeAdapter
    result = _run_agent("x", "nope", FakeAdapter(auto="yes"))
    assert result.startswith("[error]") and all(n in result for n in ("general-purpose", "explore", "plan"))


# --- T3.5 reload ------------------------------------------------------------------------------------


def test_reload_on_mtime(proj, agents_home, monkeypatch):
    import os

    from dostuff.helpers.tools.project_root import use_project_root
    calls: list[Path] = []
    real = defs_module.parse_agent_file
    monkeypatch.setattr(defs_module, "parse_agent_file", lambda p: (calls.append(p), real(p))[1])
    monkeypatch.setattr(defs_module, "_cache", None)
    folder = proj / ".dostuff" / "agents"
    path = _agent_file(folder, "rev.md", "v1", name="rev", description="d")
    with use_project_root(proj):
        first = defs_module.current_definitions()
        assert first.types["rev"].prompt == "v1" and len(calls) == 1
        assert defs_module.current_definitions() is first and len(calls) == 1  # no change, no re-read

        path.write_text(path.read_text().replace("v1", "v2 longer"), encoding="utf-8")
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
        assert defs_module.current_definitions().types["rev"].prompt == "v2 longer"
        assert defs_module.get_definitions()["rev"].prompt == "v2 longer"

        _agent_file(agents_home / ".claude" / "agents", "new.md", "n", name="new", description="d")
        assert "new" in defs_module.current_definitions().types
        path.unlink()
        assert "rev" not in defs_module.current_definitions().types

        before = len(calls)
        reloaded = defs_module.reload()
        assert len(calls) > before and "new" in reloaded.types


def test_read_claude_agents_switch(proj, agents_home, monkeypatch):
    from dostuff.config import get_config
    from dostuff.helpers.tools.project_root import use_project_root
    monkeypatch.setattr(defs_module, "_cache", None)
    _agent_file(proj / ".claude" / "agents", "cc.md", "b", name="cc", description="d")
    with use_project_root(proj):
        assert "cc" in defs_module.current_definitions().types
        monkeypatch.setattr(get_config(), "agents_read_claude_agents", False)
        assert "cc" not in defs_module.current_definitions().types


# --- T3.6 model resolution --------------------------------------------------------------------------


@pytest.fixture
def main_model(monkeypatch):
    """Main model `openai/m-main`, `agents.models: {fast: anthropic/claude-haiku-5-5}`, explore_model inherit."""
    from dostuff.config import get_config
    cfg = get_config()
    monkeypatch.setattr(cfg, "_session_model", {"name": "openai/m-main", "provider": "openai"})
    monkeypatch.setattr(cfg, "agents_models", {"fast": "anthropic/claude-haiku-5-5", "cheap": "fast"})
    monkeypatch.setattr(cfg, "agents_explore_model", "inherit")
    monkeypatch.setattr(cfg, "api_key_env_explicit", "")
    return cfg


@pytest.mark.parametrize("call, type_model, explore_model, expected, warns", [
    ("", "inherit", "inherit", "openai/m-main", None),
    ("inherit", "fast", "inherit", "openai/m-main", None),
    ("fast", "inherit", "inherit", "anthropic/claude-haiku-5-5", None),
    ("", "fast", "inherit", "anthropic/claude-haiku-5-5", None),
    ("anthropic/claude-haiku-5-5", "inherit", "inherit", "anthropic/claude-haiku-5-5", None),
    ("", "sonnet", "inherit", "openai/m-main", "`sonnet` isn't in agents.models"),
    ("opus", "inherit", "inherit", "openai/m-main", "`opus` isn't in agents.models"),
    ("", "gpt-weird", "inherit", "gpt-weird", None),
    ("openai/other", "fast", "inherit", "openai/other", None),  # the call wins over the type (open point 3)
    ("", "inherit", "fast", "anthropic/claude-haiku-5-5", None),  # explore → explore_model
    ("openai/x", "inherit", "fast", "openai/x", None),
])
def test_model_resolution(main_model, call, type_model, explore_model, expected, warns):
    from dostuff.agent.subagents.definitions import AgentDef, resolve_model
    main_model.agents_explore_model = explore_model
    model, warning = resolve_model(call, AgentDef("explore", "d", model=type_model), main_model)
    assert model == expected
    assert (warning is None) if warns is None else (warns in (warning or ""))


def test_explore_model_only_for_explore(main_model):
    from dostuff.agent.subagents.definitions import AgentDef, resolve_model
    main_model.agents_explore_model = "fast"
    assert resolve_model("", AgentDef("plan", "d"), main_model) == ("openai/m-main", None)


def test_alias_to_alias_and_inherit(main_model):
    from dostuff.agent.subagents.definitions import AgentDef, resolve_model
    assert resolve_model("cheap", AgentDef("x", "d"), main_model)[0] == "anthropic/claude-haiku-5-5"
    main_model.agents_models = {"same": "inherit", "loop": "loop"}
    assert resolve_model("same", AgentDef("x", "d"), main_model)[0] == "openai/m-main"
    assert resolve_model("loop", AgentDef("x", "d"), main_model)[0] == "loop"  # no endless lookup


# --- T3.7 current_model ContextVar ------------------------------------------------------------------


def _glob_call():
    from conftest import fake_tool_call
    return fake_tool_call("glob", {"pattern": "*.nothing"})


def _slow(text: str):
    from conftest import FakeStep, fake_response
    return FakeStep(fake_response(text), delay=0.05)


def _start_agent(prompt: str, subagent_type: str = "general-purpose"):
    from dostuff.tools.agent import agent
    return agent(description="a", prompt=prompt, mcp_client=object(), session_id="s",  # type: ignore[arg-type]
                 turn_id="t", subagent_type=subagent_type)


def test_model_contextvar_isolated(agents, fake_model, agent_type, main_model, monkeypatch):
    import asyncio

    from conftest import fake_response
    from dostuff.lib.model import get_model
    main_model.agents_models = {"m-fast": "openai/m-fast"}
    agent_type("quick", model="m-fast")
    seen: dict[str, list[str]] = {"[A]": [], "[B]": []}
    real = fake_model.acompletion

    async def spy(**kwargs):
        marker = fake_model._marker(kwargs["messages"])
        if marker in seen:
            seen[marker].append(await asyncio.to_thread(get_model))  # a worker thread sees the run's model
        return await real(**kwargs)
    from dostuff.agent import call_agent as call_agent_module
    monkeypatch.setattr(call_agent_module, "acompletion", spy)
    fake_model.script("[A]", fake_response(None, tool_calls=[_glob_call()]), _slow("a done"))
    fake_model.script("[B]", fake_response(None, tool_calls=[_glob_call()]), _slow("b done"))

    async def go():
        main_seen: list[str] = []

        async def main_side():
            for _ in range(5):
                main_seen.append(get_model())
                await asyncio.sleep(0.02)
        results = await asyncio.gather(_start_agent("[A] x", "quick"), _start_agent("[B] x"), main_side())
        return results, main_seen
    (a, b, _), main_seen = asyncio.run(go())
    assert a.startswith("a done") and b.startswith("b done")
    assert [c["model"] for c in fake_model.calls_for("[A]")] == ["openai/m-fast"] * 2
    assert [c["model"] for c in fake_model.calls_for("[B]")] == ["openai/m-main"] * 2
    assert seen["[A]"] == ["openai/m-fast"] * 2 and seen["[B]"] == ["openai/m-main"] * 2
    assert set(main_seen) == {"openai/m-main"} and get_model() == "openai/m-main"


def test_run_cost_uses_run_model(agents, fake_model, agent_type, main_model, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from conftest import fake_response
    from dostuff.agent import loop as loop_module
    from dostuff.helpers.agent import get_model_token_limit as limit_module
    main_model.agents_models = {"m-fast": "openai/m-fast"}
    monkeypatch.setattr(main_model, "context_window", 0)
    agent_type("quick", model="m-fast")
    priced: list[str] = []
    monkeypatch.setattr(loop_module, "call_cost", lambda model, usage: (priced.append(model), 0.5)[1])
    limits: list[str] = []
    monkeypatch.setattr(limit_module, "_litellm_limit", lambda model: (limits.append(model), 1000)[1])
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    fake_model.script("[A]", fake_response("ok", usage=usage))

    async def go():
        from dostuff.helpers.agent.get_model_token_limit import get_model_token_limit
        from dostuff.lib.model import current_model
        await _start_agent("[A] x", "quick")
        current_model.set("openai/m-fast")  # what the run's own context window lookups read
        get_model_token_limit()
    asyncio.run(go())
    assert priced == ["openai/m-fast"]
    assert limits and set(limits) == {"openai/m-fast"}
    [run] = agents.runs()
    assert run.model == "openai/m-fast" and run.cost_usd == 0.5


def test_api_key_only_for_main_provider(main_model, monkeypatch):
    from dostuff.lib.model import completion_kwargs, current_model
    monkeypatch.setattr(main_model, "api_key_env_explicit", "MY_OPENAI_KEY")
    monkeypatch.setenv("MY_OPENAI_KEY", "sk-main")
    assert completion_kwargs() == {"model": "openai/m-main", "api_key": "sk-main"}
    token = current_model.set("openai/m-fast")
    try:
        assert completion_kwargs() == {"model": "openai/m-fast", "api_key": "sk-main"}
        current_model.set("anthropic/claude-haiku-5-5")
        assert completion_kwargs() == {"model": "anthropic/claude-haiku-5-5"}
    finally:
        current_model.reset(token)


def test_missing_key_fails_run(agents, fake_model, agent_type, main_model, monkeypatch):
    import asyncio
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent_type("quick", model="fast")
    result = asyncio.run(_start_agent("[A] x", "quick"))
    assert "ANTHROPIC_API_KEY" in result and "anthropic/claude-haiku-5-5" in result
    assert fake_model.calls_for("[A]") == []
    assert agents.runs()[0].status == "failed"


# --- T3.8 main prompt lists the types ---------------------------------------------------------------


def test_main_prompt_lists_types(proj, agents_home, monkeypatch):
    from dostuff.agent.instructions import build_base_instructions
    from dostuff.helpers.tools.project_root import use_project_root
    monkeypatch.setattr(defs_module, "_cache", None)
    folder = proj / ".dostuff" / "agents"
    _agent_file(folder, "rev.md", "b", name="code-reviewer", description="Reviews a diff for bugs.")
    _agent_file(folder, "explore.md", "b", name="explore", description="My own explorer.")
    with use_project_root(proj):
        base, _ = build_base_instructions()
    section = base.split("## Delegating to subagents", 1)[1].split("\n## ", 1)[0]
    assert "- code-reviewer: Reviews a diff for bugs." in section
    assert "- explore: My own explorer." in section and section.count("- explore:") == 1
    assert "- general-purpose: " in section and "- plan: " in section
    assert "parallel" in section and "self-contained" in section
    assert "delegate_to_subagent" not in base


def test_agent_call_model_applied(agents, fake_model, main_model):
    """`agent(model=…)` reaches the run, and a model warning ends up in the report (T3.6, found in T6.2)."""
    import asyncio

    from conftest import fake_response
    from dostuff.tools.agent import agent
    main_model.agents_models = {"fast": "openai/m-fast"}  # the main provider: no other key needed
    fake_model.script("[A]", fake_response("a done"))
    fake_model.script("[B]", fake_response("b done"))

    async def go():
        kw = dict(description="a", mcp_client=object(), session_id="s", turn_id="t")
        return (await agent(prompt="[A] x", model="fast", **kw),  # type: ignore[arg-type]
                await agent(prompt="[B] x", model="sonnet", **kw))  # type: ignore[arg-type]
    a, b = asyncio.run(go())
    assert [c["model"] for c in fake_model.calls_for("[A]")] == ["openai/m-fast"]
    assert a.startswith("a done")
    assert [c["model"] for c in fake_model.calls_for("[B]")] == ["openai/m-main"]
    assert b.startswith("[warning] `sonnet` isn't in agents.models") and "b done" in b


# --- ask_user is restricted (plan/2026-11-02-ask-user T2.4, D10, D11) -------------------------------

def test_subagent_restrictions():
    from dostuff.agent.subagents import RESTRICTED_SUBAGENT_TOOLS
    from dostuff.agent.subagents.definitions import CLAUDE_TOOL_NAMES, map_tools
    from dostuff.agent.subagents.run import resolve_subagent_tools

    assert "ask_user" in RESTRICTED_SUBAGENT_TOOLS
    assert "ask_user" not in resolve_subagent_tools(None)[0]

    tools, warnings = map_tools("ask_user, read_file", "a.md")
    assert tools == ("read_file",) and len(warnings) == 1 and "ask_user" in warnings[0]
    assert "not available to subagents" in warnings[0]

    # Claude Code's name is known, so it is reported as restricted rather than unknown
    assert CLAUDE_TOOL_NAMES["AskUserQuestion"] == ("ask_user",)
    tools, warnings = map_tools("AskUserQuestion", "b.md")
    assert tools == () and len(warnings) == 1
    assert "AskUserQuestion" in warnings[0] and "restricted" not in warnings[0]
    assert "not available to subagents" in warnings[0]
