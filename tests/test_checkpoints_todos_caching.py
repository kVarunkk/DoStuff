"""File checkpoints + /undo, the todo_write tool, and prompt-caching request kwargs."""
import asyncio
from types import SimpleNamespace

import pytest

from dostuff.agent import call_agent as call_agent_module
from dostuff.agent import run_tool as run_tool_module
from dostuff.agent.permissions import PermissionConfig
from dostuff.agent.run_tool import run_tool
from dostuff.cli_tui import TODO_RESULT_PREFIX, _format_undo
from dostuff.lib import checkpoints as checkpoints_module
from dostuff.lib import model as model_module
from dostuff.lib.checkpoints import CheckpointStore
from dostuff.tools.definitions import TOOL_SCHEMAS
from dostuff.agent.subagents.run import resolve_subagent_tools
from dostuff.tools.todo_write import get_todos, todo_write


@pytest.fixture
def project(monkeypatch, tmp_path):
    """cwd = tmp project, checkpoints in tmp, every tool call auto-approved."""
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.chdir(root)
    store = CheckpointStore(tmp_path / "checkpoints")
    monkeypatch.setattr(checkpoints_module, "_store", store)
    monkeypatch.setattr(run_tool_module, "permission_config", lambda: PermissionConfig(mode="auto"))
    return root, store


def call(name, args, turn="t1", session="s1"):
    return asyncio.run(run_tool(name, args, session_id=session, turn_id=turn))


# ── Checkpoints / undo ───────────────────────────────────────────────────────

def test_undo_restores_edits_and_removes_created_files(project):
    root, store = project
    (root / "a.txt").write_text("one\n", encoding="utf-8")
    call("edit_file", {"path": "a.txt", "old_str": "one", "new_str": "two"})
    call("edit_file", {"path": "a.txt", "old_str": "two", "new_str": "three"})  # same turn: first snapshot wins
    call("write_file", {"path": "sub/new.txt", "content": "x"})

    result = store.undo("s1")
    assert result is not None and not result.remaining
    assert (root / "a.txt").read_text(encoding="utf-8") == "one\n"
    assert not (root / "sub" / "new.txt").exists()
    assert store.undo("s1") is None  # checkpoint consumed


def test_undo_restores_deleted_file(project):
    root, store = project
    (root / "a.txt").write_text("keep me", encoding="utf-8")
    call("delete_file", {"path": "a.txt"})
    store.undo("s1")
    assert (root / "a.txt").read_text(encoding="utf-8") == "keep me"


def test_undo_walks_back_turn_by_turn(project):
    root, store = project
    call("write_file", {"path": "a.txt", "content": "v1"}, turn="t1")
    call("write_file", {"path": "a.txt", "content": "v2"}, turn="t2")
    store.undo("s1")
    assert (root / "a.txt").read_text(encoding="utf-8") == "v1"
    store.undo("s1")
    assert not (root / "a.txt").exists()


def test_undo_keeps_files_changed_since_unless_forced(project):
    root, store = project
    call("write_file", {"path": "a.txt", "content": "agent"})
    (root / "a.txt").write_text("user edit", encoding="utf-8")

    result = store.undo("s1")
    assert result is not None and result.conflicts and result.remaining and not result.changed
    assert (root / "a.txt").read_text(encoding="utf-8") == "user edit"
    assert "/undo force" in _format_undo(result)

    forced = store.undo("s1", force=True)
    assert forced is not None and forced.removed
    assert not (root / "a.txt").exists()


def test_failed_tool_call_undo_is_harmless(project):
    root, store = project
    (root / "a.txt").write_text("same", encoding="utf-8")
    with pytest.raises(ValueError, match="not found"):
        call("edit_file", {"path": "a.txt", "old_str": "zzz", "new_str": "y"})
    result = store.undo("s1")
    assert result is not None and not result.conflicts
    assert (root / "a.txt").read_text(encoding="utf-8") == "same"


def test_no_checkpoint_without_session(project):
    root, store = project
    asyncio.run(run_tool("write_file", {"path": "a.txt", "content": "x"}))
    assert store.undo("s1") is None


def test_sessions_are_separate(project):
    root, store = project
    call("write_file", {"path": "a.txt", "content": "x"}, session="other")
    assert store.undo("s1") is None
    assert store.undo("other") is not None


# ── todo_write ───────────────────────────────────────────────────────────────

def test_todo_write_replaces_list_and_formats():
    out = todo_write([
        {"content": "Read code", "status": "completed"},
        {"content": "Fix bug", "status": "in_progress"},
        {"content": "Add test", "status": "bogus"},  # unknown status → pending
        {"content": "  ", "status": "pending"},  # blank → dropped
    ], session_id="todo-s")
    assert out.splitlines() == ["☑ Read code", "◐ Fix bug", "☐ Add test"]
    assert [t["status"] for t in get_todos("todo-s")] == ["completed", "in_progress", "pending"]
    assert todo_write([], session_id="todo-s") == "Todo list cleared."


def test_todo_write_warns_on_two_in_progress():
    out = todo_write([{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}], session_id="x")
    assert "keep exactly one" in out


def test_todo_schema_is_inlined_and_not_given_to_subagents():
    todos = TOOL_SCHEMAS["todo_write"]["function"]["parameters"]["properties"]["todos"]
    assert "$ref" not in str(todos) and "$defs" not in str(todos)
    assert todos["items"]["properties"]["status"]["enum"] == ["pending", "in_progress", "completed"]
    assert "session_id" not in TOOL_SCHEMAS["todo_write"]["function"]["parameters"]["properties"]
    allowed, dropped = resolve_subagent_tools(["read_file", "todo_write"])
    assert allowed == ["read_file"] and dropped == ["todo_write"]


def test_todo_result_prefix_matches_loop_format():
    assert TODO_RESULT_PREFIX == "todo_write: "


# ── Prompt caching ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("provider, model, enabled, expected", [
    ("anthropic", "anthropic/claude-sonnet-5-5", True, True),
    ("anthropic", "anthropic/claude-sonnet-5-5", False, False),
    ("bedrock", "bedrock/anthropic.claude-sonnet-5-5", True, True),
    ("openrouter", "openrouter/anthropic/claude-opus-5-5", True, True),
    ("openrouter", "openrouter/openai/gpt-5", True, False),
    ("gemini", "gemini/gemini-2.5-pro", True, False),
    ("openai", "openai/gpt-5", True, False),
])
def test_cache_breakpoints_only_for_claude(monkeypatch, provider, model, enabled, expected):
    monkeypatch.setattr(model_module, "get_provider", lambda: provider)
    monkeypatch.setattr(model_module, "get_model", lambda: model)
    monkeypatch.setattr(model_module, "get_config", lambda: SimpleNamespace(prompt_caching=enabled))
    assert model_module.uses_explicit_prompt_caching() is expected

    monkeypatch.setattr(call_agent_module, "completion_kwargs", lambda: {"model": model})
    monkeypatch.setattr(call_agent_module, "supports_reasoning", lambda m: False)
    kwargs = call_agent_module.build_request_kwargs(stream=False)
    assert ("cache_control_injection_points" in kwargs) is expected


@pytest.mark.parametrize("max_tokens,expected", [(0, None), (8192, 8192)])
def test_max_tokens_sent_only_when_set(monkeypatch, max_tokens, expected):
    monkeypatch.setattr(call_agent_module, "completion_kwargs", lambda: {"model": "openai/gpt-4o"})
    monkeypatch.setattr(call_agent_module, "supports_reasoning", lambda m: False)
    monkeypatch.setattr(call_agent_module, "uses_explicit_prompt_caching", lambda: False)
    monkeypatch.setattr(call_agent_module, "get_config", lambda: SimpleNamespace(
        max_tokens=max_tokens, reasoning_effort="auto"))
    assert call_agent_module.build_request_kwargs(stream=False).get("max_tokens") == expected



def test_request_max_tokens_names_its_source(monkeypatch):
    from dostuff.lib.model import current_max_tokens
    monkeypatch.setattr(call_agent_module, "get_config", lambda: SimpleNamespace(max_tokens=32000))
    assert call_agent_module.request_max_tokens() == (32000, "loop.max_tokens")
    token = current_max_tokens.set(64000)  # a typed subagent run's own cap wins
    try:
        assert call_agent_module.request_max_tokens() == (64000, "the agent type's max_tokens:")
    finally:
        current_max_tokens.reset(token)


@pytest.mark.parametrize("error,cap,hinted", [
    ("model[MiniMax-M3] does not support max tokens > 524288", 600000, True),
    ("max_tokens: 200000 > 64000, which is the maximum allowed number of output tokens", 200000, True),
    ("Invalid 'max_output_tokens': integer above maximum value", 9999999, True),
    ("max_completion_tokens is too large", 9999999, True),
    ("max tokens > 524288", 0, False),  # nothing sent: provider default, not ours
    ("This model's maximum context length is 128000 tokens", 32000, False),
    ("tools[0].name is invalid", 32000, False),
])
def test_output_cap_hint(error, cap, hinted):
    from dostuff.helpers.agent.tool_output import output_cap_hint
    hint = output_cap_hint(error, cap, "loop.max_tokens")
    assert bool(hint) is hinted
    if hinted:
        assert f"max_tokens={cap} from loop.max_tokens" in hint

def test_undo_restores_multi_edit(project):
    root, store = project
    (root / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    call("multi_edit", {"path": "a.txt", "edits": [{"old_str": "one", "new_str": "1"},
                                                   {"old_str": "two", "new_str": "2"}]})
    assert (root / "a.txt").read_text(encoding="utf-8") == "1\n2\n"
    result = store.undo("s1")
    assert result is not None and result.restored
    assert (root / "a.txt").read_text(encoding="utf-8") == "one\ntwo\n"


@pytest.mark.parametrize("effort,reasoning,expected", [
    ("auto", True, None), ("high", True, "high"), ("off", True, "none"), ("high", False, None),
])
def test_reasoning_effort_request(monkeypatch, effort, reasoning, expected):
    monkeypatch.setattr(call_agent_module, "completion_kwargs", lambda: {"model": "anthropic/claude-sonnet-5-5"})
    monkeypatch.setattr(call_agent_module, "supports_reasoning", lambda m: reasoning)
    monkeypatch.setattr(call_agent_module, "get_config", lambda: SimpleNamespace(reasoning_effort=effort))
    monkeypatch.setattr(call_agent_module, "uses_explicit_prompt_caching", lambda: False)
    assert call_agent_module.build_request_kwargs(stream=False).get("reasoning_effort") == expected


def test_reasoning_effort_config_and_session(tmp_path):
    from dostuff.config import Config
    path = tmp_path / "config.yaml"
    path.write_text("loop:\n  reasoning_effort: Medium\n", encoding="utf-8")
    cfg = Config(path)
    assert cfg.reasoning_effort == "medium"
    cfg.session_reasoning_effort = "low"
    cfg.reload()
    assert cfg.reasoning_effort == "low"
    path.write_text("loop:\n  reasoning_effort: off\n  other: 1\n", encoding="utf-8")
    cfg.session_reasoning_effort = None
    cfg.reload()
    assert cfg.reasoning_effort == "off"
    path.write_text("loop:\n  reasoning_effort: bogus\n", encoding="utf-8")
    cfg.reload()
    assert cfg.reasoning_effort == "auto"
