"""The create_agent_type tool: the main agent defines a new subagent type as an agent file."""
import pytest

from dostuff.agent.subagents.definitions import current_definitions, parse_agent_file
from dostuff.tools.create_agent_type import create_agent_type, create_agent_type_preview


@pytest.fixture
def proj(tmp_path, monkeypatch, agents_home):
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.chdir(root)
    return root


def test_writes_a_loadable_project_type(proj):
    result = create_agent_type(
        name="code-reviewer", description="Reviews a diff for bugs.", prompt="You review code.\n\nBe terse.",
        tools=["Read", "grep", "glob"], model="fast", max_iterations=20)
    path = proj / ".dostuff" / "agents" / "code-reviewer.md"
    assert result.startswith(f"Created agent type 'code-reviewer' at {path}.")
    assert 'agent(subagent_type="code-reviewer"' in result
    agent_def, warnings = parse_agent_file(path)
    assert warnings == [] and agent_def is not None
    assert (agent_def.name, agent_def.description, agent_def.prompt) == (
        "code-reviewer", "Reviews a diff for bugs.", "You review code.\n\nBe terse.")
    assert agent_def.tools == ("read_file", "grep", "glob")
    assert (agent_def.model, agent_def.max_iterations) == ("fast", 20)
    assert current_definitions().types["code-reviewer"].source == str(path)  # `agent` can start it now


def test_user_scope_and_defaults(proj, agents_home):
    create_agent_type(name="helper", description="Helps.", prompt="Help.", scope="user")
    agent_def, _ = parse_agent_file(agents_home / ".dostuff" / "agents" / "helper.md")
    assert agent_def is not None
    assert (agent_def.tools, agent_def.model, agent_def.max_iterations) == (None, "inherit", None)


def test_refuses_to_replace_unless_asked(proj):
    create_agent_type(name="rev", description="One.", prompt="First.")
    with pytest.raises(ValueError, match="already exists"):
        create_agent_type(name="rev", description="Two.", prompt="Second.")
    assert create_agent_type(name="rev", description="Two.", prompt="Second.", replace=True).startswith("Replaced")
    assert current_definitions().types["rev"].prompt == "Second."


@pytest.mark.parametrize("kwargs, message", [
    ({"name": "Code Reviewer"}, "lowercase"),
    ({"name": "fork"}, "reserved"),
    ({"description": "  "}, "description"),
    ({"prompt": ""}, "prompt"),
    ({"tools": ["agent", "todo_write"]}, "none of the tools"),
    ({"max_iterations": 0}, "max_iterations"),
])
def test_rejects_bad_input_and_writes_nothing(proj, kwargs, message):
    args = {"name": "rev", "description": "D.", "prompt": "P.", **kwargs}
    with pytest.raises(ValueError, match=message):
        create_agent_type(**args)
    assert not (proj / ".dostuff" / "agents").exists()


def test_restricted_tools_are_dropped_with_a_warning(proj):
    result = create_agent_type(name="rev", description="D.", prompt="P.", tools=["read_file", "agent"])
    assert "Warning:" in result and "'agent' dropped (not available to subagents)" in result
    assert current_definitions().types["rev"].tools == ("read_file",)


def test_notes_a_replaced_builtin(proj):
    assert "replaces the built-in 'explore' type" in create_agent_type(name="explore", description="D.", prompt="P.")


def test_preview_shows_the_file(proj):
    preview = create_agent_type_preview({"name": "rev", "description": "D.", "prompt": "Be careful."})
    assert "creates" in preview and "rev.md" in preview and "Be careful." in preview and "name: rev" in preview
    assert "it will fail" in create_agent_type_preview({"name": "BAD", "description": "D.", "prompt": "P."})


def test_main_agent_only_and_always_asks():
    from dostuff.agent.permissions import PermissionConfig, check
    from dostuff.agent.subagents import RESTRICTED_SUBAGENT_TOOLS
    from dostuff.tools.definitions import TOOL_MAP, default_tool_names
    assert "create_agent_type" in default_tool_names()
    assert "create_agent_type" in RESTRICTED_SUBAGENT_TOOLS
    spec = TOOL_MAP["create_agent_type"]
    args = {"name": "rev", "description": "D.", "prompt": "P."}
    decision = check("create_agent_type", spec.kind, args, None, PermissionConfig(mode="auto"))
    assert decision.action == "ask"
    assert "Create agent type 'rev' (.dostuff/agents)" in decision.prompt


def test_system_prompt_mentions_it():
    from dostuff.helpers.agent.constants import SYSTEM_INSTRUCTIONS
    assert "`create_agent_type`" in SYSTEM_INSTRUCTIONS
