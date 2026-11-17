"""0.2.1 helpers: custom commands, `# note` memory, /context breakdown, --allowed-tools patterns."""
from pathlib import Path

from dostuff.agent.permissions import PermissionConfig, SessionPermissions, check
from dostuff.helpers.agent.context_usage import context_parts, render_context
from dostuff.helpers.agent.quick_memory import append_memory, memory_file, quick_memory_text
from dostuff.helpers.skills.custom_commands import command_prompt, discover_commands
from dostuff.helpers.skills.skill_command import parse_skill_command, skill_commands, skill_prompt


def test_custom_commands(tmp_path):
    proj, home = tmp_path / "proj", tmp_path / "home"
    (proj / ".dostuff" / "commands").mkdir(parents=True)
    (home / ".dostuff" / "commands").mkdir(parents=True)
    (proj / ".dostuff" / "commands" / "fix.md").write_text(
        "---\ndescription: \"Fix an issue\"\n---\nFix issue #$ARGUMENTS. Then run $ARGUMENTS again.\n", encoding="utf-8")
    (home / ".dostuff" / "commands" / "fix.md").write_text("user fix\n", encoding="utf-8")
    (home / ".dostuff" / "commands" / "notes.md").write_text("\n# Write notes\nBody\n", encoding="utf-8")
    (home / ".dostuff" / "commands" / "help.md").write_text("shadowed by the built-in\n", encoding="utf-8")

    found = discover_commands(proj, home)
    assert [(c["name"], c["scope"]) for c in found] == [("fix", "project"), ("fix", "user"), ("help", "user"), ("notes", "user")]
    assert found[0]["description"] == "Fix an issue" and found[3]["description"] == "Write notes"

    skill = {"name": "fix", "description": "a skill", "location": str(tmp_path / "SKILL.md")}
    table = skill_commands([skill], found)
    assert table["/fix"]["scope"] == "project" and "/help" not in table and "/notes" in table
    cmd, args = parse_skill_command("/fix 42", table)
    assert skill_prompt(cmd, args) == "Fix issue #42. Then run 42 again."
    assert command_prompt(table["/notes"], "") == "# Write notes\nBody"
    assert command_prompt(table["/notes"], "today") == "# Write notes\nBody\n\nArguments: today"


def test_quick_memory(tmp_path):
    assert quick_memory_text("# always use pytest") == "always use pytest"
    assert quick_memory_text("  ## use tabs ") == "use tabs"
    assert quick_memory_text("#") is None and quick_memory_text("hello") is None
    assert quick_memory_text("# Title\n\nsome markdown") is None

    proj, home = tmp_path / "proj", tmp_path / "home"
    proj.mkdir()
    assert memory_file("project", proj, home) == proj / "AGENTS.md"
    (proj / "CLAUDE.md").write_text("existing", encoding="utf-8")
    assert memory_file("project", proj, home) == proj / "CLAUDE.md"  # the file that's read
    target = memory_file("global", proj, home)
    assert target == home / ".dostuff" / "AGENTS.md"
    append_memory(target, "one")
    append_memory(proj / "CLAUDE.md", "two")
    assert target.read_text(encoding="utf-8") == "- one\n"
    assert (proj / "CLAUDE.md").read_text(encoding="utf-8") == "existing\n- two\n"


def test_context_breakdown():
    count = len
    parts = dict(context_parts(
        "SYSTEM" + "PROJECT", "PROJECT",
        {"read_file": {"a": 1}, "call_mcp_tool": {"b": 2}},
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": None, "tool_calls": [{"x": 1}]}],
        "notes", count,
    ))
    assert parts["System prompt"] == len("SYSTEM") and parts["Project instructions"] == len("PROJECT")
    assert parts["Tools"] == len('[{"a": 1}]') and parts["MCP tools"] == len('[{"b": 2}]')
    assert parts["Summary of earlier messages"] == 5 and parts["Messages"] == len("hello") + len('[{"x": 1}]')
    text = render_context(list(parts.items()), 1000, "m", last_request=123).plain
    assert text.startswith("Context: ~") and "Free" in text and "123 input tokens" in text
    bar = text.splitlines()[1]
    assert len(bar) == 40
    assert "Memory" not in parts  # no memory delivered: no row


def test_context_breakdown_memory_row():
    """memory recall T2.8 (REQ-5.1, D11): facts steps + the facts/episodes sections of bundles, out of Turn context."""
    from dostuff.helpers.agent.turn_context import compose_turn_context, make_context_step, make_facts_step
    count = len
    facts_step = make_facts_step("<user_facts>\n- name: Nemo\n</user_facts>", ("name:abc",))
    bundle = compose_turn_context(environment="<env>cwd=/p</env>", facts="- lint: ruff", episodes="- [DECISION] WAL",
                                  prompt_context="hook says hi")
    messages = [facts_step, {"role": "user", "content": "hello"}, make_context_step(bundle, ("ep1",))]
    parts = context_parts("SYSTEM", "", {}, messages, "", count)
    names = [name for name, _ in parts]
    assert names.index("Memory") == names.index("Turn context") + 1
    memory = dict(parts)["Memory"]
    expected = (len(facts_step["content"]) + len("<other_relevant_facts>\n- lint: ruff\n</other_relevant_facts>")
                + len("<past_episodes>\n- [DECISION] WAL\n</past_episodes>"))
    assert memory == expected
    # Every token counted once: Turn context keeps the rest (environment, hook context)
    assert dict(parts)["Turn context"] + memory == len(facts_step["content"]) + len(bundle)
    assert dict(parts)["Messages"] == len("hello")


def test_allowed_tools_patterns():
    ask = PermissionConfig(mode="ask")
    session = SessionPermissions(allow=["write_file", "npm test", "web_fetch:docs.python.org", "call_mcp_tool:gh__issue"])
    assert check("write_file", "write", {"path": "a"}, session, ask).action == "allow"
    assert check("delete_file", "write", {"path": "a"}, session, ask).action == "ask"
    assert check("bash_command", "shell", {"command": "npm test -- -x"}, session, ask).action == "allow"
    assert check("bash_command", "shell", {"command": "npm install"}, session, ask).action == "ask"
    assert check("web_fetch", "network", {"url": "https://docs.python.org/3/"}, session, ask).action == "allow"
    assert check("web_fetch", "network", {"url": "https://example.org"}, session, ask).action == "ask"
    assert check("call_mcp_tool", "mcp", {"name": "gh__issue"}, session, ask).action == "allow"
    # A tool name allows every call of it; plan mode and settings changes still win
    shell = SessionPermissions(allow=["bash_command"])
    assert check("bash_command", "shell", {"command": "rm -rf build && make"}, shell, ask).action == "allow"
    assert check("write_file", "write", {"path": "a"}, SessionPermissions(mode="plan", allow=["write_file"]), ask).action == "deny"
    assert check("edit_config", "settings", {}, SessionPermissions(allow=["edit_config"]), ask).action == "ask"


# --- Cache-stable context, T3.6 ----------------------------------------------------------------------------------

def test_context_parts_has_turn_context():
    parts = context_parts("SYSTEM", "", {}, [], "", len, next_context="ENV")
    assert [name for name, _ in parts] == ["System prompt", "Turn context", "Tools", "MCP tools", "Project instructions",
                                           "Summary of earlier messages", "Messages"]
    assert dict(parts)["System prompt"] == len("SYSTEM") and dict(parts)["Turn context"] == len("ENV")


def test_context_parts_counts_every_context_step_once():
    from dostuff.helpers.agent.turn_context import make_context_step, make_plan_mode_step
    plan = make_plan_mode_step(True, "x")
    messages = [make_context_step("bundle"), {"role": "user", "content": "hello"}, plan]
    parts = dict(context_parts("S", "", {}, messages, "", len))
    assert parts["Turn context"] == len("bundle") + len(plan["content"])
    assert parts["Messages"] == len("hello")


def test_turn_context_part_has_a_colour():
    from dostuff.helpers.agent.context_usage import PART_STYLES
    assert PART_STYLES["Turn context"]
