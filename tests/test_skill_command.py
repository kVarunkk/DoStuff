from pathlib import Path

from dostuff.helpers.agent.slash_suggestion import slash_suggestion
from dostuff.helpers.skills.skill_command import parse_skill_command, skill_commands, skill_prompt


def test_skill_commands_skip_builtins_duplicates_and_bad_names():
    skills = [
        {"name": "Release-Notes", "description": "a", "location": "a/SKILL.md"},
        {"name": "model", "description": "clashes with /model", "location": "b/SKILL.md"},
        {"name": "two words", "description": "not one word", "location": "c/SKILL.md"},
        {"name": "release-notes", "description": "found second", "location": "d/SKILL.md"},
    ]
    commands = skill_commands(skills)
    assert list(commands) == ["/release-notes"] and commands["/release-notes"]["location"] == "a/SKILL.md"


def test_parse_skill_command():
    commands = {"/review": {"name": "review"}}
    assert parse_skill_command("/review", commands) == ({"name": "review"}, "")
    assert parse_skill_command("  /Review  src/app.py\nmore ", commands) == ({"name": "review"}, "src/app.py\nmore")
    assert parse_skill_command("/reviewer", commands) is None
    assert parse_skill_command("please /review", commands) is None


def test_skill_prompt_drops_frontmatter(tmp_path):
    path = tmp_path / "review" / "SKILL.md"
    path.parent.mkdir()
    path.write_text("---\nname: review\ndescription: d\n---\n\n# Review\nCheck it.\n", encoding="utf-8")
    prompt = skill_prompt({"name": "review", "location": str(path)}, "src/app.py")
    assert "# Review\nCheck it.\n</skill>" in prompt and "description: d" not in prompt
    assert str(Path(path).parent) in prompt and prompt.endswith("Arguments: src/app.py")


def test_slash_suggestion_includes_given_commands():
    assert slash_suggestion("/rel", commands=["/help", "/release-notes"]) == "ease-notes"
    assert slash_suggestion("/rel") == ""
