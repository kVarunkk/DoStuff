from pathlib import Path

import pytest

from dostuff.helpers.agent.file_mentions import strip_attachments
from dostuff.helpers.agent.slash_suggestion import slash_suggestion
from dostuff.helpers.skills.skill_command import (
    mention_note, parse_skill_command, skill_commands, skill_mentions, skill_prompt, with_mention_note,
)


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


def test_manual_skill_is_still_a_command(tmp_path):
    path = tmp_path / "deploy" / "SKILL.md"
    path.parent.mkdir()
    path.write_text("---\nname: deploy\ndescription: d\ndisable-model-invocation: true\n---\n\nShip.\n", encoding="utf-8")
    skill = {"name": "deploy", "description": "d", "location": str(path), "disable_model_invocation": True}
    commands = skill_commands([skill])
    assert list(commands) == ["/deploy"]
    assert "Ship." in skill_prompt(*parse_skill_command("/deploy now", commands))


# --- inline mentions (plan/2026-11-06-skills-and-multi-edit D10 - D12) ------------------------------------------

_MENTIONABLE = {
    "/deploy": {"name": "deploy", "description": "Ship it", "dir": "/s/deploy"},
    "/review": {"name": "review", "description": "Check it", "dir": "/s/review"},
    "/v1.2": {"name": "v1.2", "description": "Dotted", "dir": "/s/v"},
    "/standup": {"name": "standup", "kind": "command", "description": "A custom command"},
}


@pytest.mark.parametrize("text, names", [
    ("go ahead and /deploy", ["deploy"]),
    ("go ahead and /deploy now", ["deploy"]),
    ("go ahead and /deploy.", ["deploy"]),
    ("run /deploy, then stop", ["deploy"]),
    ("can you /deploy?", ["deploy"]),
    ("do it: /deploy!", ["deploy"]),
    ("first /deploy; then", ["deploy"]),
    ("this /deploy: it", ["deploy"]),
    ("go /DEPLOY", ["deploy"]),
    ("line one\n/deploy", ["deploy"]),
    ("tab\t/deploy", ["deploy"]),
    ("use /review and /deploy", ["review", "deploy"]),
    ("use /deploy and /review and /deploy", ["deploy", "review"]),
    ("a dotted /v1.2 name", ["v1.2"]),
    ("a dotted /v1.2.", ["v1.2"]),
    # Not mentions
    ("/deploy", []),
    ("  /deploy now", []),
    ("/deploy then /review", []),  # a message starting with a skill is that skill's run; stacking is out of scope
    ("/standup then /deploy", []),  # so is one starting with a custom command
    ("read /tmp/notes.md", []),
    ("in /usr/local/bin", []),
    ("say `/deploy`", []),
    ("try (/deploy)", []),
    ("see /deploy/x", []),
    ("see a/deploy", []),
    ("two dots /deploy..", []),
    ("close /deploy)", []),
    ("run /standup", []),
    ("run /unknown", []),
    ("", []),
])
def test_skill_mentions(text, names):
    assert [s["name"] for s in skill_mentions(text, _MENTIONABLE)] == names


def test_a_message_starting_with_a_path_still_scans():
    assert [s["name"] for s in skill_mentions("/etc/hosts then /deploy", _MENTIONABLE)] == ["deploy"]


NOTE_HEAD = (
    "<skill-mentions>\n"
    "The user named these skills in the message above. That permits using them for this message only; use one only "
    "if the message asks for it. Before using one, read the SKILL.md in its dir; the files it refers to are relative "
    "to that dir.\n"
)


def test_mention_note():
    deploy = {"name": "deploy", "description": "Ship it", "dir": "/s/deploy", "disable_model_invocation": True}
    review = {"name": "review", "description": "Check it", "dir": "/s/review"}
    assert mention_note([]) == ""
    assert mention_note([deploy]) == NOTE_HEAD + "- name: deploy\n  description: Ship it\n  dir: /s/deploy\n</skill-mentions>"
    assert mention_note([deploy, review]) == (
        NOTE_HEAD + "- name: deploy\n  description: Ship it\n  dir: /s/deploy\n"
        "- name: review\n  description: Check it\n  dir: /s/review\n</skill-mentions>"
    )


def test_with_mention_note():
    deploy = {"name": "deploy", "description": "Ship it", "dir": "/s/deploy"}
    assert with_mention_note("text", []) == "text"
    assert with_mention_note("text", [deploy]) == "text\n\n" + mention_note([deploy])


def test_strip_attachments_removes_the_note_and_nothing_else():
    note = mention_note([{"name": "deploy", "description": "Ship it", "dir": "/s/deploy"}])
    attached = 'hi\n\n<attached_file path="a">\n1\tx\n</attached_file>'
    assert strip_attachments("go /deploy\n\n" + note) == "go /deploy"
    assert strip_attachments(f"{attached}\n\n{note}") == "hi"
    # Typed by a user, it is just text: only the block the harness writes is taken out
    assert strip_attachments("what is <skill-mentions> for? </skill-mentions>") \
        == "what is <skill-mentions> for? </skill-mentions>"
