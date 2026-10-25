from dostuff.helpers.agent.slash_suggestion import slash_suggestion


def test_completes_command_name():
    assert slash_suggestion("/mo") == "del"
    assert slash_suggestion("/") != ""
    assert slash_suggestion("/model") == ""
    assert slash_suggestion("/zzz") == ""
    assert slash_suggestion("hello /mo") == ""


def test_follows_highlighted_command():
    assert slash_suggestion("/", highlighted="/model") == "model"
    assert slash_suggestion("/c", highlighted="/compact") == "ompact"


def test_suggests_flag_after_command():
    assert slash_suggestion("/model ") == "--active"
    assert slash_suggestion("/model anthropic/claude-sonnet-5-5 ") == "--active"
    assert slash_suggestion("/model --a") == "ctive"
    assert slash_suggestion("/model --active") == ""
    assert slash_suggestion("/model --active ") == ""
    assert slash_suggestion("/model anth") == ""
    assert slash_suggestion("/fork ") == "--worktree"
    assert slash_suggestion("/compact ") == ""
    assert slash_suggestion("/model \nx") == ""
