from dostuff.helpers.agent.slash_suggestion import mention_suggestion, slash_suggestion


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


def test_arg_hint_after_command_name():
    from dostuff.helpers.agent.slash_suggestion import slash_arg_hint
    assert slash_arg_hint("/rename") == " <title>"
    assert slash_arg_hint("/rename ") == "<title>"
    assert slash_arg_hint("/rename foo") == ""
    assert slash_arg_hint("/model") == " [provider/model] [--active]"
    assert slash_arg_hint("/help") == ""
    assert slash_arg_hint("/ren") == ""
    assert slash_arg_hint("hello /rename") == ""
    assert slash_arg_hint("/rename\n") == ""


def test_arg_hint_for_skill_commands():
    from dostuff.helpers.agent.slash_suggestion import slash_arg_hint
    assert slash_arg_hint("/release-notes", ["/help", "/release-notes"]) == " [arguments]"
    assert slash_arg_hint("/release-notes") == ""


def test_mention_suggestion():
    skills = ["/deploy", "/deploy-staging", "/review"]
    assert mention_suggestion("/de", None, skills) == "ploy"
    assert mention_suggestion("/DE", None, skills) == "ploy"
    assert mention_suggestion("/de", "/deploy-staging", skills) == "ploy-staging"  # the menu's pick wins
    assert mention_suggestion("/deploy", None, skills) == ""  # nothing left of the first match
    assert mention_suggestion("/mo", None, skills) == ""  # a built-in cannot run mid-message
    assert mention_suggestion("/zz", None, skills) == ""
