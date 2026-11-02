"""The `ask_user` tool and its answer grammar (plan/2026-11-02-ask-user, REQ-1.1 - 1.4).

Pure unit tests: no model, no widgets, no subprocess. The TUI side lives in tests/test_tui.py.
"""


def test_imports():
    from dostuff.helpers.ui.ask_answers import format_answers, parse_answers
    from dostuff.tools.ask_user import ask_user

    assert callable(ask_user) and callable(format_answers) and callable(parse_answers)


# --- T1.1 the grammar (REQ-1.3, D5) --------------------------------------------------------------

def test_answers_round_trip():
    from dostuff.helpers.ui.ask_answers import format_answers, parse_answers

    pairs = [
        ("Which database should I use?", "Postgres"),
        ("Which auth methods?", "JWT, OAuth"),
    ]
    text = format_answers(pairs)
    assert text == (
        "Asked the user 2 questions; they answered:\n"
        "1. Which database should I use?\n"
        "   → Postgres\n"
        "2. Which auth methods?\n"
        "   → JWT, OAuth"
    )
    assert parse_answers(text) == pairs
    assert format_answers(pairs[:1]) == (
        "Asked the user 1 question; they answered:\n1. Which database should I use?\n   → Postgres"
    )
    assert parse_answers(format_answers(pairs[:1])) == pairs[:1]


def test_answers_hostile_text():
    """An answer holding the grammar's own punctuation survives; only lines starting `<n>. ` open a pair."""
    from dostuff.helpers.ui.ask_answers import format_answers, parse_answers

    pairs = [
        ("What should the message say?", 'say "a → b", then c'),
        ("Which one → really?", "the first: one, or the second one"),
    ]
    assert parse_answers(format_answers(pairs)) == pairs


def test_parse_answers_rejects_other_text():
    from dostuff.helpers.ui.ask_answers import parse_answers

    for text in (
        "",
        "wrote t.txt",
        "The user cancelled the questions; no answers.\nUse your best judgement and say what you assumed.",
        "No user is available in this run, so the questions were not shown.\n"
        "Ask nothing further; state your assumption and continue.",
        "error: ask_user needs at least one question with text.",
    ):
        assert parse_answers(text) is None, text


def test_format_answers_collapses():
    """An answer holding a newline is emitted on one line, and that collapsed value parses back."""
    from dostuff.helpers.ui.ask_answers import format_answers, parse_answers

    text = format_answers([("Which one?", "line one\nline two")])
    assert text.splitlines() == ["Asked the user 1 question; they answered:", "1. Which one?", "   → line one line two"]
    assert parse_answers(text) == [("Which one?", "line one line two")]
    assert format_answers([("  Which   one?  ", "  a   b  ")]) == (
        "Asked the user 1 question; they answered:\n1. Which one?\n   → a b"
    )


def test_result_text_constants():
    """The two non-answers are module constants (the tool and the renderer share them)."""
    from dostuff.helpers.ui import ask_answers

    assert ask_answers.CANCELLED_ASK_RESULT == (
        "The user cancelled the questions; no answers.\nUse your best judgement and say what you assumed."
    )
    assert ask_answers.UNAVAILABLE_ASK_RESULT == (
        "No user is available in this run, so the questions were not shown.\n"
        "Ask nothing further; state your assumption and continue."
    )
    assert ask_answers.ASK_ANSWER_ARROW == "   → "


# --- T1.2 the tool's shape and schema (REQ-1.1, D3) ----------------------------------------------

def test_limits_are_named():
    import dostuff.tools.ask_user as ask_user_module

    assert ask_user_module.ASK_MAX_QUESTIONS == 4
    assert ask_user_module.ASK_MAX_OPTIONS == 4
    assert ask_user_module.ASK_HEADER_CHARS == 12
    assert ask_user_module.ASK_LABEL_CHARS == 60
    assert ask_user_module.ASK_DESCRIPTION_CHARS == 200


def test_tool_schema_shape():
    from dostuff.helpers.tools.generate_tool_schema import generate_litellm_tool_schema
    from dostuff.tools.ask_user import ask_user

    schema = generate_litellm_tool_schema(ask_user, name="ask_user", exclude={"adapter"})
    params = schema["function"]["parameters"]
    assert schema["function"]["name"] == "ask_user"
    assert params["type"] == "object"
    assert params["required"] == ["questions"]
    assert set(params["properties"]) == {"questions"}
    assert "$defs" not in schema["function"]["parameters"]

    questions = params["properties"]["questions"]
    assert questions["type"] == "array"
    item = questions["items"]
    assert item["type"] == "object"
    assert item.get("required") == ["question"]
    assert set(item["properties"]) == {"question", "header", "options", "multi_select"}
    assert all(p.get("description") for p in item["properties"].values())

    options = item["properties"]["options"]
    assert options["type"] == "array"
    option = options["items"]
    assert option["type"] == "object"
    assert option.get("required") == ["label"]
    assert set(option["properties"]) == {"label", "description"}
    assert all(p.get("description") for p in option["properties"].values())
    assert item["properties"]["multi_select"]["type"] == "boolean"

    blob = str(schema)
    for absent in ("preview", "answers", "adapter"):
        assert f"'{absent}'" not in blob, absent
    assert "ask_user" in schema["function"]["description"]
