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


# --- T1.3 cleaning and clamping (REQ-1.2, D4) ----------------------------------------------------

def _clean(questions):
    import dostuff.tools.ask_user as ask_user_module
    return ask_user_module._clean_questions(questions)


def test_cleaning_clamps():
    cleaned = _clean([{
        "question": "  Which   database\n should I  use? ",
        "header": "A" * 30,
        "options": [
            {"label": "L" * 90, "description": "D" * 400},
            {"label": "Short", "description": "  one   line  "},
        ],
    }])
    assert len(cleaned) == 1
    q = cleaned[0]
    assert q["question"] == "Which database should I use?"
    assert q["header"] == "A" * 12  # hard cut, no ellipsis: it is a fixed-width tab strip
    assert len(q["options"][0]["label"]) == 60 and q["options"][0]["label"].endswith("…")
    assert len(q["options"][0]["description"]) == 200 and q["options"][0]["description"].endswith("…")
    assert q["options"][1] == {"label": "Short", "description": "one line"}
    assert q["multi_select"] is False


def test_cleaning_drops_and_numbers():
    cleaned = _clean([
        {"question": "   "},                       # no question text: dropped, and it doesn't count for Q<n>
        {"options": [{"label": "a"}, {"label": "b"}]},   # no question text either
        {"question": "Second kept?", "header": "   "},   # blank header → Q1 (it is the first kept question)
        {"question": "Third kept?", "options": [
            {"label": "   "},                      # no label: dropped
            "not a dict",                          # not an option: dropped
            {"label": "Fine"},
        ]},
    ])
    assert [q["question"] for q in cleaned] == ["Second kept?", "Third kept?"]
    assert [q["header"] for q in cleaned] == ["Q1", "Q2"]
    assert cleaned[1]["options"] == [{"label": "Fine"}]


def test_cleaning_ignores_unknown_keys():
    cleaned = _clean([{
        "question": "Q?", "preview": "x", "answers": {"Q?": "a"},
        "options": [{"label": "a", "preview": "y", "style": "dim"}, {"label": "b"}],
    }])
    assert set(cleaned[0]) == {"question", "header", "options", "multi_select"}
    assert set(cleaned[0]["options"][0]) == {"label"}


def test_cleaning_multi_select_defaults_and_survives():
    cleaned = _clean([{"question": "One?", "options": [{"label": "a"}, {"label": "b"}], "multi_select": True}])
    assert cleaned[0]["multi_select"] is True
    cleaned = _clean([{"question": "One?", "options": [{"label": "a"}, {"label": "b"}]}])
    assert cleaned[0]["multi_select"] is False


def test_cleaned_questions_reach_the_adapter(ask_script):
    """The cleaning happens in the tool, so the adapter sees the clamped values, not the raw ones."""
    import asyncio
    from dostuff.tools.ask_user import ask_user

    adapter = ask_script([("Which database should I use?", "Postgres")])
    asyncio.run(ask_user([{
        "question": "  Which   database\n should I  use? ",
        "header": "A" * 30,
        "options": [{"label": "L" * 90}, {"label": "Short"}],
    }], adapter=adapter))
    assert adapter.questions == [[{
        "question": "Which database should I use?",
        "header": "A" * 12,
        "options": [{"label": "L" * 59 + "…"}, {"label": "Short"}],
        "multi_select": False,
    }]]


def test_raw_json_arguments_tolerated():
    """`questions` as a JSON string (a fake model, an old transcript) is parsed; a non-list is the D4 refusal."""
    import asyncio
    import json

    from dostuff.helpers.ui.ask_answers import parse_answers
    from dostuff.tools.ask_user import ask_user

    from conftest import AskScript

    pairs = [("Which db?", "SQLite")]
    adapter = AskScript(pairs)
    text = asyncio.run(ask_user(json.dumps([{"question": "Which db?", "options": [{"label": "a"}, {"label": "b"}]}]),
                               adapter=adapter))
    assert parse_answers(text) == pairs
    assert adapter.questions[0][0]["question"] == "Which db?"

    for bad in (None, "x", 5, {"question": "q"}):
        assert asyncio.run(ask_user(bad, adapter=AskScript(pairs))).startswith("error: ")


# --- T1.4 structural refusals (REQ-1.2, D4) ------------------------------------------------------

def _refused(questions, adapter):
    import asyncio
    from dostuff.tools.ask_user import ask_user
    result = asyncio.run(ask_user(questions, adapter=adapter))
    assert adapter.calls == 0, "a refused call must not reach the adapter"
    return result


def test_refusals(ask_script):
    adapter = ask_script([])
    two_options = [{"label": "Postgres"}, {"label": "SQLite"}]

    assert _refused([], adapter) == "error: ask_user needs at least one question with text."
    assert _refused([{"question": "  "}], adapter) == "error: ask_user needs at least one question with text."
    assert _refused([{"question": f"Q{i}?"} for i in range(5)], adapter) == (
        "error: ask_user takes 1-4 questions, got 5. Split them across calls."
    )
    assert _refused([{"question": "Which db?", "options": [{"label": f"o{i}"} for i in range(5)]}], adapter) == (
        'error: question "Which db?" has 5 options; the limit is 4.'
    )
    assert _refused([{"question": "Which db?", "options": [{"label": "Postgres"}]}], adapter) == (
        'error: question "Which db?" has one option. Give it 2-4, or leave options out for a free-text answer.'
    )
    assert _refused([{"question": "Which db?", "options": [{"label": "Postgres"}, {"label": "Postgres"}]}],
                    adapter) == 'error: question "Which db?" repeats the option "Postgres".'
    assert _refused([{"question": "Which db?", "options": two_options},
                     {"question": "Which db? ", "options": two_options}], adapter) == (
        'error: ask_user repeats the question "Which db?".'
    )
    assert _refused([{"question": "Which db?", "multi_select": True}], adapter) == (
        'error: question "Which db?" is multi_select but has no options.'
    )


def test_refusal_order(ask_script):
    """The first fault in D4's order wins: the question count, then each question's options."""
    adapter = ask_script([])
    both_wrong = [{"question": "Which db?", "options": [{"label": "one"}]} for _ in range(5)]
    assert _refused(both_wrong, adapter) == "error: ask_user takes 1-4 questions, got 5. Split them across calls."
    # Two faulty questions: the first one's fault is reported
    assert _refused([{"question": "First?", "options": [{"label": "a"}]},
                     {"question": "Second?", "options": [{"label": "a"}, {"label": "a"}]}], adapter) == (
        'error: question "First?" has one option. Give it 2-4, or leave options out for a free-text answer.'
    )


def test_refusal_never_raises(ask_script):
    """Five malformed shapes all return an `error: ` string, never an exception."""
    adapter = ask_script([])
    for bad in (None, "x", [{}], [{"question": 5}], [{"question": "q", "options": "Postgres"}]):
        assert _refused(bad, adapter).startswith("error: ")


# --- T1.5 results: answers, cancelled, unavailable (REQ-1.4, D5) ---------------------------------

def test_answers_result(ask_script):
    import asyncio
    from dostuff.helpers.ui.ask_answers import format_answers
    from dostuff.tools.ask_user import ask_user

    pairs = [("Which database should I use?", "Postgres")]
    adapter = ask_script(pairs)
    result = asyncio.run(ask_user([{"question": "Which database should I use?",
                                    "options": [{"label": "Postgres"}, {"label": "SQLite"}]}], adapter=adapter))
    assert result == format_answers(pairs)


def test_cancelled_result(ask_script):
    import asyncio
    from dostuff.helpers.ui.ask_answers import CANCELLED_ASK_RESULT
    from dostuff.tools.ask_user import ask_user

    assert asyncio.run(ask_user([{"question": "Which db?"}], adapter=ask_script(None))) == CANCELLED_ASK_RESULT


def test_unavailable_result(ask_script):
    import asyncio
    from types import SimpleNamespace

    from dostuff.helpers.ui.ask_answers import NoUserAvailable, UNAVAILABLE_ASK_RESULT
    from dostuff.tools.ask_user import ask_user

    questions = [{"question": "Which db?"}]
    assert asyncio.run(ask_user(questions)) == UNAVAILABLE_ASK_RESULT
    assert asyncio.run(ask_user(questions, adapter=SimpleNamespace(emit=lambda *a: None))) == UNAVAILABLE_ASK_RESULT
    assert asyncio.run(ask_user(questions, adapter=ask_script(raise_exc=NoUserAvailable()))) == UNAVAILABLE_ASK_RESULT


def test_adapter_exception_is_a_result(ask_script):
    import asyncio
    from dostuff.tools.ask_user import ask_user

    result = asyncio.run(ask_user([{"question": "Which db?"}], adapter=ask_script(raise_exc=RuntimeError("boom"))))
    assert result.startswith("error: ask_user could not ask")
    assert "boom" in result


def test_answers_are_one_line_each(ask_script):
    import asyncio
    from dostuff.helpers.ui.ask_answers import parse_answers
    from dostuff.tools.ask_user import ask_user

    adapter = ask_script([("Which db?", "line one\nline two")])
    result = asyncio.run(ask_user([{"question": "Which db?"}], adapter=adapter))
    assert parse_answers(result) == [("Which db?", "line one line two")]
    assert "\nline two" not in result


def test_length_mismatch_is_cancelled(ask_script):
    """An adapter that answers fewer pairs than it was asked is a cancelled call, not an IndexError."""
    import asyncio
    from dostuff.helpers.ui.ask_answers import CANCELLED_ASK_RESULT
    from dostuff.tools.ask_user import ask_user

    result = asyncio.run(ask_user([{"question": "First?"}, {"question": "Second?"}],
                                  adapter=ask_script([("First?", "a")])))
    assert result == CANCELLED_ASK_RESULT


# --- T2.1 registration (REQ-2.1, CON-008) --------------------------------------------------------

def test_registered():
    from dostuff.helpers.tools.generate_tool_schema import generate_litellm_tool_schema
    from dostuff.tools.ask_user import ask_user
    from dostuff.tools.definitions import TOOL_MAP, TOOL_SCHEMAS
    from dostuff.tools.names import BUILTIN_TOOL_NAMES

    spec = TOOL_MAP["ask_user"]
    assert spec.kind == "read"
    assert spec.injected == frozenset({"adapter"})
    assert "ask_user" in BUILTIN_TOOL_NAMES
    assert "ask_user" in TOOL_SCHEMAS
    assert spec.schema == generate_litellm_tool_schema(ask_user, name="ask_user", exclude={"adapter"})
