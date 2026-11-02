"""The `ask_user` tool and its answer grammar (plan/2026-11-02-ask-user, REQ-1.1 - 1.4).

Pure unit tests: no model, no widgets, no subprocess. The TUI side lives in tests/test_tui.py.
"""


def test_imports():
    from dostuff.helpers.ui.ask_answers import format_answers, parse_answers
    from dostuff.tools.ask_user import ask_user

    assert callable(ask_user) and callable(format_answers) and callable(parse_answers)
