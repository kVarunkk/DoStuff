"""`session_to_markdown` (`dostuff export`, `/export`)."""
from dostuff.helpers.memory.export_session import session_to_markdown


def test_pin_export_roundtrip():
    steps = [{"role": "user", "content": "fix the bug"}, {"role": "assistant", "content": "done"}]
    text = session_to_markdown(steps, None, "s1")
    assert "## User" in text and "fix the bug" in text
    assert "## Assistant" in text and "done" in text


def test_export_omits_context_step():
    from dostuff.helpers.agent.turn_context import make_context_step, make_plan_mode_step
    steps = [make_context_step("<turn-context>\ncwd: /secret-path\n</turn-context>"),
             {"role": "user", "content": "fix the bug"}, make_plan_mode_step(True, "## Plan mode"),
             {"role": "assistant", "content": "done"}]
    text = session_to_markdown(steps, None, "s1")
    assert text.count("## User") == 1 and "fix the bug" in text
    assert "<turn-context>" not in text and "/secret-path" not in text and "Plan mode" not in text


def test_dostuff_export_uses_the_same_reader():
    """`dostuff export` and `/export` both go through `session_to_markdown`: there is no second path to hide it in."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "dostuff"
    for name in ("cli.py", "cli_tui.py"):
        assert "session_to_markdown(" in (root / name).read_text(encoding="utf-8")
