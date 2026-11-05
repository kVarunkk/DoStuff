"""`session_to_markdown` (`dostuff export`, `/export`)."""
from dostuff.helpers.memory.export_session import session_to_markdown


def test_pin_export_roundtrip():
    steps = [{"role": "user", "content": "fix the bug"}, {"role": "assistant", "content": "done"}]
    text = session_to_markdown(steps, None, "s1")
    assert "## User" in text and "fix the bug" in text
    assert "## Assistant" in text and "done" in text
