"""Terminal title / notification sequences and the config keys behind them (roadmap 0.2.1)."""
from pathlib import Path

from dostuff.config import Config, parse_compact_threshold
from dostuff.helpers.ui.terminal_signals import (
    POP_TITLE, format_title, notification_sequence, restore_title_sequence, should_notify, title_sequence,
)


def test_title_and_notifications():
    assert format_title("running", "proj") == "⏳ dostuff · proj"
    assert format_title("waiting", "proj") == "✋ dostuff · proj"
    assert format_title("done", "") == "✓ dostuff"
    assert format_title("idle", "proj") == "dostuff · proj"
    # A BEL / ESC in the text must not end the sequence early
    assert title_sequence("a\x07b\x1bc") == "\x1b]0;a b c\x07"
    assert restore_title_sequence() == "\x1b]0;\x07" + POP_TITLE
    assert notification_sequence("bell", "x") == "\x07"
    assert notification_sequence("osc9", "turn\nfinished") == "\x1b]9;turn finished\x07"
    assert notification_sequence("off", "x") == ""


def test_should_notify():
    assert not should_notify("off", 999, 30, focused=False)
    assert should_notify("bell", 1, 30, focused=False)
    assert not should_notify("bell", 1, 30, focused=True)
    assert should_notify("osc9", 45, 30, focused=True)


def test_compact_threshold_parsing():
    assert parse_compact_threshold(80) == 80
    assert parse_compact_threshold("75%") == 75
    for off in ("off", False, 0, "none"):
        assert parse_compact_threshold(off) is None
    for bad in (150, "lots", -5):
        try:
            parse_compact_threshold(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} accepted")


def test_config_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text("loop:\n  auto_compact_threshold: off\nui:\n  notifications: on\n  terminal_title: false\n"
                    "  notify_after_seconds: 5\npermissions:\n  mode: plan\n", encoding="utf-8")
    cfg = Config(path)
    assert cfg.compact_threshold is None and cfg.notifications == "bell" and not cfg.terminal_title
    assert cfg.notify_after_seconds == 5 and cfg.permission_mode == "plan"
    cfg.session_compact_threshold = (90,)
    cfg.reload()  # a /compact auto choice survives a settings reload
    assert cfg.compact_threshold == 90

    path.write_text("loop:\n  auto_compact_threshold: lots\nui:\n  notifications: loud\n", encoding="utf-8")
    cfg = Config(Path(path))
    assert cfg.compact_threshold == 50 and cfg.notifications == "off" and cfg.terminal_title
