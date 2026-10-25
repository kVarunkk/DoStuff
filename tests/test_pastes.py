"""Large pastes: placeholder collapse, inline vs. spill-to-file, retention sweep (helpers/agent/pastes.py)."""
import os
import time

from dostuff.helpers.agent.pastes import PASTE_DIR, PasteStore, sweep_old_pastes
from dostuff.tools.files.read_file import read_file


def _lines(n: int) -> str:
    return "\n".join(f"line {i}" for i in range(1, n + 1))


def test_collapse_threshold():
    store = PasteStore()
    assert not store.should_collapse(_lines(10), 10)
    assert store.should_collapse(_lines(11), 10)
    assert store.should_collapse("x" * 5000, 10)  # one long line
    assert not store.should_collapse(_lines(500), 0)  # 0 = never collapse


def test_small_paste_goes_back_inline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = PasteStore()
    ph = store.add(_lines(30))
    assert ph == "[Pasted text #1 +30 lines]"
    out = store.resolve(f"look at this:\n{ph}\nthanks", "sess1234abcd", 200, 30_000)
    assert out == f"look at this:\n{_lines(30)}\nthanks"
    assert not (tmp_path / PASTE_DIR).exists()


def test_big_paste_spills_to_a_readable_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = PasteStore()
    body = _lines(1000)
    out = store.resolve(f"why does this fail? {store.add(body)}", "sess1234abcd", 200, 30_000)
    path = tmp_path / PASTE_DIR / "sess1234-1.txt"
    assert path.read_text(encoding="utf-8") == body
    assert (tmp_path / PASTE_DIR / ".gitignore").read_text() == "*\n"
    assert out.startswith("why does this fail? <pasted_content file=\".dostuff/pastes/sess1234-1.txt\" lines=\"1000\"")
    assert "line 20\n" in out and "line 21" not in out  # 20-line preview only
    assert len(out) < len(body) / 5
    # read_file can page through it inside the project sandbox
    part = read_file(".dostuff/pastes/sess1234-1.txt", offset=500, limit=3)
    assert "line 500" in part and "line 502" in part and "line 503" not in part
    # resubmitting (history recall) reuses the same file
    store.resolve("[Pasted text #1 +1000 lines]", "sess1234abcd", 200, 30_000)
    assert len(list((tmp_path / PASTE_DIR).glob("*.txt"))) == 1


def test_char_limit_and_inline_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = PasteStore()
    ph = store.add("y" * 50_000)  # one line, but over the char cap
    assert "<pasted_content file=" in store.resolve(ph, "abcdefgh", 200, 30_000)
    ph2 = store.add(_lines(1000))
    assert store.resolve(ph2, "abcdefgh", 0, 30_000) == _lines(1000)  # 0 = always inline
    assert store.inline(f"! grep x {ph2}") == f"! grep x {_lines(1000)}"


def test_unknown_placeholder_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert PasteStore().resolve("[Pasted text #7 +3 lines]", "s", 200, 30_000) == "[Pasted text #7 +3 lines]"


def test_delete_files_and_sweep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = PasteStore()
    store.resolve(store.add(_lines(500)), "sessaaaa", 200, 30_000)
    assert store.delete_files() == 1
    assert not list((tmp_path / PASTE_DIR).glob("*.txt"))

    folder = tmp_path / PASTE_DIR
    old, new = folder / "old-1.txt", folder / "new-1.txt"
    old.write_text("a")
    new.write_text("b")
    ten_days_ago = time.time() - 10 * 86_400
    os.utime(old, (ten_days_ago, ten_days_ago))
    assert sweep_old_pastes(0) == 0  # 0 = no sweep (deleted at exit instead)
    assert sweep_old_pastes(7) == 1
    assert not old.exists() and new.exists()
