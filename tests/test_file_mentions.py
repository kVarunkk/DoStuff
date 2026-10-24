"""`@path` mentions: attaching files/folders to the prompt, and path completion."""
from pathlib import Path

import pytest

from dostuff.helpers.agent import file_mentions
from dostuff.helpers.agent.file_mentions import (attached_paths, expand_file_mentions, format_mention,
                                                 mention_candidates, strip_attachments)


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(file_mentions, "_index", {})
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\nx = 1\n", encoding="utf-8")
    (tmp_path / "my notes.txt").write_text("note", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\0\0data")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("x", encoding="utf-8")
    return tmp_path


def test_attaches_files_and_folders(proj):
    result = expand_file_mentions('fix @src/app.py, see @src and @"my notes.txt". ping @someone')
    assert result.attached == ["src/app.py", "src/", "my notes.txt"]
    assert result.skipped == []
    assert '<attached_file path="src/app.py">\n1\tprint(\'hi\')\n2\tx = 1\n</attached_file>' in result.content
    assert '<attached_dir path="src/">\napp.py\n</attached_dir>' in result.content
    assert result.content.startswith('fix @src/app.py, see @src and @"my notes.txt". ping @someone\n\n')
    # Shown and exported as typed
    assert strip_attachments(result.content) == 'fix @src/app.py, see @src and @"my notes.txt". ping @someone'
    assert attached_paths(result.content) == ["src/app.py", "src/", "my notes.txt"]


def test_secrets_binary_and_outside_paths_are_not_attached(proj, tmp_path):
    (tmp_path.parent / "outside.txt").write_text("nope", encoding="utf-8")
    result = expand_file_mentions("@.env @logo.png @../outside.txt @missing.py email me@src/app.py")
    assert result.attached == []
    assert result.skipped == [".env: may contain secrets", "logo.png: binary file"]
    assert result.content == "@.env @logo.png @../outside.txt @missing.py email me@src/app.py"


def test_large_file_is_capped_and_mentioned_once(proj):
    (proj / "big.txt").write_text("\n".join(f"line {i}" for i in range(5000)), encoding="utf-8")
    result = expand_file_mentions("@big.txt and again @big.txt", max_chars=1000)
    assert result.attached == ["big.txt"]
    assert "chars truncated" in result.content
    assert len(result.content) < 1500


def test_mention_candidates(proj):
    assert mention_candidates("") == ["src/", "logo.png", "my notes.txt"]  # top level, folders first; no .env / node_modules
    assert mention_candidates("app") == ["src/app.py"]
    assert mention_candidates("src/") == ["src/", "src/app.py"]
    assert mention_candidates("NOTES") == ["my notes.txt"]
    assert mention_candidates("dep") == []
    assert format_mention("my notes.txt") == '@"my notes.txt"'
    assert format_mention("src/app.py") == "@src/app.py"
