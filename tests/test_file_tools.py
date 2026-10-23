"""edit_file, read_file offset/limit, grep and glob (REQ-5.1 – 5.3, 5.7)."""
from pathlib import Path

import pytest

from dostuff.tools.files.edit_file import edit_file
from dostuff.tools.files.read_file import read_file
from dostuff.tools.files.write_file import write_file
from dostuff.tools.search.glob import glob
from dostuff.tools.search.grep import grep


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ── edit_file ────────────────────────────────────────────────────────────────

def test_edit_replaces_unique_match_and_returns_plain_diff(project):
    (project / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    result = edit_file("a.py", "y = 2", "y = 3")
    assert (project / "a.py").read_text(encoding="utf-8") == "x = 1\ny = 3\n"
    assert "-y = 2" in result and "+y = 3" in result
    assert "\033[" not in result  # no ANSI codes reach history


def test_edit_absent_old_str_errors(project):
    (project / "a.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not found"):
        edit_file("a.py", "nope", "y")


def test_edit_non_unique_requires_replace_all(project):
    (project / "a.py").write_text("x\nx\n", encoding="utf-8")
    with pytest.raises(ValueError, match="occurs 2 times"):
        edit_file("a.py", "x", "y")
    edit_file("a.py", "x", "y", replace_all=True)
    assert (project / "a.py").read_text(encoding="utf-8") == "y\ny\n"


def test_edit_preserves_crlf(project):
    (project / "a.txt").write_bytes(b"one\r\ntwo\r\n")
    edit_file("a.txt", "one\ntwo", "uno\ndos")
    assert (project / "a.txt").read_bytes() == b"uno\r\ndos\r\n"


def test_edit_missing_file_and_outside_project(project):
    with pytest.raises(FileNotFoundError):
        edit_file("missing.py", "a", "b")
    with pytest.raises(ValueError):  # PathOutsideProject
        edit_file("../outside.py", "a", "b")


def test_write_file_result_has_no_ansi(project):
    write_file("b.txt", "hello\n")
    result = write_file("b.txt", "hello\nworld\n")
    assert "+world" in result and "\033[" not in result


# ── read_file ────────────────────────────────────────────────────────────────

def test_read_file_numbers_lines_and_reports_truncation(project):
    (project / "f.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")
    full = read_file("f.txt")
    assert full.splitlines()[0] == " 1\tline1"
    assert "showing lines" not in full
    part = read_file("f.txt", offset=3, limit=2)
    assert part.splitlines()[:2] == ["3\tline3", "4\tline4"]
    assert "of 10" in part and "offset=5" in part


def test_read_file_offset_past_end(project):
    (project / "f.txt").write_text("a\n", encoding="utf-8")
    assert "past the end" in read_file("f.txt", offset=5)


# ── grep / glob ──────────────────────────────────────────────────────────────

def _tree(project: Path) -> None:
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("def main():\n    return 'TODO'\n", encoding="utf-8")
    (project / "README.md").write_text("todo list\n", encoding="utf-8")
    for excluded in (".git", "node_modules", "venv-x", "__pycache__", "build", "dist"):
        (project / excluded).mkdir()
        (project / excluded / "junk.py").write_text("TODO\n", encoding="utf-8")
    (project / ".env").write_text("SECRET=TODO\n", encoding="utf-8")


def test_grep_finds_matches_and_skips_default_excludes(project):
    _tree(project)
    result = grep("TODO")
    assert "src/app.py:2:" in result
    for excluded in (".git", "node_modules", "venv-x", "__pycache__", "build", "dist", ".env"):
        assert excluded not in result


def test_grep_ignore_case_glob_and_limit(project):
    _tree(project)
    assert "README.md" in grep("todo", ignore_case=True)
    assert "README.md" not in grep("todo", ignore_case=True, glob="*.py")
    assert "max_results=1" in grep("o", ignore_case=True, max_results=1)


def test_grep_invalid_regex(project):
    with pytest.raises(ValueError, match="Invalid regular expression"):
        grep("(")


def test_glob_patterns(project):
    _tree(project)
    assert glob("*.py").splitlines() == ["src/app.py"]
    assert "src/app.py" in glob("**/*.py")
    assert glob("src/*.py").splitlines() == ["src/app.py"]
    assert "No files" in glob("*.rs")
