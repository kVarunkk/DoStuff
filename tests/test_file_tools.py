"""edit_file, read_file offset/limit, grep and glob (REQ-5.1 – 5.3, 5.7)."""
import os
import shutil
from pathlib import Path

import pytest

from dostuff.helpers.skills.discover_skills import discover_skills
from dostuff.lib.exceptions import PathAccessDenied, PathOutsideProject
from dostuff.tools.files.delete_file import delete_file
from dostuff.tools.files.edit_file import apply_edit, edit_file
from dostuff.tools.files.list_files import list_files
from dostuff.tools.files.multi_edit import multi_edit
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


# ── discovered skill dirs (skill-loading D2/B) ───────────────────────────────

SKILL_MD = "---\nname: {name}\ndescription: d\n---\nbody\n"
READ_TOOLS = {
    "read_file": lambda d: read_file(str(d / "SKILL.md")),
    "list_files": lambda d: list_files(str(d)),
    "grep": lambda d: grep("name:", str(d)),
    "glob": lambda d: glob("*.md", str(d)),
}


def _add_skill(folder, name):
    skill = folder / name
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text(SKILL_MD.format(name=name), encoding="utf-8")
    (skill / "scripts" / "x.sh").write_text("echo sibling\n", encoding="utf-8")
    return skill


@pytest.fixture
def home_skill(tmp_path, monkeypatch, home_skills):
    """A discovered skill in `~/.agents/skills`, with the session in a project that does not contain the fake `~`."""
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    skill = _add_skill(home_skills / ".agents" / "skills", "demo")
    discover_skills()
    return skill


@pytest.mark.parametrize("tool", READ_TOOLS)
def test_home_skill_is_readable_by_file_tools(home_skill, tool):
    result = READ_TOOLS[tool](home_skill)
    assert "SKILL.md" in result or "name: demo" in result


def test_sibling_file_of_a_home_skill_is_readable(home_skill):
    assert "echo sibling" in read_file(str(home_skill / "scripts" / "x.sh"))


def test_tilde_path_to_a_home_skill_resolves(home_skill):
    assert "name: demo" in read_file("~/.agents/skills/demo/SKILL.md")


@pytest.mark.parametrize("tool", READ_TOOLS)
@pytest.mark.parametrize("folder", [".agents/skills/demo-sibling", ".agents/other"])
def test_non_skill_dir_outside_root_still_denied(home_skill, home_skills, tool, folder):
    denied = home_skills / folder
    denied.mkdir(parents=True)
    (denied / "SKILL.md").write_text("no frontmatter, so never discovered\n", encoding="utf-8")
    with pytest.raises(PathOutsideProject):
        READ_TOOLS[tool](denied)


def test_symlinked_skill_md_does_not_widen_reads(tmp_path, monkeypatch, home_skills):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "note.md").write_text(SKILL_MD.format(name="x"), encoding="utf-8")
    (notes / "secret.txt").write_text("secret\n", encoding="utf-8")
    entry = home_skills / ".agents" / "skills" / "x"
    entry.mkdir()
    try:
        os.symlink(notes / "note.md", entry / "SKILL.md")
    except OSError:
        pytest.skip("creating a symlink needs a privilege this user lacks")
    discover_skills()

    with pytest.raises(PathOutsideProject):
        read_file(str(notes / "secret.txt"))


def test_write_edit_delete_deny_skill_dirs(home_skill):
    target = home_skill / "SKILL.md"
    before = target.read_text(encoding="utf-8")
    for attempt in (lambda: write_file(str(target), "x"),
                    lambda: edit_file(str(target), "body", "x"),
                    lambda: delete_file(str(target))):
        with pytest.raises(PathOutsideProject):
            attempt()
    assert target.read_text(encoding="utf-8") == before


def test_dostuff_skills_are_readable_when_discovered(tmp_path, monkeypatch, home_skills):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    skill = _add_skill(home_skills / ".dostuff" / "skills", "kept")
    discover_skills()

    assert "name: kept" in read_file(str(skill / "SKILL.md"))
    assert "name: kept" in grep("name:", str(skill))
    assert "SKILL.md" in list_files(str(skill))


def test_env_files_in_skill_dirs_still_denied(tmp_path, monkeypatch, home_skills):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    skills = [_add_skill(home_skills / ".agents" / "skills", "a"), _add_skill(home_skills / ".dostuff" / "skills", "b")]
    discover_skills()

    for skill in skills:
        (skill / ".env").write_text("TOKEN=hunter2\n", encoding="utf-8")
        with pytest.raises(PathAccessDenied):
            read_file(str(skill / ".env"))
        assert "hunter2" not in grep("TOKEN", str(skill))


def test_allowlist_follows_discovery(home_skill, home_skills):
    target = str(home_skill / "SKILL.md")
    assert "name: demo" in read_file(target)

    shutil.rmtree(home_skill)
    home_skill.mkdir()
    (home_skill / "SKILL.md").write_text("no frontmatter any more\n", encoding="utf-8")
    discover_skills()
    with pytest.raises(PathOutsideProject):
        read_file(target)

    added = _add_skill(home_skills / ".agents" / "skills", "added")
    discover_skills()
    assert "name: added" in read_file(str(added / "SKILL.md"))


# ── pins: edit_file's messages are part of its contract (spec CON-001) ────────

@pytest.mark.parametrize("old_str, new_str, replace_all, error, message", [
    ("nope", "y", False, ValueError, "old_str not found in a.py. Read the file and copy the text exactly."),
    ("x", "y", False, ValueError,
     "old_str occurs 2 times in a.py. Add surrounding lines to make it unique, or set replace_all=true."),
    ("", "y", False, ValueError, "old_str must not be empty; use write_file to create a file."),
    ("x", "x", False, ValueError, "old_str and new_str are identical; nothing to change."),
    ("x", "x", True, ValueError, "old_str and new_str are identical; nothing to change."),
])
def test_pin_edit_file_errors(project, old_str, new_str, replace_all, error, message):
    (project / "a.py").write_bytes(b"x\nx\n")
    with pytest.raises(error) as caught:
        edit_file("a.py", old_str, new_str, replace_all)
    assert str(caught.value) == message
    assert (project / "a.py").read_bytes() == b"x\nx\n"


def test_pin_edit_file_missing_file_message(project):
    with pytest.raises(FileNotFoundError) as caught:
        edit_file("missing.py", "a", "b")
    assert str(caught.value) == "File not found: missing.py"


# ── apply_edit: the matching step edit_file and multi_edit share ──────────────

def test_apply_edit():
    assert apply_edit("a b a", "b", "c", False, "f") == ("a c a", 1)
    assert apply_edit("a b a", "a", "z", True, "f") == ("z b z", 2)
    assert apply_edit("one\ntwo\n", "one\r\ntwo", "uno\r\ndos", False, "f") == ("uno\ndos\n", 1)
    with pytest.raises(ValueError, match="old_str not found in f"):
        apply_edit("a", "b", "c", False, "f")


# ── multi_edit ───────────────────────────────────────────────────────────────

def _op(old_str, new_str, **kwargs):
    return {"old_str": old_str, "new_str": new_str, **kwargs}


def test_multi_edit_applies_in_order(project):
    """Edit 2 sees edit 1's result: its old_str does not exist until edit 1 has run."""
    (project / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    result = multi_edit("a.py", [_op("x = 1", "x = 10"), _op("x = 10\ny", "x = 11\nz")])
    assert (project / "a.py").read_text(encoding="utf-8") == "x = 11\nz = 2\n"
    assert result.startswith("File edited (2 edits, 2 replacement(s)): a.py")
    assert "-x = 1" in result and "+x = 11" in result and "x = 10" not in result  # one diff of the final text


def test_multi_edit_reads_once_and_writes_once(project, monkeypatch):
    import builtins
    (project / "a.py").write_text("a b c\n", encoding="utf-8")
    real_open, modes = builtins.open, []

    def counting_open(file, mode="r", *args, **kwargs):
        if str(file).endswith("a.py"):
            modes.append(mode)
        return real_open(file, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", counting_open)
    multi_edit("a.py", [_op("a", "1"), _op("b", "2"), _op("c", "3")])
    assert modes == ["r", "w"]


def test_multi_edit_replace_all_per_edit(project):
    (project / "a.py").write_text("x x y\n", encoding="utf-8")
    result = multi_edit("a.py", [_op("x", "z", replace_all=True), _op("y", "w")])
    assert (project / "a.py").read_text(encoding="utf-8") == "z z w\n"
    assert "(2 edits, 3 replacement(s))" in result


@pytest.mark.parametrize("edits, message", [
    ([_op("nope", "b"), _op("two", "2"), _op("three", "3")],
     "edit 1 of 3: old_str not found in a.txt. Read the file and copy the text exactly."),
    ([_op("one", "1"), _op("nope", "b"), _op("three", "3")],
     "edit 2 of 3: old_str not found in a.txt. Read the file and copy the text exactly."),
    ([_op("one", "1"), _op("two", "2"), _op("nope", "b")],
     "edit 3 of 3: old_str not found in a.txt. Read the file and copy the text exactly."),
    ([_op("one", "1"), _op("dup", "d")],
     "edit 2 of 2: old_str occurs 2 times in a.txt. Add surrounding lines to make it unique, "
     "or set replace_all=true."),
    ([_op("one", "1"), _op("two", "two")], "edit 2 of 2: old_str and new_str are identical; nothing to change."),
    ([_op("one", "1"), _op("", "x")], "edit 2 of 2: old_str must not be empty; use write_file to create a file."),
    # An earlier edit can remove what a later one needs
    ([_op("one", "1"), _op("one", "uno")],
     "edit 2 of 2: old_str not found in a.txt. Read the file and copy the text exactly."),
])
def test_multi_edit_failure_leaves_file(project, edits, message):
    target = project / "a.txt"
    target.write_bytes(b"one\ntwo\nthree\ndup dup\n")
    os.utime(target, (1_000_000_000, 1_000_000_000))
    with pytest.raises(ValueError) as caught:
        multi_edit("a.txt", edits)
    assert str(caught.value) == message
    assert target.read_bytes() == b"one\ntwo\nthree\ndup dup\n"
    assert target.stat().st_mtime == 1_000_000_000


def test_multi_edit_bad_targets(project):
    (project / "a.txt").write_text("one\n", encoding="utf-8")
    with pytest.raises(ValueError, match="edits must not be empty"):
        multi_edit("a.txt", [])
    with pytest.raises(FileNotFoundError, match="File not found: missing.txt"):
        multi_edit("missing.txt", [_op("a", "b")])
    with pytest.raises(PathOutsideProject):
        multi_edit("../outside.txt", [_op("a", "b")])


def test_multi_edit_crlf(project):
    (project / "win.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")
    multi_edit("win.txt", [_op("one\ntwo", "uno\ndos"), _op("dos\nthree", "dos\ntres\nmas")])
    assert (project / "win.txt").read_bytes() == b"uno\r\ndos\r\ntres\r\nmas\r\n"

    (project / "unix.txt").write_bytes(b"one\ntwo\n")
    multi_edit("unix.txt", [_op("one\r\ntwo", "uno\r\ndos")])
    assert (project / "unix.txt").read_bytes() == b"uno\ndos\n"


def test_multi_edit_mixed_endings_match_edit_file(project):
    """A file with both endings comes out all CRLF from either tool: the same rule, applied once."""
    (project / "a.txt").write_bytes(b"one\r\ntwo\nthree\n")
    (project / "b.txt").write_bytes(b"one\r\ntwo\nthree\n")
    edit_file("a.txt", "two", "2")
    multi_edit("b.txt", [_op("two", "2")])
    assert (project / "b.txt").read_bytes() == (project / "a.txt").read_bytes() == b"one\r\n2\r\nthree\r\n"


@pytest.mark.parametrize("edits, message", [
    # Nothing between the model and the tool validates against the schema, so a weak tool-caller's shapes land here
    ('[{"old_str": "one", "new_str": "1"}]', "edits must be a list of objects with old_str and new_str, got str."),
    ({"old_str": "one", "new_str": "1"}, "edits must be a list of objects with old_str and new_str, got dict."),
    (["one"], "edit 1 of 1: must be an object with old_str and new_str, got str."),
    ([{"old_str": "one"}], "edit 1 of 1: must be an object with old_str and new_str."),
    ([_op("one", "1"), {"new_str": "1"}], "edit 2 of 2: must be an object with old_str and new_str."),
    ([_op("one", 1)], "edit 1 of 1: must be an object with old_str and new_str."),
])
def test_multi_edit_string_edits_is_a_clear_error(project, edits, message):
    (project / "a.txt").write_bytes(b"one\n")
    with pytest.raises(ValueError) as caught:
        multi_edit("a.txt", edits)
    assert str(caught.value) == message
    assert (project / "a.txt").read_bytes() == b"one\n"


def test_multi_edit_call_label_stays_on_one_line():
    from dostuff.helpers.agent.tool_output import format_call_label
    edits = [{"old_str": "line\n" * 50, "new_str": "other\n" * 50} for _ in range(20)]
    label = format_call_label("multi_edit", {"path": "a.py", "edits": edits})
    assert "\n" not in label and len(label) <= 200 and 'path="a.py"' in label


def test_system_instructions_name_multi_edit():
    from dostuff.helpers.agent.constants import SYSTEM_INSTRUCTIONS
    assert "`multi_edit`" in SYSTEM_INSTRUCTIONS
