import os
import subprocess
import sys
from pathlib import Path

import pytest

from dostuff.agent.instructions import skills_summary
from dostuff.helpers.skills import discover_skills as ds

REPO = Path(__file__).resolve().parents[1]


def _raw(root, folder, frontmatter):
    """A skill whose frontmatter block is exactly `frontmatter`."""
    d = root / folder
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\n{frontmatter}\n---\nbody\n", encoding="utf-8")
    return d


def _discover(root, monkeypatch):
    monkeypatch.setattr(ds, "SKILLS_DIRS", [str(root)])
    return ds.discover_skills()


def _only(root, monkeypatch, frontmatter):
    """The one skill `frontmatter` yields, from folder `folder`."""
    _raw(root, "folder", frontmatter)
    (skill,) = _discover(root, monkeypatch)
    return skill


def _skill(root, folder, name, desc):
    d = root / folder
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\nbody\n", encoding="utf-8")


def test_same_name_in_two_dirs_listed_once_first_dir_wins(tmp_path, monkeypatch):
    project, home = tmp_path / "project", tmp_path / "home"
    _skill(project, "review", "review", "project copy")
    _skill(home, "review", "review", "home copy")
    _skill(home, "other", "other", "only in home")
    monkeypatch.setattr(ds, "SKILLS_DIRS", [str(project), str(home)])

    skills = ds.discover_skills()

    assert sorted(s["name"] for s in skills) == ["other", "review"]
    review = next(s for s in skills if s["name"] == "review")
    assert review["description"] == "project copy"


def test_same_name_in_different_folders_of_one_dir(tmp_path, monkeypatch):
    _skill(tmp_path, "a-folder", "dup", "first")
    _skill(tmp_path, "b-folder", "dup", "second")
    monkeypatch.setattr(ds, "SKILLS_DIRS", [str(tmp_path)])

    assert [s["description"] for s in ds.discover_skills()] == ["first"]


def test_same_dir_reached_twice_scanned_once(tmp_path, monkeypatch):
    _skill(tmp_path, "x", "x", "d")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ds, "SKILLS_DIRS", [".", str(tmp_path)])

    assert [s["name"] for s in ds.discover_skills()] == ["x"]


# ── frontmatter (spec D3) ────────────────────────────────────────────────────

@pytest.mark.parametrize("written, parsed", [
    ('"Review the changes"', "Review the changes"),
    ("'Create a README.md file'", "Create a README.md file"),
    (r'"or asks to \"review since X\"."', 'or asks to "review since X".'),
])
def test_frontmatter_quoted_description_unquoted(tmp_path, monkeypatch, written, parsed):
    assert _only(tmp_path, monkeypatch, f"name: x\ndescription: {written}")["description"] == parsed


@pytest.mark.parametrize("block, parsed", [
    ("description: >\n  folded over\n  two lines", "folded over two lines"),
    ("description: |\n  line one\n  line two", "line one line two"),
])
def test_frontmatter_folded_and_literal_block(tmp_path, monkeypatch, block, parsed):
    assert _only(tmp_path, monkeypatch, f"name: x\n{block}")["description"] == parsed


def test_frontmatter_colon_and_hash_inside_quotes(tmp_path, monkeypatch):
    assert _only(tmp_path, monkeypatch, 'name: x\ndescription: "a: b # c"')["description"] == "a: b # c"


def test_frontmatter_plain_scalar_drops_trailing_comment(tmp_path, monkeypatch):
    assert _only(tmp_path, monkeypatch, "name: x\ndescription: Reviews code # bugs")["description"] == "Reviews code"


@pytest.mark.parametrize("above", ["", "short_description: wrong\n"])
def test_frontmatter_invalid_yaml_falls_back_to_regex(tmp_path, monkeypatch, above):
    description = 'Research a company. Auto-fires on: "research companies" and more'
    skill = _only(tmp_path, monkeypatch, f"name: company-research\n{above}description: {description}")
    assert skill["name"] == "company-research" and skill["description"] == description


def test_frontmatter_absent_is_still_skipped(tmp_path, monkeypatch):
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "SKILL.md").write_text("# no frontmatter\n", encoding="utf-8")
    assert _discover(tmp_path, monkeypatch) == []


def test_short_description_does_not_shadow(tmp_path, monkeypatch):
    skill = _only(tmp_path, monkeypatch, "name: x\nshort_description: wrong\ndescription: right")
    assert skill["description"] == "right"


def test_non_utf8_skill_does_not_stop_discovery(tmp_path, monkeypatch):
    _skill(tmp_path, "good", "good", "fine")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "SKILL.md").write_bytes(b"---\nname: bad\ndescription: caf\xff\n---\nbody\n")

    assert sorted(s["name"] for s in _discover(tmp_path, monkeypatch)) == ["bad", "good"]


@pytest.mark.parametrize("written", ["1.10", "yes", "true"])
def test_name_keeps_its_source_spelling(tmp_path, monkeypatch, written):
    assert _only(tmp_path, monkeypatch, f"name: {written}\ndescription: d")["name"] == written


def test_empty_name_and_description(tmp_path, monkeypatch):
    skill = _only(tmp_path, monkeypatch, "name:\ndescription:")
    assert skill["name"] == "folder" and skill["description"] == ""


def test_non_string_description_counts_as_absent(tmp_path, monkeypatch):
    assert _only(tmp_path, monkeypatch, "name: x\ndescription:\n  - a\n  - b")["description"] == ""


def test_absent_name_falls_back_to_folder(tmp_path, monkeypatch):
    assert _only(tmp_path, monkeypatch, "description: d")["name"] == "folder"


def test_absent_description_is_empty_string(tmp_path, monkeypatch):
    assert _only(tmp_path, monkeypatch, "name: x")["description"] == ""


# ── location and dir (spec D4–D6) ────────────────────────────────────────────

def test_location_and_dir_are_absolute(tmp_path, monkeypatch):
    _skill(tmp_path / "skills", "x", "x", "d")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ds, "SKILLS_DIRS", ["skills"])

    (skill,) = ds.discover_skills()

    assert Path(skill["location"]).is_absolute() and Path(skill["location"]).name == "SKILL.md"
    assert skill["dir"] == str(Path(skill["location"]).parent)


def test_dir_exists_at_discovery(tmp_path, monkeypatch):
    _skill(tmp_path, "x", "x", "d")
    (skill,) = _discover(tmp_path, monkeypatch)
    assert Path(skill["dir"]).is_dir() and Path(skill["dir"]) == (tmp_path / "x").resolve()


def test_symlinked_skill_md_does_not_move_dir(tmp_path, monkeypatch):
    elsewhere = tmp_path / "notes"
    elsewhere.mkdir()
    (elsewhere / "note.md").write_text("---\nname: x\ndescription: d\n---\nbody\n", encoding="utf-8")
    entry = tmp_path / "skills" / "x"
    entry.mkdir(parents=True)
    try:
        os.symlink(elsewhere / "note.md", entry / "SKILL.md")
    except OSError:
        pytest.skip("creating a symlink needs a privilege this user lacks")

    (skill,) = _discover(tmp_path / "skills", monkeypatch)

    assert Path(skill["dir"]) == entry.resolve()


# ── the index over a mixed tree ──────────────────────────────────────────────

def test_index_string_is_exact_for_a_mixed_tree(tmp_path, monkeypatch):
    dirs = [
        _raw(tmp_path, "a", 'name: quoted\ndescription: "Say \\"hi\\": twice"'),
        _raw(tmp_path, "b", "name: folded\ndescription: >\n  over\n  two lines"),
        _raw(tmp_path, "c", "name: commented\ndescription: plain # not this"),
        _raw(tmp_path, "d", 'name: broken\ndescription: fires on: "x" always'),
    ]
    descriptions = ['Say "hi": twice', "over two lines", "plain", 'fires on: "x" always']
    names = ["quoted", "folded", "commented", "broken"]

    index = skills_summary(_discover(tmp_path, monkeypatch))

    assert index == "\n\n".join(f"- name: {n}\n  description: {d}\n  dir: {p.resolve()}"
                                for n, d, p in zip(names, descriptions, dirs))


def test_skills_import_is_light():
    code = ("import sys, dostuff.helpers.skills.discover_skills; "
            "print(sorted(m for m in ('litellm', 'chromadb', 'mcp') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO)
    assert out.stdout.strip() == "[]"
