import os
import subprocess
import sys
from pathlib import Path

import pytest

from dostuff.agent.instructions import skills_summary
from dostuff.helpers.skills import discover_skills as ds
from dostuff.helpers.tools.resolve_safe_path import set_skill_roots, skill_roots

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


def test_discover_publishes_skill_roots(tmp_path, monkeypatch):
    _skill(tmp_path, "x", "x", "d")
    _skill(tmp_path, "y", "y", "d")
    (tmp_path / "not-a-skill").mkdir()

    skills = _discover(tmp_path, monkeypatch)

    assert skill_roots() == {Path(s["dir"]).resolve() for s in skills}


def test_skill_roots_are_resolved_and_deduped(tmp_path):
    (tmp_path / "x").mkdir()
    set_skill_roots([tmp_path / "x", tmp_path / "x" / ".." / "x"])
    assert skill_roots() == {(tmp_path / "x").resolve()}


def test_readme_documents_skills():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start = readme.index("\n## Skills\n")
    skills = readme[start:readme.index("\n## ", start + 1)]
    assert "`dir`" in skills and "`read_file`" in skills
    folders = ["<cwd>/.dostuff/skills/", "<cwd>/.agents/skills/", "~/.dostuff/skills/", "~/.agents/skills/", "<cwd>/skills/"]
    positions = [skills.index(folder) for folder in folders]
    assert positions == sorted(positions)
    assert "No deduplication" not in readme


def test_skills_import_is_light():
    code = ("import sys, dostuff.helpers.skills.discover_skills; "
            "print(sorted(m for m in ('litellm', 'chromadb', 'mcp') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO)
    assert out.stdout.strip() == "[]"


# --- disable-model-invocation (plan/2026-11-06-skills-and-multi-edit D7 - D9) ----------------------------------

@pytest.mark.parametrize("line, hidden", [
    ("disable-model-invocation: true", True),
    ("disable-model-invocation: True", True),
    ("disable-model-invocation: yes", True),
    ("disable-model-invocation: on", True),
    ("disable-model-invocation: 1", True),
    ("disable-model-invocation: banana", True),  # a typo must not make a protected skill visible
    ("disable-model-invocation: [a, b]", True),  # not even a string
    ("disable-model-invocation: {a: b}", True),
    ("disable-model-invocation: false", False),
    ("disable-model-invocation: FALSE", False),
    ("disable-model-invocation: no", False),
    ("disable-model-invocation: off", False),
    ("disable-model-invocation: 0", False),
    ("disable-model-invocation:", False),
    ("disable-model-invocation: ''", False),
    ("other-key: true", False),
])
def test_disable_model_invocation_values(tmp_path, monkeypatch, line, hidden):
    skill = _only(tmp_path, monkeypatch, f"name: x\ndescription: d\n{line}")
    assert skill["disable_model_invocation"] is hidden


@pytest.mark.parametrize("line, hidden", [
    ("disable-model-invocation: true", True),
    ("disable-model-invocation: banana", True),
    ("disable-model-invocation: false", False),
    ("disable-model-invocation:", False),
    ("", False),
])
def test_disable_model_invocation_invalid_yaml(tmp_path, monkeypatch, line, hidden):
    """A broken SKILL.md must not lose its protection: the regex fallback reads this key too."""
    description = 'Ship it. Auto-fires on: "deploy" and more'  # the unquoted colon makes this invalid YAML
    skill = _only(tmp_path, monkeypatch, f"name: x\ndescription: {description}\n{line}")
    assert skill["description"] == description  # proof the fallback ran
    assert skill["disable_model_invocation"] is hidden


def test_manual_skill_stays_readable(tmp_path, monkeypatch):
    """Hidden, not blocked (D9): `/name` tells the model to use the files next to SKILL.md."""
    from dostuff.tools.files.read_file import read_file
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    home_skills = tmp_path / "home-skills"
    skill_dir = _raw(home_skills, "deploy", "name: deploy\ndescription: d\ndisable-model-invocation: true")
    (skill_dir / "steps.md").write_text("step one\n", encoding="utf-8")
    try:
        (skill,) = _discover(home_skills, monkeypatch)
        assert skill["dir"] in {str(root) for root in skill_roots()}
        assert "step one" in read_file(str(skill_dir / "steps.md"))
    finally:
        set_skill_roots([])
