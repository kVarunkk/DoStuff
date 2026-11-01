from dostuff.helpers.skills import discover_skills as ds


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
