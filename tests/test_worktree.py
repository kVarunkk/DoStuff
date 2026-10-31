import asyncio
import subprocess

import pytest

from dostuff.agent.permissions import _is_settings_path, has_unsafe_constructs
from dostuff.helpers.tools.project_root import project_root
from dostuff.tools import delegate_to_subagent as delegate_module
from dostuff.tools.files.write_file import write_file
from dostuff.lib.worktree import (WorktreeError, create_worktree, detect_worktree, has_changes, remove_worktree,
                                  worktree_changes)


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("a\n")
    (root / ".gitignore").write_text(".env\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "init")
    (root / ".env").write_text("KEY=1\n")  # ignored, so a fresh checkout lacks it
    return root.resolve()


def test_create_detect_and_remove_unchanged(repo):
    wt = create_worktree("feat", repo)
    assert wt.path == repo / ".dostuff" / "worktrees" / "feat"
    assert wt.branch == "dostuff/feat"
    assert (wt.path / "a.txt").read_text() == "a\n"
    assert (wt.path / ".env").read_text() == "KEY=1\n"
    assert _git(repo, "status", "--porcelain") == ""  # the worktrees dir ignores itself

    found = detect_worktree(wt.path)
    assert found is not None and found.name == "feat" and found.base == wt.base
    assert detect_worktree(repo) is None
    assert not has_changes(wt)  # the copied .env doesn't count

    remove_worktree(wt)
    assert not wt.path.exists()
    assert _git(repo, "branch", "--list", "dostuff/feat") == ""


def test_changes_are_detected(repo):
    wt = create_worktree("edit", repo)
    (wt.path / "a.txt").write_text("changed\n")
    assert worktree_changes(wt) == (["a.txt"], 0)
    _git(wt.path, "commit", "-qam", "change")
    assert worktree_changes(wt) == ([], 1)
    remove_worktree(wt)
    assert not wt.path.exists()


def test_errors(repo, tmp_path):
    create_worktree("dup", repo)
    with pytest.raises(WorktreeError, match="already exists"):
        create_worktree("dup", repo)
    with pytest.raises(WorktreeError, match="Invalid"):
        create_worktree("../x", repo)
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(WorktreeError, match="not inside a git repository"):
        create_worktree(None, plain)


def test_settings_path_is_relative_to_cwd(tmp_path, monkeypatch):
    wt = tmp_path / ".dostuff" / "worktrees" / "x"
    wt.mkdir(parents=True)
    monkeypatch.chdir(wt)
    assert not _is_settings_path(str(wt / "src" / "main.py"))
    assert _is_settings_path(str(wt / ".dostuff" / "config.yaml"))
    assert _is_settings_path(".dostuff/config.yaml")
    assert not _is_settings_path("src/main.py")


def _delegate(monkeypatch, repo, writes: bool, isolation="worktree"):
    """Run delegate_to_subagent with a fake loop that writes a file through the real write_file tool."""
    seen = {}

    async def fake_loop(task, role, mcp_client, session_id, turn_id, adapter, permissions,
                        instruction, working_history, tool_names, owner=None):
        seen["root"] = project_root()
        seen["instruction"] = instruction
        if writes:
            await asyncio.to_thread(write_file, "new.txt", "hi\n")  # threads see the subagent's root too
        return "report"

    monkeypatch.chdir(repo)
    monkeypatch.setattr(delegate_module, "_run_subagent_loop", fake_loop)
    result = asyncio.run(delegate_module._impl_delegate_to_subagent(
        task="t", role="coder", mcp_client=object(), session_id="s", turn_id="t",  # type: ignore[arg-type]
        requested_tools=["write_file"], isolation=isolation,
    ))
    return result, seen


def test_subagent_worktree_removed_when_unchanged(monkeypatch, repo):
    result, seen = _delegate(monkeypatch, repo, writes=False)
    assert seen["root"].parent == repo / ".dostuff" / "worktrees"
    assert "git worktree" in seen["instruction"]
    assert "changed no files" in result and not seen["root"].exists()
    assert project_root() == repo  # override reset after the subagent


def test_subagent_worktree_kept_with_changes(monkeypatch, repo):
    result, seen = _delegate(monkeypatch, repo, writes=True)
    assert (seen["root"] / "new.txt").read_text() == "hi\n"
    assert not (repo / "new.txt").exists()  # the main checkout is untouched
    assert "changes are kept in .dostuff/worktrees/" in result and "new.txt" in result


def test_subagent_isolation_errors(monkeypatch, repo, tmp_path):
    result, _ = _delegate(monkeypatch, repo, writes=False, isolation="docker")
    assert result.startswith("[error] Unknown isolation")
    plain = tmp_path / "plain"
    plain.mkdir()
    result, seen = _delegate(monkeypatch, plain, writes=False)
    assert "not inside a git repository" in result and not seen


def test_shell_may_name_worktree_paths():
    assert not has_unsafe_constructs("git -C .dostuff/worktrees/agent-1 diff")
    assert has_unsafe_constructs("cat .dostuff/worktrees/../config.yaml")
    assert has_unsafe_constructs("cat .dostuff/worktrees/x/.dostuff/config.yaml")
    assert has_unsafe_constructs("cat .dostuff/config.yaml")
