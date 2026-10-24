"""Git worktrees: run a session in its own checkout so parallel sessions don't edit the same files.

A worktree lives at `<repo>/.dostuff/worktrees/<name>` on a new branch `dostuff/<name>`, made from
the repo's HEAD. The commit it started from is kept in git config (`branch.<branch>.dostuffBase`),
so a resumed session can still tell whether the worktree has changes; `git branch -D` drops it.

Untracked local config (`.env`, `.dostuff/config.yaml`, ...) is copied in, since a fresh checkout
only has tracked files. Those copies don't count as changes.
"""
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

WORKTREES_DIR = Path(".dostuff") / "worktrees"
BRANCH_PREFIX = "dostuff/"
_BASE_KEY = "dostuffBase"
# Local, usually untracked files a session needs in its checkout too
COPIED_FILES = (".env", ".dostuff/.env", ".dostuff/config.yaml", ".dostuff/mcp_config.json")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class WorktreeError(Exception):
    """A git or validation failure, with a message fit to show the user."""


@dataclass
class Worktree:
    name: str
    path: Path  # the checkout
    branch: str  # dostuff/<name>
    repo_root: Path  # the main checkout it was made from
    base: str  # commit it started from


def _git(args: list[str], cwd: Path | str, strip: bool = True) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        raise WorktreeError("git is not installed or not on PATH.")
    except subprocess.TimeoutExpired:
        raise WorktreeError(f"git {args[0]} timed out.")
    if result.returncode != 0:
        raise WorktreeError((result.stderr or result.stdout).strip() or f"git {args[0]} failed.")
    return result.stdout.strip() if strip else result.stdout


def main_repo_root(cwd: Path | str | None = None) -> Path:
    """The main checkout of the repo containing cwd, even when cwd is inside a linked worktree."""
    cwd = Path(cwd or Path.cwd())
    try:
        common_dir = _git(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd)
    except WorktreeError:
        raise WorktreeError(f"{cwd} is not inside a git repository; worktrees need one (run `git init` first).")
    common = Path(common_dir)
    if common.name != ".git":
        raise WorktreeError("Bare repositories are not supported.")
    return common.parent.resolve()


def _new_name() -> str:
    return "wt-" + uuid.uuid4().hex[:6]


def create_worktree(name: str | None = None, cwd: Path | str | None = None) -> Worktree:
    """`git worktree add .dostuff/worktrees/<name> -b dostuff/<name>` from HEAD; copies local config in."""
    root = main_repo_root(cwd)
    name = name or _new_name()
    if not _NAME_RE.match(name) or ".." in name or name.endswith((".lock", ".")):
        raise WorktreeError(f"Invalid worktree name '{name}': use letters, digits, '.', '_' or '-'.")
    try:
        base = _git(["rev-parse", "HEAD"], cwd or root)  # cwd: a worktree branches from its own HEAD
    except WorktreeError:
        raise WorktreeError("The repository has no commits yet; make one before creating a worktree.")
    branch = BRANCH_PREFIX + name
    path = root / WORKTREES_DIR / name
    if path.exists():
        raise WorktreeError(f"A worktree named '{name}' already exists at {path}.")
    if _git(["branch", "--list", branch], root):
        raise WorktreeError(f"Branch '{branch}' already exists; pick another name.")

    # A `*` .gitignore inside the dir keeps every worktree out of `git status` without touching the user's .gitignore
    ignore = root / WORKTREES_DIR / ".gitignore"
    ignore.parent.mkdir(parents=True, exist_ok=True)
    if not ignore.exists():
        ignore.write_text("*\n", encoding="utf-8")

    _git(["worktree", "add", "-b", branch, str(path), base], root)
    _git(["config", f"branch.{branch}.{_BASE_KEY}", base], root)
    for rel in COPIED_FILES:
        src, dst = root / rel, path / rel
        if src.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return Worktree(name=name, path=path.resolve(), branch=branch, repo_root=root, base=base)


def detect_worktree(cwd: Path | str | None = None) -> Worktree | None:
    """The dostuff worktree containing cwd, or None (not a repo, the main checkout, or not one of ours)."""
    cwd = Path(cwd or Path.cwd()).resolve()
    try:
        root = main_repo_root(cwd)
        top = Path(_git(["rev-parse", "--show-toplevel"], cwd)).resolve()
    except WorktreeError:
        return None
    if top.parent != (root / WORKTREES_DIR).resolve():
        return None
    branch = BRANCH_PREFIX + top.name
    try:
        if _git(["branch", "--show-current"], top) != branch:
            return None
        base = _git(["config", f"branch.{branch}.{_BASE_KEY}"], root)
    except WorktreeError:
        return None
    return Worktree(name=top.name, path=top, branch=branch, repo_root=root, base=base)


def worktree_changes(wt: Worktree) -> tuple[list[str], int]:
    """(uncommitted paths, commits since the base); the copied config files don't count."""
    status = _git(["status", "--porcelain", "--untracked-files=all"], wt.path, strip=False)  # keep the "XY " columns
    copied = set(COPIED_FILES)
    dirty = [line[3:] for line in status.splitlines() if line[3:].strip('"') not in copied]
    commits = int(_git(["rev-list", "--count", f"{wt.base}..HEAD"], wt.path) or 0)
    return dirty, commits


def has_changes(wt: Worktree) -> bool:
    dirty, commits = worktree_changes(wt)
    return bool(dirty or commits)


def remove_worktree(wt: Worktree) -> None:
    """Delete the checkout and its branch. Callers check has_changes() first; this forces.

    The process must not be inside the worktree (Windows can't delete a cwd).
    """
    if wt.path.exists():
        _git(["worktree", "remove", "--force", str(wt.path)], wt.repo_root)
    else:
        _git(["worktree", "prune"], wt.repo_root)
    _git(["branch", "-D", wt.branch], wt.repo_root)
