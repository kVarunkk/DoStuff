"""The <environment> block added to the system prompt each turn (cwd, OS, shell, date, git)."""
import datetime
import platform
import subprocess
from pathlib import Path

from dostuff.tools.bash.bash_command import select_shell

GIT_STATUS_MAX_LINES = 40


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def build_environment_block(cwd: Path | None = None) -> str:
    """Blocking (runs git); call it via asyncio.to_thread from async code."""
    cwd = cwd or Path.cwd()
    lines = [
        f"cwd: {cwd}",
        f"os: {platform.system()} {platform.release()}",
        f"shell: {select_shell()[0]} (used by bash_command)",
        f"date: {datetime.date.today().isoformat()}",
    ]
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    if branch is not None:
        lines.append(f"git branch: {branch.strip()}")
        status = (_git(["status", "--short"], cwd) or "").splitlines()
        if status:
            shown = status[:GIT_STATUS_MAX_LINES]
            if len(status) > GIT_STATUS_MAX_LINES:
                shown.append(f"... {len(status) - GIT_STATUS_MAX_LINES} more")
            lines.append("git status --short:\n" + "\n".join(shown))
        else:
            lines.append("git status: clean")
    else:
        lines.append("git: not a repository")
    return "<environment>\n" + "\n".join(lines) + "\n</environment>"
