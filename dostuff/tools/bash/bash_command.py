import asyncio
import functools
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dostuff.tools.bash.jobs import Job


DEFAULT_TIMEOUT_S = 60
MAX_TIMEOUT_S = 110  # stays under the loop's per-tool timeout (TOOL_TIMEOUT_S)
IMMEDIATE_EXIT_S = 0.2  # a background job gone this fast is reported in its start result
IMMEDIATE_EXIT_LINES = 10
DETACHED_RE = re.compile(r'^moved to background by the user after \S+ as job "([^"]+)"', re.M)


@functools.lru_cache(maxsize=1)
def select_shell() -> tuple[str, list[str]]:
    """Pick the shell commands run in: bash (Git Bash on Windows) → PowerShell → /bin/sh.

    Returns (display name, argv prefix); the command string is appended to the prefix.
    """
    if sys.platform == "win32":
        # Prefer Git Bash; System32\bash.exe is the WSL launcher, which runs in a different filesystem.
        git = shutil.which("git")
        candidates = []
        if git:
            git_root = Path(git).resolve().parent.parent
            candidates += [git_root / "bin" / "bash.exe", git_root / "usr" / "bin" / "bash.exe"]
        found = shutil.which("bash")
        if found and "system32" not in found.lower():
            candidates.append(Path(found))
        for bash in candidates:
            if bash.is_file():
                return "bash", [str(bash), "-c"]
        for ps in ("pwsh", "powershell"):
            ps_path = shutil.which(ps)
            if ps_path:
                return "PowerShell", [ps_path, "-NoProfile", "-NonInteractive", "-Command"]
        return "cmd", [os.environ.get("COMSPEC", "cmd.exe"), "/c"]
    bash_path = shutil.which("bash")
    if bash_path:
        return "bash", [bash_path, "-c"]
    return "sh", ["/bin/sh", "-c"]


def _detach_kwargs() -> dict:
    """Keep the child off the TUI's terminal.

    Without this the child inherits our console: it can read the keystrokes and mouse events
    meant for the TUI and change the console mode, which freezes the UI until it exits.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the shell and everything it started (servers, apps, background jobs)."""
    try:
        if sys.platform == "win32":
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/F", "/T", "/PID", str(proc.pid),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            await asyncio.wait_for(killer.wait(), timeout=5)
        else:
            os.killpg(proc.pid, signal.SIGKILL)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


async def _read_into(stream: asyncio.StreamReader | None, buf: bytearray) -> None:
    if stream is None:
        return
    while chunk := await stream.read(65536):
        buf.extend(chunk)


def _stop_reading(proc: asyncio.subprocess.Process, readers: asyncio.Future) -> None:
    readers.cancel()
    for stream in (proc.stdout, proc.stderr):
        transport = getattr(stream, "_transport", None)
        if transport is not None:
            transport.close()


def _read_log(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""


async def run_shell(command: str, cwd: str = "", timeout: float = DEFAULT_TIMEOUT_S, owner: str = "") -> tuple[int | None, str, str, str]:
    """Run `command` detached from the terminal, with no stdin, as a foreground job (spec D1).

    Output goes to the job's log files, so a background process started by the command can't hold
    the call open. Returns (exit code or None if killed, stdout, stderr, note). Cancelling the
    caller kills the process tree. `owner` defaults to the running session (`"user"` for `!command`).
    """
    from dostuff.lib.model import current_session_id
    from dostuff.tools.bash.jobs import get_registry

    registry = get_registry()
    job = await registry.start(command, cwd, owner or current_session_id.get(), background=False)
    note = ""
    try:
        outcome = await registry.wait(job, timeout)
        if outcome == "timeout":
            await registry.kill(job, by="timeout")
            note = (
                f"Timed out after {timeout:.0f}s; the process and its children were killed. "
                "Start long-running programs (servers, GUI apps, watchers) with run_in_background=true "
                "instead, then read their output with bash_output."
            )
        elif outcome == "detached":
            note = _detached_note(job)
    except asyncio.CancelledError:
        await registry.kill(job, by="user")
        raise
    code = job.exit_code if outcome == "exited" else None
    return code, _read_log(job.out_path), _read_log(job.err_path), note


def _detached_note(job: "Job") -> str:
    """Ctrl+B's result body (spec D9). The shown output is consumed, so bash_output returns what comes after."""
    from dostuff.tools.bash.jobs import NOTICE_TAIL_LINES, fmt_runtime, get_registry, indent_block

    output = get_registry().read_new(job)
    return (
        f'moved to background by the user after {fmt_runtime(job.runtime)} as job "{job.id}" (pid {job.pid})\n'
        f"stdout so far:\n{indent_block(output.stdout.splitlines()[-NOTICE_TAIL_LINES:])}\n"
        f"stderr so far:\n{indent_block(output.stderr.splitlines()[-NOTICE_TAIL_LINES:])}\n"
        f'Use bash_output("{job.id}") to read more and kill_shell("{job.id}") to stop it.'
    )


def detached_job_id(result: str) -> str | None:
    """The job id when a bash_command result says Ctrl+B moved it to the background."""
    match = DETACHED_RE.search(result)
    return match.group(1) if match else None


async def _start_background(command: str, cwd: str, background_id: str, owner: str) -> str:
    """Start a background job and return at once (spec D4)."""
    from dostuff.lib.model import current_session_id
    from dostuff.tools.bash.jobs import get_registry

    registry = get_registry()
    owner = owner or current_session_id.get()
    try:
        job = await registry.start(command, cwd, owner, background=True, job_id=background_id)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return f"$ {command}\nerror: {e}"
    # A quick exit is almost always a typo or a port in use: say so now rather than in a notice
    if await registry.wait(job, IMMEDIATE_EXIT_S) == "exited":
        registry.discard_notices(owner, job)
        output = registry.read_new(job)
        stream, text = ("stderr", output.stderr) if output.stderr.strip() else ("stdout", output.stdout)
        head = "".join(text.splitlines(keepends=True)[:IMMEDIATE_EXIT_LINES])
        text = f'$ {command}\nbackground job "{job.id}" exited immediately with code {job.exit_code}\n'
        return text + f"{stream}:\n{head}" if head else text
    return (
        f"$ {command}\n"
        f'started background job "{job.id}" (pid {job.pid})\n'
        f"stdout: {job.out_path}\n"
        f"stderr: {job.err_path}\n"
        f'Use bash_output("{job.id}") to read its output and kill_shell("{job.id}") to stop it.'
    )


async def bash_command(
    command: str, cwd: str = "", timeout: int = DEFAULT_TIMEOUT_S,
    run_in_background: bool = False, background_id: str = "", owner: str = "",
) -> str:
    """Run a shell command and return its output as a single string.

    The shell is not sandboxed: commands that are not read-only require user approval.
    Commands get no stdin and are killed when they time out, so never run interactive or
    long-running programs (servers, GUI apps, watchers) in the foreground: start them with
    run_in_background=true, then read their output with bash_output and stop them with kill_shell.

    Args:
        command: Shell command to run, e.g. 'ls -la'.
        cwd: Optional working directory (defaults to the project root).
        timeout: Seconds before the command and its children are killed (default 60, max 110). Ignored for background jobs.
        run_in_background: Start the command as a background job and return at once instead of waiting for it to exit.
        background_id: Optional short name for a background job, e.g. 'dev' (lowercase letters, digits, '.', '_', '-'); one is generated if omitted.

    Returns:
        Combined stdout, stderr, and exit code as a formatted string; for a background job, its id, pid and log files.
    """
    if run_in_background:
        return await _start_background(command, cwd, background_id, owner)
    try:
        limit = max(1, min(int(timeout or DEFAULT_TIMEOUT_S), MAX_TIMEOUT_S))
    except (TypeError, ValueError):
        limit = DEFAULT_TIMEOUT_S
    try:
        code, stdout, stderr, note = await run_shell(command, cwd, limit, owner=owner)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return f"$ {command}\nerror: {e}"
    if DETACHED_RE.match(note):
        return f"$ {command}\n{note}"
    text = (
        f"$ {command}\n"
        f"exit: {code if code is not None else 'killed'}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )
    return f"{text}\nnote: {note}" if note else text
