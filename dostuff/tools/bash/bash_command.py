import asyncio
import functools
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

DEFAULT_TIMEOUT_S = 60
MAX_TIMEOUT_S = 110  # stays under the loop's per-tool timeout (TOOL_TIMEOUT_S)
_DRAIN_GRACE_S = 2.0  # how long to keep reading after the shell exits (background jobs may hold the pipes)


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


async def run_shell(command: str, cwd: str = "", timeout: float = DEFAULT_TIMEOUT_S) -> tuple[int | None, str, str, str]:
    """Run `command` detached from the terminal, with no stdin.

    Returns (exit code or None if killed, stdout, stderr, note). Cancelling the caller kills the
    process tree.
    """
    _, argv = select_shell()
    proc = await asyncio.create_subprocess_exec(
        *argv, command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd or str(Path.cwd()),
        **_detach_kwargs(),
    )
    out, err = bytearray(), bytearray()
    readers = asyncio.gather(_read_into(proc.stdout, out), _read_into(proc.stderr, err))
    readers.add_done_callback(lambda f: None if f.cancelled() else f.exception())
    note = ""
    try:
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            await _kill_tree(proc)
            note = (
                f"Timed out after {timeout:.0f}s; the process and its children were killed. "
                "Long-running programs (servers, GUI apps, watchers) must be started in the background "
                "with output redirected to a file, e.g. `nohup python app.py > app.log 2>&1 &`, then "
                "check the log in a separate command."
            )
        # A background job started by the command may keep the pipes open after the shell exits
        done, _ = await asyncio.wait({readers}, timeout=_DRAIN_GRACE_S)
        if not done:
            _stop_reading(proc, readers)
            if not note:
                note = "Output capture stopped: a background process is still holding stdout/stderr."
    except asyncio.CancelledError:
        _stop_reading(proc, readers)
        await _kill_tree(proc)
        raise
    code = proc.returncode if not note.startswith("Timed out") else None
    decode = lambda b: b.decode("utf-8", errors="replace")  # noqa: E731
    return code, decode(out), decode(err), note


async def bash_command(command: str, cwd: str = "", timeout: int = DEFAULT_TIMEOUT_S) -> str:
    """Run a shell command and return its output as a single string.

    The shell is not sandboxed: commands that are not read-only require user approval.
    Commands get no stdin and are killed when they time out, so never run interactive or
    long-running programs (servers, GUI apps, watchers) in the foreground: start them in the
    background with output redirected to a file and read the file afterwards.

    Args:
        command: Shell command to run, e.g. 'ls -la'.
        cwd: Optional working directory (defaults to the project root).
        timeout: Seconds before the command and its children are killed (default 60, max 110).

    Returns:
        Combined stdout, stderr, and exit code as a formatted string.
    """
    try:
        limit = max(1, min(int(timeout or DEFAULT_TIMEOUT_S), MAX_TIMEOUT_S))
    except (TypeError, ValueError):
        limit = DEFAULT_TIMEOUT_S
    try:
        code, stdout, stderr, note = await run_shell(command, cwd, limit)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return f"$ {command}\nerror: {e}"
    text = (
        f"$ {command}\n"
        f"exit: {code if code is not None else 'killed'}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )
    return f"{text}\nnote: {note}" if note else text
