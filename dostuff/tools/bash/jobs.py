"""Shell jobs: every command the shell tools run is a job in one process-wide registry.

A job's stdout/stderr go straight into two files under `~/.dostuff/jobs/<session_id>/` (no pipes),
a watcher task notices when it exits, and background jobs that end on their own leave a notice for
their owner's next model call. See plan/2026-10-31-background-shell/spec.md (D2, D3, D7, D8, D11).
"""
import asyncio
import atexit
import codecs
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal

from dostuff.helpers.tools.project_root import project_root
from dostuff.tools.bash import bash_command as _bash  # select_shell / _detach_kwargs, looked up per call

logger = logging.getLogger(__name__)

POLL_S = 0.2  # wait_for polling interval (same as hooks)
KILL_WAIT_S = 5.0
MAX_BACKGROUND_JOBS = 8  # running background jobs per owner
MIN_WAIT_S, MAX_WAIT_S = 1, 110  # wait_for timeout clamp; stays under the loop's TOOL_TIMEOUT_S
NOTICE_TAIL_LINES = 20
NOTICE_MAX_CHARS = 2000
_PARTIAL_FLUSH_BYTES = 4096  # hand out a line with no newline yet once it gets this long (progress bars)
_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,31}")

Status = Literal["running", "exited", "killed"]
KilledBy = Literal["model", "user", "timeout", "owner_ended"]
JobEvent = Literal["start", "detach", "end"]


class JobError(ValueError):
    """A refused start or an unknown job; the tools turn the message into an `error:` result."""


def jobs_root() -> Path:
    return Path.home() / ".dostuff" / "jobs"


def session_of(owner: str) -> str:
    """The session id an owner belongs to: subagent owners are `<session_id>/sub/<run_id>`."""
    return owner.split("/sub/", 1)[0]


def _job_dir(session: str) -> Path:
    return jobs_root() / (re.sub(r"[^\w.-]", "_", session) or "_")


def _log_paths(session: str, uid: str) -> tuple[Path, Path]:
    d = _job_dir(session)
    return d / f"{uid}.out", d / f"{uid}.err"


def fmt_runtime(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def _tail(path: Path, n_lines: int, max_bytes: int = 65536) -> list[str]:
    """Last `n_lines` lines of a log file (reads at most `max_bytes` from the end)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n").splitlines()[-n_lines:]


@dataclass
class Job:
    uid: str            # unique, used for the log file names
    id: str             # what the model and user see: "dev", "bg-3"
    owner: str          # main: session_id; subagent: f"{session_id}/sub/{run_id}"
    command: str
    cwd: str
    proc: asyncio.subprocess.Process
    started: float
    out_path: Path
    err_path: Path
    foreground: bool    # a caller is currently waiting on it
    ended: float | None = None
    exit_code: int | None = None
    status: Status = "running"
    killed_by: KilledBy | None = None
    out_offset: int = 0  # bash_output's read position in .out
    err_offset: int = 0  # ... and in .err
    _decoders: dict = field(default_factory=dict, repr=False)
    _exited: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _detached: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _watcher: "asyncio.Task | None" = field(default=None, repr=False)

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def runtime(self) -> float:
        return (self.ended or time.time()) - self.started

    def describe(self) -> str:
        """`"dev" (pid 4812, 3m12s)` — used in refusals and job lists."""
        return f'"{self.id}" (pid {self.pid}, {fmt_runtime(self.runtime)})'

    def tail(self, stream: Literal["stdout", "stderr"], n_lines: int) -> list[str]:
        """Last lines of a stream's log, without moving the model's read offsets (`/jobs <id>`)."""
        return _tail(self.out_path if stream == "stdout" else self.err_path, n_lines)


@dataclass
class JobOutput:
    stdout: str
    stderr: str


@dataclass
class WaitForResult:
    reason: Literal["matched", "exited", "timeout"]
    output: JobOutput


@dataclass
class Notice:
    id: str
    status: Status
    code: int | None
    runtime: float
    killed_by: KilledBy | None
    stdout_tail: list[str]
    stderr_tail: list[str]


def clamp_wait_timeout(timeout: object) -> float:
    try:
        value = float(timeout)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = 30.0
    return max(MIN_WAIT_S, min(value, MAX_WAIT_S))


def _kill_argv(pid: int) -> list[str] | None:
    """The command that kills a process tree on Windows; None means use `os.killpg` (POSIX)."""
    return ["taskkill", "/F", "/T", "/PID", str(pid)] if sys.platform == "win32" else None


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[tuple[str, str], Job] = {}
        self._counters: dict[str, int] = {}
        self._notices: dict[str, list[Notice]] = {}
        self._listeners: list[Callable[[Job, JobEvent], None]] = []
        self._roles: dict[str, str] = {}  # subagent owner -> role, for /jobs
        self._atexit_registered = False

    # --- listeners (loop `job` events, the TUI status bar) ------------------------------------

    def add_listener(self, fn: "Callable[[Job, JobEvent], None]") -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn: "Callable[[Job, JobEvent], None]") -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _notify(self, job: "Job", event: JobEvent) -> None:
        for fn in list(self._listeners):
            try:
                fn(job, event)
            except Exception as e:  # a broken listener must not break the job
                logger.warning("job listener failed: %r", e)

    # --- owners -----------------------------------------------------------------------------

    def set_role(self, owner: str, role: str) -> None:
        """Remember a subagent owner's role so `/jobs` can name it."""
        self._roles[owner] = role

    def owner_label(self, owner: str) -> str:
        """How `/jobs` names an owner: `main`, `subagent: <role>` or `user` (a `!command`)."""
        if "/sub/" in owner:
            role = self._roles.get(owner)
            return f"subagent: {role}" if role else "subagent"
        return "user" if owner == "user" else "main"

    def transfer(self, old_owner: str, new_owner: str) -> list[Job]:
        """Hand `old_owner`'s running background jobs and queued notices to `new_owner` (/clear, /resume, D8).

        The jobs keep running and their log files stay where they are; ended jobs stay with the old owner.
        """
        moved = [j for j in self.jobs(old_owner) if j.ended is None and not j.foreground]
        for job in moved:
            del self._jobs[(old_owner, job.id)]
            job.owner = new_owner
            self._jobs[(new_owner, job.id)] = job
        if old_owner in self._notices:
            self._notices.setdefault(new_owner, []).extend(self._notices.pop(old_owner))
        # never reuse a bg-<n> the jobs already carry
        self._counters[new_owner] = max(self._counters.get(new_owner, 0), self._counters.get(old_owner, 0))
        if moved:
            self._write_index(session_of(old_owner))
            self._write_index(session_of(new_owner))
        return moved

    # --- start ------------------------------------------------------------------------------

    def _check_start(self, owner: str, job_id: str, background: bool) -> str:
        """Validate / assign the id and apply the collision and limit rules; returns the id."""
        if job_id:
            job_id = job_id.casefold()
            if not _ID_RE.fullmatch(job_id):
                raise JobError(f"invalid background_id {job_id!r}: must match {_ID_RE.pattern}")
        else:
            n = self._counters.get(owner, 0)
            while True:
                n += 1
                job_id = f"bg-{n}"
                old = self._jobs.get((owner, job_id))
                if old is None or old.status != "running":
                    break
            self._counters[owner] = n
        old = self._jobs.get((owner, job_id))
        if old is not None and old.status == "running":
            raise JobError(
                f"job {old.describe()} is already running; kill it with kill_shell or pick another background_id"
            )
        if background:
            running = [j for j in self.jobs(owner) if j.status == "running" and not j.foreground]
            if len(running) >= MAX_BACKGROUND_JOBS:
                names = ", ".join(j.describe() for j in running)
                raise JobError(
                    f"too many background jobs ({len(running)} running, limit {MAX_BACKGROUND_JOBS}): {names}. "
                    "Stop one with kill_shell first."
                )
        return job_id

    async def start(self, command: str, cwd: str, owner: str, *, background: bool, job_id: str = "") -> Job:
        """Spawn `command` in the platform shell with its output going to the job's log files."""
        job_id = self._check_start(owner, job_id, background)
        workdir = project_root() / cwd  # an absolute cwd replaces the root
        if not workdir.is_dir():
            raise JobError(f"working directory not found: {workdir}")
        _, argv = _bash.select_shell()
        uid = uuid.uuid4().hex[:12]
        session = session_of(owner)
        out_path, err_path = _log_paths(session, uid)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(out_path, "wb")
        err_f = open(err_path, "wb")
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, command,
                stdin=subprocess.DEVNULL, stdout=out_f, stderr=err_f,
                cwd=str(workdir), **_bash._detach_kwargs(),
            )
        except BaseException:
            out_f.close()
            err_f.close()
            for p in (out_path, err_path):
                p.unlink(missing_ok=True)
            raise
        finally:
            out_f.close()  # the child has its own handles; holding ours would lock the files on Windows
            err_f.close()
        job = Job(
            uid=uid, id=job_id, owner=owner, command=command, cwd=str(workdir), proc=proc,
            started=time.time(), out_path=out_path, err_path=err_path, foreground=not background,
        )
        self._jobs[(owner, job_id)] = job
        if not self._atexit_registered:  # last resort when nobody stopped them (CON-008)
            atexit.register(self.kill_all_sync)
            self._atexit_registered = True
        job._watcher = asyncio.create_task(self._watch(job))
        job._watcher.add_done_callback(_log_watcher_error)
        self._write_index(session)
        self._notify(job, "start")
        return job

    async def _watch(self, job: Job) -> None:
        code = await job.proc.wait()
        self._finish(job, code)

    def _finish(self, job: Job, code: int | None) -> None:
        """Record the end of a job once (watcher, kill and sync kill all land here)."""
        if job.ended is not None:
            return
        job.ended = time.time()
        job.exit_code = code
        job.status = "killed" if job.killed_by else "exited"
        job._exited.set()
        if not job.foreground and job.killed_by in (None, "user"):
            self._notices.setdefault(job.owner, []).append(Notice(
                id=job.id, status=job.status, code=code, runtime=job.runtime, killed_by=job.killed_by,
                stdout_tail=_tail(job.out_path, NOTICE_TAIL_LINES),
                stderr_tail=_tail(job.err_path, NOTICE_TAIL_LINES),
            ))
        self._write_index(session_of(job.owner))
        self._notify(job, "end")

    def _write_index(self, session: str) -> None:
        """Rewrite `jobs.json` for the session (atomically) so a later run can spot orphans (D11)."""
        entries = [
            {"id": j.id, "uid": j.uid, "owner": j.owner, "pid": j.pid, "command": j.command,
             "started": j.started, "ended": j.ended, "status": j.status, "exit_code": j.exit_code}
            for j in self._jobs.values() if session_of(j.owner) == session
        ]
        path = _job_dir(session) / "jobs.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(entries, indent=1), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("could not write %s: %s", path, e)

    # --- lookup -----------------------------------------------------------------------------

    def jobs(self, owner: str | None = None) -> list[Job]:
        """An owner's jobs, or every job (the user's view) when `owner` is None; oldest first."""
        return [j for (o, _), j in self._jobs.items() if owner is None or o == owner]

    def get(self, owner: str, job_id: str) -> Job:
        job = self._jobs.get((owner, (job_id or "").casefold()))
        if job is None:
            mine = self.jobs(owner)
            known = ", ".join(f'"{j.id}" ({j.status})' for j in mine) if mine else "none"
            raise JobError(f'no job "{job_id}". Your jobs: {known}')
        return job

    # --- waiting ----------------------------------------------------------------------------

    async def wait(self, job: Job, timeout: float | None) -> Literal["exited", "detached", "timeout"]:
        """Wait for the job to exit or be detached (Ctrl+B). On timeout the caller decides what to do."""
        if job.ended is not None:
            return "exited"
        exited = asyncio.ensure_future(job._exited.wait())
        detached = asyncio.ensure_future(job._detached.wait())
        try:
            done, _ = await asyncio.wait({exited, detached}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            exited.cancel()
            detached.cancel()
        if exited in done or job.ended is not None:
            return "exited"
        if detached in done:
            return "detached"
        return "timeout"

    def detach(self, job: Job) -> None:
        """Turn a foreground job into a background one; the process and its files are untouched."""
        if job.ended is not None:
            return
        job.foreground = False
        job._detached.set()
        self._notify(job, "detach")

    def detach_foreground(self, owner: str) -> list[Job]:
        """Detach every running foreground job of `owner` and its subagents (Ctrl+B)."""
        detached = []
        for job in self.jobs():
            if job.foreground and job.ended is None and (job.owner == owner or job.owner.startswith(owner + "/sub/")):
                self.detach(job)
                detached.append(job)
        return detached

    # --- reading ----------------------------------------------------------------------------

    def _new_bytes(self, job: Job, stream: str) -> tuple[bytes, int]:
        """Unread bytes of a stream and the offset after them; a trailing partial line is held back."""
        path = job.out_path if stream == "out" else job.err_path
        offset = job.out_offset if stream == "out" else job.err_offset
        try:
            with open(path, "rb") as f:
                f.seek(offset)
                data = f.read()
        except OSError:
            return b"", offset
        if job.ended is None:
            cut = data.rfind(b"\n") + 1
            if cut == 0 and len(data) < _PARTIAL_FLUSH_BYTES:
                return b"", offset
            if cut:
                data = data[:cut]
        return data, offset + len(data)

    def read_new(self, job: Job, filter: str = "", cap: int = 0) -> JobOutput:
        """Output written since the last read, per stream. `filter` keeps matching lines only (the rest
        is still consumed); `cap` keeps the latest `cap` chars per stream behind a skip header."""
        try:
            pattern = re.compile(filter) if filter else None
        except re.error as e:
            raise JobError(f"invalid filter regex {filter!r}: {e}") from e
        texts = []
        for stream in ("out", "err"):
            data, new_offset = self._new_bytes(job, stream)
            decoder = job._decoders.setdefault(stream, codecs.getincrementaldecoder("utf-8")(errors="replace"))
            text = decoder.decode(data, final=job.ended is not None).replace("\r\n", "\n")
            if stream == "out":
                job.out_offset = new_offset
            else:
                job.err_offset = new_offset
            if pattern is not None:
                text = "".join(line for line in text.splitlines(keepends=True) if pattern.search(line))
            texts.append(self._cap(text, cap, job.out_path if stream == "out" else job.err_path))
        return JobOutput(*texts)

    @staticmethod
    def _cap(text: str, cap: int, path: Path) -> str:
        if not cap or len(text) <= cap:
            return text
        header = f"[… {{}} earlier bytes skipped; full log: {path}]\n"
        budget = max(0, cap - len(header.format(len(text.encode("utf-8")))))
        cut = len(text) - budget
        nl = text.find("\n", cut - 1)  # start the kept part on a line boundary when one is near
        if nl != -1 and nl + 1 < len(text):
            cut = nl + 1
        skipped = len(text[:cut].encode("utf-8"))
        return header.format(skipped) + text[cut:]

    async def wait_for(self, job: Job, pattern: str, timeout: float, filter: str = "", cap: int = 0) -> WaitForResult:
        """Block until a new line matches `pattern`, the job ends, or `timeout` passes, then
        `read_new(job, filter, cap)`. Cancelling leaves the job running and its offsets where they were."""
        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise JobError(f"invalid wait_for regex {pattern!r}: {e}") from e
        if filter:
            try:
                re.compile(filter)
            except re.error as e:
                raise JobError(f"invalid filter regex {filter!r}: {e}") from e
        deadline = time.monotonic() + clamp_wait_timeout(timeout)
        while True:
            reason: Literal["matched", "exited", "timeout"] | None = None
            for stream in ("out", "err"):
                data, _ = self._new_bytes(job, stream)
                text = data.decode("utf-8", errors="replace")
                if any(regex.search(line) for line in text.splitlines()):
                    reason = "matched"
            if reason is None and job.ended is not None:
                reason = "exited"
            if reason is None and time.monotonic() >= deadline:
                reason = "timeout"
            if reason is not None:
                return WaitForResult(reason, self.read_new(job, filter, cap))
            await asyncio.sleep(POLL_S)

    # --- killing ----------------------------------------------------------------------------

    async def kill(self, job: Job, by: KilledBy) -> Status:
        """Kill the job's process tree and wait (bounded) for it to go. Killing an ended job is a no-op."""
        if job.ended is not None:
            return job.status
        job.killed_by = by
        argv = _kill_argv(job.pid)
        try:
            if argv:
                killer = await asyncio.create_subprocess_exec(
                    *argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,  # type: ignore[attr-defined]
                )
                await asyncio.wait_for(killer.wait(), timeout=KILL_WAIT_S)
            else:
                os.killpg(job.pid, signal.SIGKILL)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            job.proc.kill()
        except Exception:
            pass
        try:
            code: int | None = await asyncio.wait_for(job.proc.wait(), timeout=KILL_WAIT_S)
        except (asyncio.TimeoutError, Exception):
            code = job.proc.returncode
        self._finish(job, code)
        return job.status

    async def kill_owner(self, owner: str, by: KilledBy) -> list[Job]:
        """Kill every running job of exactly this owner (a subagent run ending, D8)."""
        running = [j for j in self.jobs(owner) if j.ended is None]
        await asyncio.gather(*(self.kill(j, by) for j in running))
        return running

    async def kill_all(self, by: KilledBy = "owner_ended") -> list[Job]:
        running = [j for j in self.jobs() if j.ended is None]
        await asyncio.gather(*(self.kill(j, by) for j in running))
        return running

    def kill_all_sync(self, by: KilledBy = "owner_ended") -> list[Job]:
        """Best-effort synchronous kill of every running job, for `atexit` (no event loop needed)."""
        running = [j for j in self.jobs() if j.ended is None]
        for job in running:
            job.killed_by = by
            argv = _kill_argv(job.pid)
            try:
                if argv:
                    subprocess.run(
                        argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW,  # type: ignore[attr-defined]
                        timeout=KILL_WAIT_S,
                    )
                else:
                    os.killpg(job.pid, signal.SIGKILL)  # type: ignore[attr-defined]
            except Exception:
                pass
            self._finish(job, None)
        return running

    # --- notices ----------------------------------------------------------------------------

    def drain_notices(self, owner: str) -> list[Notice]:
        return self._notices.pop(owner, [])

    def pending_notices(self, owner: str) -> list[Notice]:
        """The owner's queued notices, left in place for the loop to drain (the TUI's idle wake-up, D15)."""
        return list(self._notices.get(owner, []))

    def discard_notices(self, owner: str, job: Job) -> None:
        """Drop queued notices for `job` (its start result already reported how it ended)."""
        queue = self._notices.get(owner)
        if queue:
            queue[:] = [n for n in queue if n.id != job.id]


def indent_block(lines: list[str]) -> str:
    """Tool-result body for a stream: lines indented two spaces, or `  (none)`."""
    return "\n".join(f"  {line}" for line in lines) if lines else "  (none)"


def format_status(job: Job) -> str:
    """`running (pid 4812, 2m10s)` / `exited (code 1)` / `killed (by model)` — shared by the shell tools."""
    if job.status == "running":
        return f"running (pid {job.pid}, {fmt_runtime(job.runtime)})"
    if job.status == "killed":
        return f"killed (by {job.killed_by})"
    return f"exited (code {job.exit_code})"


def format_notice(notices: Iterable[Notice]) -> str:
    """The `<background-job …>` blocks the loop appends as one user step (D7)."""
    blocks = []
    for n in notices:
        attrs = f'id="{n.id}" status="{n.status}"'
        if n.code is not None:
            attrs += f' code="{n.code}"'
        if n.killed_by:
            attrs += f' killed_by="{n.killed_by}"'
        attrs += f' runtime="{fmt_runtime(n.runtime)}"'
        body = []
        budget = NOTICE_MAX_CHARS
        for name, lines in (("stdout", n.stdout_tail), ("stderr", n.stderr_tail)):
            if not lines:
                continue
            section = f"{name} (last {len(lines)} lines):\n" + "\n".join(f"  {line}" for line in lines)
            if len(section) > budget:
                section = section[len(section) - budget:]
                section = "…" + section[section.find("\n"):] if "\n" in section else "…"
            budget -= len(section)
            body.append(section)
        body.append(f'Full output: bash_output("{n.id}").')
        blocks.append(f"<background-job {attrs}>\n" + "\n".join(body) + "\n</background-job>")
    return "\n".join(blocks)


_NOTICE_RE = re.compile(
    r'^<background-job id="([^"]*)" status="(\w+)"(?: code="(-?\d+)")?(?: killed_by="(\w+)")? runtime="([^"]*)">$',
    re.MULTILINE)  # tail lines are indented, so a log line quoting the tag can't match


def notice_line(job_id: str, status: str, code: int | str | None, killed_by: str | None, runtime: str) -> str:
    """The TUI's system line for a job that ended: `⚙ job "test" exited (1) after 42s`."""
    how = f"killed by {killed_by}" if status == "killed" else f"exited ({code})" if code is not None else "exited"
    return f'⚙ job "{job_id}" {how} after {runtime}'


def parse_notice(content: object) -> list[str] | None:
    """The `notice_line`s of a stored `<background-job>` user step (D7), else None."""
    if not isinstance(content, str) or not content.startswith("<background-job ") \
            or not content.rstrip().endswith("</background-job>"):
        return None
    return [notice_line(m[1], m[2], m[3], m[4], m[5]) for m in _NOTICE_RE.finditer(content)] or None


def _log_watcher_error(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.warning("job watcher failed: %r", task.exception())


# --- startup housekeeping (D8, D11) -----------------------------------------------------------

def prune_logs(days: float = 7) -> list[Path]:
    """Remove session log directories untouched for `days`; files still in use are skipped."""
    root = jobs_root()
    cutoff = time.time() - days * 86400
    removed: list[Path] = []
    try:
        dirs = [d for d in root.iterdir() if d.is_dir()]
    except OSError:
        return removed
    for d in dirs:
        try:
            if d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed.append(d)
        except OSError:
            pass
    return removed


def find_orphans(session_id: str) -> list[tuple[str, int]]:
    """`(command, pid)` of jobs other sessions' `jobs.json` never marked ended (a hard crash).
    Reported only — without psutil a live job can't be told from a reused pid."""
    root = jobs_root()
    own = _job_dir(session_id)
    found: list[tuple[str, int]] = []
    try:
        indexes = sorted(root.glob("*/jobs.json"))
    except OSError:
        return found
    for path in indexes:
        if path.parent == own:
            continue
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
            found += [(str(e["command"]), int(e["pid"])) for e in entries if e.get("ended") is None]
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return found


_registry: JobRegistry | None = None


def get_registry() -> JobRegistry:
    global _registry
    if _registry is None:
        _registry = JobRegistry()
    return _registry
