"""Notices for an owner's next model call: background shell jobs and background agents that ended.

One queue per owner (a session id, or `<session>/sub/<run>` for a subagent's shell jobs), drained by the loop
before each model call and peeked by the TUI's idle wake-up (subagents spec D7, background-shell spec D7/D15).
"""
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

NoticeKind = Literal["job", "agent"]
# Automatic turns in a row without a message from the user: the TUI's idle wake-up (jobs and agents together,
# background-shell D15, subagents D7) and headless waiting for background agents (subagents D17)
WAKE_LIMIT = 3


@dataclass
class AgentNotice:
    """A background agent run that ended (completed, failed or stopped)."""
    id: str
    agent_type: str
    description: str
    status: str
    runtime: float  # seconds
    tokens: int
    report: str
    wake: bool = True  # ended on its own; False when the user or the model stopped it (no idle wake-up)
    kind: Literal["agent"] = "agent"


class NoticeQueue:
    def __init__(self) -> None:
        self._queues: dict[str, list[Any]] = {}

    def put(self, owner: str, notice: Any) -> None:
        self._queues.setdefault(owner, []).append(notice)

    def drain(self, owner: str, kind: NoticeKind | None = None) -> list[Any]:
        """Remove and return the owner's notices (of one kind, or all), oldest first."""
        if kind is None:
            return self._queues.pop(owner, [])
        queue = self._queues.get(owner, [])
        taken = [n for n in queue if n.kind == kind]
        queue[:] = [n for n in queue if n.kind != kind]
        return taken

    def pending(self, owner: str, kind: NoticeKind | None = None) -> list[Any]:
        """The owner's notices, left in place."""
        return [n for n in self._queues.get(owner, []) if kind is None or n.kind == kind]

    def discard(self, owner: str, match: Callable[[Any], bool]) -> None:
        queue = self._queues.get(owner)
        if queue:
            queue[:] = [n for n in queue if not match(n)]

    def transfer(self, old_owner: str, new_owner: str) -> None:
        """Move the old owner's notices to the new one (/clear, /resume)."""
        if old_owner in self._queues:
            self._queues.setdefault(new_owner, []).extend(self._queues.pop(old_owner))


_queue: NoticeQueue | None = None


def get_notice_queue() -> NoticeQueue:
    global _queue
    if _queue is None:
        _queue = NoticeQueue()
    return _queue


def fmt_agent_runtime(seconds: float) -> str:
    """`2m 10s` / `42s`."""
    s = int(seconds)
    return f"{s // 60}m {s % 60}s" if s >= 60 else f"{s}s"


def fmt_tokens(tokens: int) -> str:
    return f"{round(tokens / 1000)}k" if tokens >= 1000 else str(tokens)


def format_agent_notice(n: AgentNotice, max_chars: int | None = None) -> str:
    report = n.report.strip()
    if max_chars is not None and len(report) > max_chars:
        report = report[:max_chars] + "\n[… report truncated]"
    description = n.description.replace('"', "'")
    attrs = (f'id="{n.id}" type="{n.agent_type}" description="{description}" status="{n.status}" '
             f'runtime="{fmt_agent_runtime(n.runtime)}" tokens="{fmt_tokens(n.tokens)}"')
    return f"<background-agent {attrs}>\n{report}\n</background-agent>"


def format_notices(notices: Iterable[Any], max_chars: int | None = None) -> str:
    """The `<background-job>` / `<background-agent>` blocks the loop appends as one user step."""
    from dostuff.tools.bash.jobs import format_notice  # jobs imports this module
    blocks = [format_agent_notice(n, max_chars) if n.kind == "agent" else format_notice([n]) for n in notices]
    return "\n".join(blocks)



_ENDED_WORDS = {"completed": "finished", "failed": "failed", "stopped": "was stopped", "interrupted": "was interrupted"}


def agent_notice_line(description: str, agent_type: str, status: str, runtime: str) -> str:
    """The TUI's system line for a background agent that ended: `⚙ agent "find x" (explore) finished after 2m 10s`."""
    return f'⚙ agent "{description}" ({agent_type}) {_ENDED_WORDS.get(status, status)} after {runtime}'


_AGENT_OPEN_RE = re.compile(
    r'^<background-agent id="([^"]*)" type="([^"]*)" description="([^"]*)" status="(\w+)" runtime="([^"]*)" '
    r'tokens="[^"]*">$')
_CLOSING = ("</background-job>", "</background-agent>")


def parse_notices(content: object) -> list[str] | None:
    """The system lines of a stored notice step (`<background-job>` and `<background-agent>` blocks), else None."""
    from dostuff.tools.bash.jobs import _NOTICE_RE, notice_line
    if not isinstance(content, str) or not content.startswith(("<background-job ", "<background-agent "))             or not content.rstrip().endswith(_CLOSING):
        return None
    lines: list[str] = []
    in_report = False  # a report isn't indented, so a quoted tag inside it must not count
    for text in content.splitlines():
        if in_report:
            in_report = text != "</background-agent>"
            continue
        agent = _AGENT_OPEN_RE.match(text)
        if agent:
            lines.append(agent_notice_line(agent[3], agent[2], agent[4], agent[5]))
            in_report = True
            continue
        job = _NOTICE_RE.match(text)
        if job:
            lines.append(notice_line(job[1], job[2], job[3], job[4], job[5]))
    return lines or None
