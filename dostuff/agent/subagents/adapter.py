"""The adapter a subagent run talks to: its own cancel state, approvals forwarded to the parent (spec D5)."""
import contextlib
from typing import TYPE_CHECKING, Any, Callable, Iterator

from dostuff.helpers.agent.loop_check_cancelled import is_cancelled

if TYPE_CHECKING:
    from dostuff.agent.subagents.registry import AgentRun


class RunCancel:
    """A run's cancel state as the loop sees it: its own event, plus the parent's ESC while it is in the foreground.

    `clear()` does nothing: a cancelled run is over, and the registry reads the state afterwards.
    """

    def __init__(self, run: "AgentRun", parent: Any):
        self._run = run
        self._parent = parent

    def is_set(self) -> bool:
        return self._run.cancel.is_set() or (not self._run.background and is_cancelled(self._parent))

    def set(self) -> None:
        self._run.cancel.set()

    def clear(self) -> None:
        pass


class SubagentAdapter:
    """Wraps the parent adapter for one run.

    Foreground: output goes to the parent prefixed `[agent: <description>]`. Background: only errors and usage do;
    the registry reports its start and end. Usage is also added to the run's own tokens and cost. Approval prompts go
    to the parent titled `[agent: <description> · <type>]`, one at a time behind the parent's own lock, and the
    run is `waiting_approval` meanwhile (D8).
    """

    def __init__(self, parent: Any, run: "AgentRun", set_status: "Callable[[AgentRun, Any], None] | None" = None):
        self._parent = parent
        self._run = run
        self._prefix = f"[agent: {run.description}]"
        self._ask_prefix = f"[agent: {run.description} · {run.agent_type}]"
        self._cancel_event = RunCancel(run, parent)
        self._set_status = set_status or (lambda r, status: setattr(r, "status", status))
        self._asking = 0

    @contextlib.contextmanager
    def _waiting(self) -> Iterator[None]:
        self._asking += 1
        self._set_status(self._run, "waiting_approval")
        try:
            yield
        finally:
            self._asking -= 1
            if not self._asking and self._run.status == "waiting_approval":
                self._set_status(self._run, "running")

    @property
    def _cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @_cancelled.setter
    def _cancelled(self, value: bool) -> None:
        if value:
            self._run.cancel.set()

    def emit(self, event_type: str, data: Any) -> None:
        if event_type == "status":  # context-window % of the subagent would overwrite the parent's
            return
        if event_type == "usage" and isinstance(data, dict):
            self._run.tokens += int(data.get("total_tokens") or 0)
            self._run.cost_usd += float(data.get("cost") or 0.0)
        if self._parent is None or not hasattr(self._parent, "emit"):
            return
        if self._run.background and event_type not in ("usage", "error"):
            return
        if event_type == "hook" and isinstance(data, dict):  # hook_line renders the prefix
            self._parent.emit(event_type, {**data, "subagent": self._run.description})
        elif event_type == "usage" or not isinstance(data, str):
            self._parent.emit(event_type, data)
        else:
            self._parent.emit(event_type, f"{self._prefix} {data}")

    async def ask(self, message: str, **kwargs: Any) -> Any:
        if self._parent is None or not hasattr(self._parent, "ask"):
            return "no"
        if self._run.background:
            kwargs["background"] = True  # the TUI holds it back while the user types (D8)
        with self._waiting():
            return await self._parent.ask(f"{self._ask_prefix} {message}", **kwargs)

    async def ask_batch(self, messages: list[str], **kwargs: Any) -> Any:
        if not hasattr(self._parent, "ask_batch"):
            return "each"
        if self._run.background:
            kwargs["background"] = True
        with self._waiting():
            return await self._parent.ask_batch([f"{self._ask_prefix} {m}" for m in messages], **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parent, name)
