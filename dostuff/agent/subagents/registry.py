"""Every subagent run, foreground or background, is an asyncio task owned by the registry (spec D1, D4)."""
import asyncio
import contextlib
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from dostuff.agent.subagents.adapter import SubagentAdapter
from dostuff.agent.subagents.definitions import UnknownAgentType, get_definitions
from dostuff.agent.subagents.prompt import build_subagent_prompt
from dostuff.agent.subagents.run import _close_worktree, resolve_subagent_tools, run_agent_loop
from dostuff.helpers.tools.project_root import use_project_root
from dostuff.lib.model import get_litellm_model
from dostuff.lib.worktree import Worktree
from dostuff.tools.bash.jobs import get_registry

logger = logging.getLogger(__name__)

AgentStatus = Literal["queued", "running", "waiting_approval", "completed", "failed", "stopped", "interrupted"]

DEFAULT_MAX_CONCURRENT = 4  # running + waiting_approval at once; more starts are queued (D4)
STOP_GRACE_S = 1.0  # after setting a run's cancel event, cancel its task if it hasn't ended (§Stop)
_SLOT_STATUSES = ("running", "waiting_approval")
_ENDED_STATUSES = ("completed", "failed", "stopped", "interrupted")


@dataclass
class AgentRun:
    id: str  # "a3f2c1", what the model and the user see
    owner: str  # session id
    agent_type: str  # "explore", "code-reviewer", "fork", …
    description: str  # 3–5 words from the call, shown in the UI
    model: str  # resolved litellm model string
    status: AgentStatus = "queued"
    background: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)  # kept across runs
    compaction_notes: str = ""
    worktree: Worktree | None = None
    started: float = 0.0
    ended: float | None = None
    tokens: int = 0
    cost_usd: float = 0.0
    result: str | None = None  # the last run's final report
    inbox: list[str] = field(default_factory=list)  # send_message to a running agent (D10)
    cancel: asyncio.Event = field(default_factory=asyncio.Event)  # its own, not the main agent's (D5)
    task: asyncio.Task[Any] | None = None
    done: asyncio.Future[str] | None = None  # what a foreground tool call awaits; made inside the loop
    turn_id: str = ""  # the main turn that started it (ESC stops that turn's foreground runs)


@dataclass
class _RunContext:
    """What one run needs from its caller; not kept after the run."""
    prompt: str
    mcp_client: Any
    adapter: Any
    permissions: Any
    requested_tools: list[str] | None


def _runtime(run: AgentRun) -> str:
    seconds = int((run.ended or time.time()) - (run.started or time.time()))
    return f"{seconds // 60}m {seconds % 60}s" if seconds >= 60 else f"{seconds}s"


class AgentRegistry:
    def __init__(self, max_concurrent: int = DEFAULT_MAX_CONCURRENT) -> None:
        self._runs: dict[str, AgentRun] = {}
        self.max_concurrent = max_concurrent
        self._queue: deque[tuple[AgentRun, asyncio.Future[None]]] = deque()  # FIFO, waiting for a slot

    def runs(self, owner: str | None = None) -> list[AgentRun]:
        return [r for r in self._runs.values() if owner is None or r.owner == owner]

    def get(self, agent_id: str) -> AgentRun | None:
        return self._runs.get(agent_id)

    def _new_id(self) -> str:
        while True:
            agent_id = uuid.uuid4().hex[:6]
            if agent_id not in self._runs:
                return agent_id

    # --- status and slots ----------------------------------------------------------------------

    def _set_status(self, run: AgentRun, status: AgentStatus) -> None:
        """The one place a run's status changes."""
        run.status = status
        if status in _ENDED_STATUSES:
            self._pump()

    def _slots_used(self) -> int:
        return sum(1 for r in self._runs.values() if r.status in _SLOT_STATUSES)

    def _pump(self) -> None:
        """Start queued runs, oldest first, while there are free slots."""
        while self._queue and self._slots_used() < self.max_concurrent:
            run, ready = self._queue.popleft()
            if ready.done():
                continue
            self._set_status(run, "running")
            ready.set_result(None)

    async def _acquire(self, run: AgentRun) -> None:
        if not self._queue and self._slots_used() < self.max_concurrent:
            self._set_status(run, "running")
            return
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._queue.append((run, ready))
        await ready

    # --- starting and running ------------------------------------------------------------------

    async def start(
        self,
        *,
        owner: str,
        agent_type: str,
        description: str,
        prompt: str,
        background: bool = False,
        model: str = "",
        mcp_client: Any = None,
        adapter: Any = None,
        permissions: Any = None,
        turn_id: str = "",
        worktree: Worktree | None = None,
        requested_tools: list[str] | None = None,
    ) -> AgentRun:
        """Start a run as a task the registry owns; it outlives the caller. Raises UnknownAgentType."""
        definitions = get_definitions()
        if agent_type not in definitions:
            raise UnknownAgentType(agent_type, sorted(definitions))
        run = AgentRun(
            id=self._new_id(), owner=owner, agent_type=agent_type, description=description,
            model=model or get_litellm_model(), background=background, worktree=worktree, turn_id=turn_id,
        )
        run.done = asyncio.get_running_loop().create_future()
        self._runs[run.id] = run
        ctx = _RunContext(prompt, mcp_client, adapter, permissions, requested_tools)
        run.task = asyncio.create_task(self._run(run, ctx), name=f"agent-{run.id}")
        run.task.add_done_callback(self._on_task_done)
        return run

    async def _run(self, run: AgentRun, ctx: _RunContext) -> None:
        status: AgentStatus = "failed"
        report = ""
        try:
            await self._acquire(run)
            run.started = time.time()
            if run.background:
                _emit(ctx.adapter, f'⚙ agent "{run.description}" ({run.agent_type}) started in the background')
            report, status = await self._execute(run, ctx)
        except asyncio.CancelledError:
            status, report = "stopped", report or f'[agent "{run.id}" was stopped]'
        except Exception as e:
            logger.error("agent %s (%s) failed", run.id, run.agent_type, exc_info=True)
            status, report = "failed", f"[error] {type(e).__name__}: {e}"
        finally:
            self._finish(run, status, report, ctx)

    async def _execute(self, run: AgentRun, ctx: _RunContext) -> tuple[str, AgentStatus]:
        from dostuff.config import get_config
        agent_def = get_definitions().get(run.agent_type) or get_definitions()["general-purpose"]
        requested = ctx.requested_tools if ctx.requested_tools is not None else (
            list(agent_def.tools) if agent_def.tools is not None else None)
        tool_names, dropped = resolve_subagent_tools(requested)
        system = await build_subagent_prompt(agent_def, run.worktree)
        run.history.append({"role": "user", "content": ctx.prompt})
        owner = f"{run.owner}/sub/{run.id}"  # this run's shell jobs (background shell spec D3)
        jobs = get_registry()
        jobs.set_role(owner, run.description)
        report = ""
        status: AgentStatus = "failed"
        try:
            with use_project_root(run.worktree.path) if run.worktree else contextlib.nullcontext():
                report, status = await run_agent_loop(
                    ctx.prompt, ctx.mcp_client, run.owner, run.turn_id, SubagentAdapter(ctx.adapter, run),
                    ctx.permissions, system, run.history, tool_names, owner,
                    agent_def.max_iterations or get_config().subagent_max_iterations,
                )
        finally:
            # Its jobs die with it: nobody is left to read them or hear when they end (background shell D8)
            await jobs.kill_owner(owner, "owner_ended")
            if run.worktree is not None:
                note = await asyncio.to_thread(_close_worktree, run.worktree)
                if not run.worktree.path.exists():
                    run.worktree = None  # removed: a later run gets a fresh one (D10)
                report = f"{report}\n\n{note}" if report else note
        if dropped:
            report = f"[warning] Tools not granted to the subagent (restricted or unknown): {', '.join(dropped)}\n\n{report}"
        return report, status

    def _finish(self, run: AgentRun, status: AgentStatus, report: str, ctx: _RunContext) -> None:
        run.ended = time.time()
        run.result = report
        self._set_status(run, status)
        if run.done is not None and not run.done.done():
            run.done.set_result(report)
        if run.background:
            _emit(ctx.adapter, f'⚙ agent "{run.description}" ({run.agent_type}) {_ENDED_WORDS[status]} after {_runtime(run)}')

    def _on_task_done(self, task: "asyncio.Task[Any]") -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error("agent task %s ended with an unhandled error", task.get_name(), exc_info=task.exception())

    # --- stopping ------------------------------------------------------------------------------

    def cancel_turn(self, turn_id: str) -> None:
        """ESC in the main turn: set the cancel event of that turn's foreground runs (D5)."""
        for run in self._runs.values():
            if run.turn_id == turn_id and not run.background and run.status not in _ENDED_STATUSES:
                run.cancel.set()

    async def stop(self, run: AgentRun, by: str) -> None:
        """Set the run's cancel event, then cancel its task if it hasn't ended after STOP_GRACE_S (§Stop)."""
        if run.status in _ENDED_STATUSES or run.task is None:
            return
        run.cancel.set()
        if run.status == "queued":
            run.task.cancel()  # it never got a slot, so it never called the model
        with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(run.task), STOP_GRACE_S)
        if not run.task.done():
            run.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await run.task

    async def stop_all(self, by: str) -> None:
        await asyncio.gather(*(self.stop(run, by) for run in list(self._runs.values())))


_ENDED_WORDS: dict[str, str] = {
    "completed": "finished", "failed": "failed", "stopped": "was stopped", "interrupted": "was interrupted",
}


def _emit(adapter: Any, text: str) -> None:
    if adapter is not None and hasattr(adapter, "emit"):
        adapter.emit("system", text)


_agent_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry
