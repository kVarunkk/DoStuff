"""Every subagent run, foreground or background, is an asyncio task owned by the registry (spec D1, D4)."""
import asyncio
import contextlib
import datetime
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from dostuff.agent.subagents import current_agent, read_only_shell
from dostuff.agent.subagents.adapter import SubagentAdapter
from dostuff.agent.subagents.definitions import UnknownAgentType, get_definitions, resolve_model
from dostuff.agent.subagents.fork import FORK_TYPE, MODEL_IGNORED, ForkContext, fork_first_step
from dostuff.agent.subagents.prompt import build_subagent_prompt, worktree_note
from dostuff.config import DEFAULT_AGENTS_MAX_CONCURRENT
from dostuff.agent.subagents.run import _close_worktree, resolve_subagent_tools, run_agent_loop
from dostuff.helpers.hooks.events import SUBAGENT_START_CONTEXT, subagent_start
from dostuff.helpers.agent.notices import AgentNotice, agent_notice_line, fmt_agent_runtime, get_notice_queue
from dostuff.helpers.tools.project_root import project_root, use_project_root
from dostuff.lib.model import current_model, is_main_provider
from dostuff.lib.worktree import Worktree, create_worktree, detect_worktree
from dostuff.tools.bash.jobs import get_registry

logger = logging.getLogger(__name__)

AgentEvent = Literal["status", "end"]  # every status change; then once when a run ends
AgentStatus = Literal["queued", "running", "waiting_approval", "completed", "failed", "stopped", "interrupted"]

DEFAULT_MAX_CONCURRENT = DEFAULT_AGENTS_MAX_CONCURRENT  # running + waiting_approval at once; more are queued (D4)
MOVED = "Agent \"{id}\" moved to the background; you'll get its report as a notice."
STOP_GRACE_S = 1.0  # after setting a run's cancel event, cancel its task if it hasn't ended (§Stop)
UNREAD = '[send_message] Agent "{id}" ended before reading: {messages}. Send it again to continue the agent.'
WORKTREE_GONE = ("[worktree] Your earlier worktree was removed because you changed no files. This run works in a "
                 "new worktree of HEAD at {path}; files you saw or made before are not there.")
_SLOT_STATUSES = ("running", "waiting_approval")
_ENDED_STATUSES = ("completed", "failed", "stopped", "interrupted")
ENDED_STATUSES = _ENDED_STATUSES


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
    worktree: Worktree | None = None  # kept after it is removed, so its row and /agents still say where it was
    isolation: str = ""  # "worktree": every run of this agent works in one (a new one if the last was removed)
    started: float = 0.0
    ended: float | None = None
    tokens: int = 0
    cost_usd: float = 0.0
    result: str | None = None  # the last run's final report
    inbox: list[str] = field(default_factory=list)  # send_message to a running agent (D10)
    cancel: asyncio.Event = field(default_factory=asyncio.Event)  # its own, not the main agent's (D5)
    task: asyncio.Task[Any] | None = None
    done: asyncio.Future[str] | None = None  # the run's report; made inside the loop
    reply: asyncio.Future[str] | None = None  # what a foreground tool call awaits: the report, or MOVED (D9)
    turn_id: str = ""  # the main turn that started it (ESC stops that turn's foreground runs)
    stopped_by: str | None = None  # "model" (stop_agent), "user" (/agents, ESC), "session_end"
    slot: "asyncio.Future[None] | None" = None  # resolved when a queued run gets a slot
    answer_now: bool = False  # /agents → Answer approval: show its held-back prompt even while the user types
    fork: ForkContext | None = None  # a fork's system prompt parts and tools, inherited from the main agent (D11)


@dataclass
class _RunContext:
    """What one run needs from its caller; not kept after the run."""
    prompt: str
    mcp_client: Any
    adapter: Any
    permissions: Any
    requested_tools: list[str] | None
    warnings: list[str]


class AgentTranscript:
    """The `store` a subagent's loop writes to: its steps go to `agent_steps` under its id, so the session's
    own history never sees them (D14). Writes go through the registry's ordered write chain."""

    def __init__(self, registry: "AgentRegistry", run: AgentRun) -> None:
        self._registry = registry
        self._run = run

    async def append(self, session_id: str, step: dict[str, Any]) -> None:
        agent_id = self._run.id
        await self._registry._write(lambda store: store.append_agent_step(agent_id, step))

    async def compact_working_history(self, session_id: str, keep_last: int, compaction_notes: str) -> None:
        self._run.compaction_notes = compaction_notes
        agent_id = self._run.id
        await self._registry._write(lambda store: store.compact_agent_steps(agent_id, keep_last, compaction_notes))


def _iso(ts: float | None) -> str | None:
    return datetime.datetime.fromtimestamp(ts).isoformat() if ts else None


def _ts(iso: str | None) -> float | None:
    try:
        return datetime.datetime.fromisoformat(iso).timestamp() if iso else None
    except ValueError:
        return None


def _stored_worktree(path: str | None, branch: str | None) -> Worktree | None:
    """The worktree a stored row names. One that is gone (removed, or deleted by hand) comes back as a record
    whose path doesn't exist, so the next run gets a new one (D10)."""
    if not path:
        return None
    wt_path = Path(path)
    if wt_path.exists():
        found = detect_worktree(wt_path)
        if found is not None:
            return found
    return Worktree(name=wt_path.name, path=wt_path, branch=branch or "", repo_root=project_root(), base="")


def agent_row(run: AgentRun) -> dict[str, Any]:
    """The `agent_runs` row for a run, as it is now."""
    wt = run.worktree
    return {
        "id": run.id, "session_id": run.owner, "agent_type": run.agent_type, "description": run.description,
        "model": run.model, "status": run.status, "background": run.background,
        "worktree": str(wt.path) if wt else None, "branch": wt.branch if wt else None,
        "started_at": _iso(run.started), "ended_at": _iso(run.ended), "tokens": run.tokens,
        "cost_usd": run.cost_usd, "result": run.result, "compaction_notes": run.compaction_notes,
    }


def _runtime(run: AgentRun) -> str:
    return fmt_agent_runtime((run.ended or time.time()) - (run.started or time.time()))


class AgentRegistry:
    def __init__(self, max_concurrent: int | None = None) -> None:
        self._runs: dict[str, AgentRun] = {}
        self._max_concurrent = max_concurrent  # None: agents.max_concurrent, read when it matters (D18)
        self._queue: deque[tuple[AgentRun, asyncio.Future[None]]] = deque()  # FIFO, waiting for a slot
        self._listeners: list[Callable[[AgentRun, AgentEvent], None]] = []
        self._store: Any = None  # the session store: rows and transcripts (D14); None keeps them in memory only
        self._writes: asyncio.Future[None] | None = None  # the last store write; each one waits for the one before

    @property
    def max_concurrent(self) -> int:
        """running + waiting_approval runs at once; more starts are queued (D4)."""
        if self._max_concurrent is not None:
            return self._max_concurrent
        from dostuff.config import get_config
        return get_config().agents_max_concurrent

    @max_concurrent.setter
    def max_concurrent(self, value: int) -> None:
        self._max_concurrent = value

    # --- the store (D14) -----------------------------------------------------------------------

    def attach_store(self, store: Any) -> None:
        self._store = store

    def _write(self, write: "Callable[[Any], Awaitable[None]]") -> "asyncio.Future[None]":
        """Queue a store write behind the earlier ones, so rows and steps land in order. A failed write is
        logged, never raised: a run must not fail because its transcript could not be saved."""
        store, prev = self._store, self._writes
        if store is None:
            done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            done.set_result(None)
            return done

        async def go() -> None:
            if prev is not None:
                with contextlib.suppress(BaseException):
                    await prev
            try:
                await write(store)
            except Exception as e:
                logger.warning("could not save agent data: %r", e)
        self._writes = asyncio.ensure_future(go())
        return self._writes

    def _save(self, run: AgentRun) -> None:
        """Upsert the run's row as it is now (snapshotted here, so each status change is written in order)."""
        if self._store is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        row = agent_row(run)
        self._write(lambda store: store.upsert_agent_run(row))

    async def flush(self) -> None:
        """Wait for every queued store write."""
        while self._writes is not None and not self._writes.done():
            await asyncio.shield(self._writes)

    # --- listeners (the TUI: notifications, wake-up, the status line) --------------------------

    def add_listener(self, fn: "Callable[[AgentRun, AgentEvent], None]") -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn: "Callable[[AgentRun, AgentEvent], None]") -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _notify(self, run: AgentRun, event: AgentEvent) -> None:
        if event == "status":  # status, owner or background flag changed: the stored row follows (D14)
            self._save(run)
        for fn in list(self._listeners):
            try:
                fn(run, event)
            except Exception as e:  # a broken listener must not break the run
                logger.warning("agent listener failed: %r", e)

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
        self._notify(run, "status")
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

    def _reserve(self, run: AgentRun) -> None:
        """Take a slot now, or join the FIFO queue (`run.slot` resolves when it gets one). Synchronous, so the
        caller knows at once whether the run is `running` or `queued`."""
        if not self._queue and self._slots_used() < self.max_concurrent:
            self._set_status(run, "running")
            return
        run.slot = asyncio.get_running_loop().create_future()
        self._queue.append((run, run.slot))

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
        fork: ForkContext | None = None,
    ) -> AgentRun:
        """Start a run as a task the registry owns; it outlives the caller. Raises UnknownAgentType, or
        ValueError for a fork without the main agent's `fork` context."""
        from dostuff.config import get_config
        config = get_config()
        if agent_type == FORK_TYPE:  # always the main model; `model=` is ignored with a warning (D11)
            if fork is None:
                raise ValueError("a fork needs the main agent's conversation")
            resolved = config.model_name
            model_warning = MODEL_IGNORED.format(model=model.strip(), main=resolved) if model.strip() else None
        else:
            definitions = get_definitions()
            if agent_type not in definitions:
                raise UnknownAgentType(agent_type, sorted([*definitions, FORK_TYPE]))
            resolved, model_warning = resolve_model(model, definitions[agent_type], config)
            if model_warning:
                model_warning = f"[warning] {model_warning}"
        run = AgentRun(
            id=self._new_id(), owner=owner, agent_type=agent_type, description=description,
            model=resolved, background=background, worktree=worktree, turn_id=turn_id,
            isolation="worktree" if worktree is not None else "", fork=fork,
            compaction_notes=fork.compaction_notes if fork is not None else "",
        )
        self._runs[run.id] = run
        self._launch(run, _RunContext(prompt, mcp_client, adapter, permissions, requested_tools,
                                      [model_warning] if model_warning else []))
        return run

    async def resume(
        self,
        run: AgentRun,
        message: str,
        *,
        background: bool = False,
        mcp_client: Any = None,
        adapter: Any = None,
        permissions: Any = None,
        turn_id: str = "",
        fork: ForkContext | None = None,
    ) -> AgentRun:
        """send_message to an agent that has ended: a new run on the same id, with its whole history plus
        `message`, and a fresh iteration cap (D10). Raises WorktreeError if it needs a new worktree and git fails."""
        assert run.status in _ENDED_STATUSES, run.status
        if run.agent_type == FORK_TYPE and run.fork is None:
            # Loaded after a restart: its system prompt and tools come from the main agent as it is now; its
            # history is its own (D11)
            if fork is None:
                raise ValueError("a fork needs the main agent's conversation")
            run.fork = ForkContext(fork.system_instruction, "", [], fork.tool_names, fork.session_hook_context)
        if not run.model:  # a row from before the model was stored: resolve it again
            from dostuff.config import get_config
            definitions = get_definitions()
            agent_def = definitions.get(run.agent_type) or definitions["general-purpose"]
            run.model = resolve_model("", agent_def, get_config())[0]
        prompt = message
        if run.isolation and (run.worktree is None or not run.worktree.path.exists()):
            run.worktree = await asyncio.to_thread(create_worktree, f"agent-{run.id}-{uuid.uuid4().hex[:4]}",
                                                   project_root())
            prompt = f"{WORKTREE_GONE.format(path=run.worktree.path)}\n\n{message}"
        run.background, run.turn_id, run.stopped_by = background, turn_id, None
        run.cancel = asyncio.Event()
        run.started, run.ended, run.slot = 0.0, None, None
        self._launch(run, _RunContext(prompt, mcp_client, adapter, permissions, None, []))
        return run

    async def load(self, session_id: str, store: Any) -> list[AgentRun]:
        """Session load (startup, /resume): the session's stored agents come back as records that
        `send_message` can continue. Nothing starts. A run stored as running, queued or waiting for approval
        belonged to a process that is gone, so it is `interrupted` (D14, OUT-006). Agents already here (carried
        over by /clear or /resume) are kept as they are. Returns the agents it added."""
        loaded: list[AgentRun] = []
        for row in await store.load_agent_runs(session_id):
            if row["id"] in self._runs:
                continue
            status: AgentStatus = row["status"] if row["status"] in _ENDED_STATUSES else "interrupted"
            started = _ts(row["started_at"]) or 0.0
            run = AgentRun(
                id=row["id"], owner=session_id, agent_type=row["agent_type"] or "general-purpose",
                description=row["description"] or "", model=row["model"] or "", status=status,
                background=row["background"], history=await store.load_agent_steps(row["id"], working=True),
                compaction_notes=row["compaction_notes"],
                worktree=await asyncio.to_thread(_stored_worktree, row["worktree"], row["branch"]),
                isolation="worktree" if row["worktree"] else "", started=started,
                ended=_ts(row["ended_at"]) or started or None, tokens=row["tokens"], cost_usd=row["cost_usd"],
                result=row["result"],
            )
            self._runs[run.id] = run
            if status != row["status"]:
                await store.upsert_agent_run(agent_row(run))
            loaded.append(run)
        return loaded

    def _launch(self, run: AgentRun, ctx: _RunContext) -> None:
        """Give the run a slot (or queue it) and start its task."""
        loop = asyncio.get_running_loop()
        run.done, run.reply = loop.create_future(), loop.create_future()
        self._reserve(run)
        run.task = asyncio.create_task(self._run(run, ctx), name=f"agent-{run.id}")
        run.task.add_done_callback(self._on_task_done)

    async def _run(self, run: AgentRun, ctx: _RunContext) -> None:
        status: AgentStatus = "failed"
        report = ""
        try:
            if run.slot is not None:
                await run.slot
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
        fork = run.fork if run.agent_type == FORK_TYPE else None
        if fork is not None:
            requested: list[str] | None = fork.fork_tools()  # the main agent's tools minus D12 (D11)
        else:
            requested = ctx.requested_tools if ctx.requested_tools is not None else (
                list(agent_def.tools) if agent_def.tools is not None else None)
        tool_names, dropped = resolve_subagent_tools(requested)
        current_model.set(run.model)  # this task's context only: the whole run uses its model (D15)
        read_only_shell.set(agent_def.read_only_shell)  # (D3)
        current_agent.set(run)
        missing = await _missing_key(run.model)
        if missing:
            return missing, "failed"
        transcript = AgentTranscript(self, run) if self._store is not None else None
        if fork is not None:
            # The main agent's system prompt, byte-identical, so its cached prefix is reused (D11)
            system = fork.system_instruction
            if run.history:  # a follow-up: its own history goes on (D10)
                first_steps = [{"role": "user", "content": ctx.prompt}]
            else:  # its first run: the main history as of the call, then the directive
                note = worktree_note(run.worktree) if run.worktree is not None else ""
                first_steps = [*fork.history, fork_first_step(ctx.prompt, note)]
                fork.history = []  # it lives in run.history now
        else:
            system = await build_subagent_prompt(agent_def, run.worktree)
            first_steps = [{"role": "user", "content": ctx.prompt}]
        adapter = SubagentAdapter(ctx.adapter, run, self._set_status)
        with use_project_root(run.worktree.path) if run.worktree else contextlib.nullcontext():
            hook_context = await subagent_start(
                session_id=run.owner, turn_id=run.turn_id, agent_id=run.id, agent_type=run.agent_type,
                background=run.background, prompt=ctx.prompt, adapter=adapter)
        if hook_context:  # in the run's new user step, so a fork's system prompt stays byte-identical (D11, D16)
            last = first_steps[-1]
            extra = SUBAGENT_START_CONTEXT.format(context=hook_context)
            first_steps[-1] = {**last, "content": f"{last['content']}\n\n{extra}"}
        for step in first_steps:
            run.history.append(step)
            if transcript is not None:
                await transcript.append(run.owner, step)
        owner = f"{run.owner}/sub/{run.id}"  # this run's shell jobs (background shell spec D3)
        jobs = get_registry()
        jobs.set_role(owner, run.description)
        report = ""
        status: AgentStatus = "failed"
        try:
            with use_project_root(run.worktree.path) if run.worktree else contextlib.nullcontext():
                report, status = await run_agent_loop(
                    ctx.prompt, ctx.mcp_client, run.owner, run.turn_id, adapter,
                    ctx.permissions, system, run.history, tool_names, owner,
                    agent_def.max_iterations or get_config().agents_max_iterations,
                    store=transcript, compaction_notes=run.compaction_notes, inbox=run.inbox,
                    session_hook_context=fork.session_hook_context if fork is not None else None,
                )
        finally:
            # Its jobs die with it: nobody is left to read them or hear when they end (background shell D8)
            await jobs.kill_owner(owner, "owner_ended")
            if run.worktree is not None:
                note = await asyncio.to_thread(_close_worktree, run.worktree)  # removed: the next run gets a new one
                report = f"{report}\n\n{note}" if report else note
        if dropped:
            report = f"[warning] Tools not granted to the subagent (restricted or unknown): {', '.join(dropped)}\n\n{report}"
        if ctx.warnings:
            report = "\n".join(ctx.warnings) + f"\n\n{report}"
        return report, status

    def _finish(self, run: AgentRun, status: AgentStatus, report: str, ctx: _RunContext) -> None:
        if run.inbox:  # sent after its last model call, or it was stopped first: say so rather than drop them
            unread = "; ".join(f'"{m}"' for m in run.inbox)
            run.inbox.clear()
            report = f"{report}\n\n{UNREAD.format(id=run.id, messages=unread)}"
        run.ended = time.time()
        run.result = report
        self._set_status(run, status)
        for future in (run.done, run.reply):
            if future is not None and not future.done():
                future.set_result(report)
        if run.background:
            _emit(ctx.adapter, agent_notice_line(run.description, run.agent_type, status, _runtime(run)))
            if run.stopped_by != "session_end":  # nobody is left to read it (§Stop)
                get_notice_queue().put(run.owner, AgentNotice(
                    id=run.id, agent_type=run.agent_type, description=run.description, status=status,
                    runtime=run.ended - (run.started or run.ended), tokens=run.tokens, report=report,
                    wake=run.stopped_by is None,  # stopped by the user or the model: no wake-up (D7)
                ))
        self._notify(run, "end")

    def _on_task_done(self, task: "asyncio.Task[Any]") -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error("agent task %s ended with an unhandled error", task.get_name(), exc_info=task.exception())

    # --- stopping ------------------------------------------------------------------------------

    def detach_foreground(self, owner: str) -> list[AgentRun]:
        """Ctrl+B: the owner's foreground runs (running or queued) become background runs. Their waiting tool
        calls return MOVED; the run keeps going, ESC no longer reaches it, and its end queues a notice (D9)."""
        moved = [r for r in self._runs.values()
                 if r.owner == owner and not r.background and r.status not in _ENDED_STATUSES]
        for run in moved:
            run.background = True
            if run.reply is not None and not run.reply.done():
                run.reply.set_result(MOVED.format(id=run.id))
            self._notify(run, "status")
        return moved

    def transfer(self, old_owner: str, new_owner: str) -> list[AgentRun]:
        """/clear and /resume: the old session's runs that haven't ended belong to the new one, so their end
        notice and wake-up go there and `stop_agent` reaches them. Ended runs stay with the old session.
        (The notice queue is moved by `JobRegistry.transfer`, which shares it.)"""
        moved = [r for r in self._runs.values() if r.owner == old_owner and r.status not in _ENDED_STATUSES]
        for run in moved:
            run.owner = new_owner
            self._notify(run, "status")
        return moved

    def cancel_turn(self, turn_id: str) -> None:
        """ESC in the main turn: set the cancel event of that turn's foreground runs (D5)."""
        for run in self._runs.values():
            if run.turn_id == turn_id and not run.background and run.status not in _ENDED_STATUSES:
                run.cancel.set()

    async def stop(self, run: AgentRun, by: str) -> None:
        """Set the run's cancel event, then cancel its task if it hasn't ended after STOP_GRACE_S (§Stop)."""
        if run.status in _ENDED_STATUSES or run.task is None:
            return
        run.stopped_by = by
        run.cancel.set()
        if run.status == "queued" and run.slot is not None:
            run.slot.cancel()  # it never got a slot, so it never called the model; it ends through _run
        with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(run.task), STOP_GRACE_S)
        if not run.task.done():
            run.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await run.task

    async def stop_all(self, by: str) -> None:
        await asyncio.gather(*(self.stop(run, by) for run in list(self._runs.values())))
        await self.flush()


def owner_agents_error(agent_id: str, owner: str) -> str:
    """The error for an id that isn't one of the owner's agents, listing the ones that are (D10)."""
    runs = get_agent_registry().runs(owner)
    listing = (", ".join(f"{r.id} ({r.description}, {r.agent_type}, {r.status})" for r in runs)
               if runs else "")
    return (f'[error] No agent "{agent_id}" in this session. '
            + (f"Its agents: {listing}." if listing else "It has no agents."))


async def _missing_key(model: str) -> str | None:
    """An error for a run whose model is on another provider whose API key isn't set (D15), else None.
    The main provider's key was checked at startup, or comes from `api_key_env`."""
    if is_main_provider():
        return None
    from dostuff.setup_wizard import missing_api_keys
    missing = await asyncio.to_thread(missing_api_keys, model)
    if not missing:
        return None
    return f"[error] The subagent's model {model} needs {', '.join(missing)}, which is not set."


def _emit(adapter: Any, text: str) -> None:
    if adapter is not None and hasattr(adapter, "emit"):
        adapter.emit("system", text)


_agent_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry
