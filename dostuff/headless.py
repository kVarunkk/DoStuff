"""`dostuff -p "prompt"`: run one turn without the TUI, for scripts and CI.

The final answer goes to stdout (or a JSON object with --output-format json, or one JSON event
per line with stream-json); progress (tool calls, results, errors) goes to stderr unless --quiet. Approvals are asked on the terminal when there is one;
without a terminal, calls that need approval are denied unless the permission mode allows them.
"""
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

from dostuff.agent.permissions import ApprovalReply, PermissionMode, SessionPermissions, parse_answer
from dostuff.helpers.hooks import trust
from dostuff.helpers.hooks.dispatch import hook_line, untrusted_notice_given
from dostuff.helpers.hooks.events import blocked_prompt_message, session_start, user_prompt_submit
from dostuff.helpers.ui.ask_answers import NoUserAvailable, one_line, pick_options

PREVIEW_LINES = 4
NO_ANSWER = "(no answer)"  # D13: an empty line is still an answer, and must not vanish
MCP_CONNECT_TIMEOUT_S = 150.0
OutputFormat = Literal["text", "json", "stream-json"]
OUTPUT_FORMATS: tuple[str, ...] = ("text", "json", "stream-json")
MAX_ITERATIONS_PREFIX = "Reached maximum iterations"  # loop()'s answer when it hits the cap
AGENT_EVENT_KEYS = ("id", "agent_type", "description", "status", "background")  # stream-json `agent` events
WAKE_LIMIT_MESSAGE = "⚙ Not waiting for more agents: {limit} automatic turns in a row."


def _err(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _parse_ask_line(answer: str, options: list[dict]) -> str:
    """One line of stdin as the answer to one question (D13): `2` / `1,3` pick options in option order,
    anything else is the answer as typed, and an empty line is `(no answer)`."""
    picked = pick_options(answer, options)
    if picked:
        return picked
    return one_line(answer) or NO_ANSWER


class _Refused(Exception):
    """The run stops before the model is called (exit code 2): a UserPromptSubmit hook rejected the prompt,
    or an untrusted fail_closed project hook can't be asked about. Nothing is stored."""


class _NotAttached(Exception):
    """An @mentioned image can't be sent (the model can't read images, too large, too many): exit code 1, before
    hooks and before anything is stored or sent (multimodal spec REQ-2.3)."""


class HeadlessAdapter:
    """Prints loop() events to stderr and asks for approval on the terminal (if any)."""

    def __init__(self, interactive: bool, quiet: bool, stream: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._interactive = interactive
        self._quiet = quiet
        self._stream = stream  # stream-json: gets each event as a dict
        self._cancelled = False
        self._cancel_event = asyncio.Event()
        self._pending_text = ""  # streamed text; shown on stderr if a tool call follows it
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.cost_usd: float | None = None  # None while no call had a known price
        self.denied_without_terminal = 0
        self._ask_lock = asyncio.Lock()  # parallel subagents ask one at a time (subagents spec D6)

    def _progress(self, text: str) -> None:
        if not self._quiet:
            _err(text)

    def _stream_event(self, event_type: str, data: Any) -> None:
        if self._stream is None:
            return
        if event_type == "tool_call":
            if self._pending_text.strip():
                self._stream({"type": "assistant", "text": self._pending_text.strip()})
            self._stream({"type": "tool_call", "call": str(data)})
        elif event_type == "tool_result":
            name, _, result = str(data).partition(": ")
            self._stream({"type": "tool_result", "tool": name, "result": result})
        elif event_type == "usage" and isinstance(data, dict):
            self._stream({"type": "usage", **{k: data.get(k) for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "cost")}})
        elif event_type in ("error", "system"):
            self._stream({"type": event_type, "message": str(data)})
        elif event_type == "hook" and isinstance(data, dict):
            self._stream({"type": "hook", **data})
        elif event_type == "job" and isinstance(data, dict):
            self._stream({"type": "job", **{k: data.get(k) for k in ("id", "status", "code")}})
        elif event_type == "agent" and isinstance(data, dict):
            self._stream({"type": "agent", **{k: data.get(k) for k in AGENT_EVENT_KEYS}})

    def emit(self, event_type: str, data: Any) -> None:
        self._stream_event(event_type, data)
        if event_type == "usage" and isinstance(data, dict):
            for k in self.usage:
                self.usage[k] += data.get(k) or 0
            if data.get("cost"):
                self.cost_usd = (self.cost_usd or 0.0) + data["cost"]
        elif event_type == "partial":
            self._pending_text = str(data)
        elif event_type == "tool_call":
            if self._pending_text.strip():
                self._progress(self._pending_text.strip())
            self._pending_text = ""
            self._progress(f"● {data}")
        elif event_type == "tool_result":
            lines = str(data).rstrip("\n").splitlines() or [""]
            extra = f"\n     … +{len(lines) - PREVIEW_LINES} lines" if len(lines) > PREVIEW_LINES else ""
            self._progress("  ⎿  " + "\n     ".join(lines[:PREVIEW_LINES]) + extra)
        elif event_type == "error":
            _err(str(data))  # errors show even with --quiet
        elif event_type in ("diff", "confirm"):
            self._progress(str(data))
        elif event_type == "hook" and isinstance(data, dict):
            self._progress(hook_line(data))

    async def ask(self, message: str, always_label: str = "", background: bool = False) -> ApprovalReply:
        # No terminal → deny. Project hook trust follows the same rule (trust.headless_decision)
        if not self._interactive:
            self.denied_without_terminal += 1
            self._progress(f"✗ Denied (no terminal to ask): {message.splitlines()[0] if message else ''}")
            return ApprovalReply("no", feedback="Running non-interactively (dostuff -p); nobody can approve this call.")
        async with self._ask_lock:
            _err(f"⚠ {message}")
            choices = "[y]es / [a]lways" if always_label else "[y]es"
            answer = await asyncio.to_thread(input, f"Allow? {choices} / [N]o: ")
        reply = parse_answer(answer)
        return ApprovalReply("yes" if reply == "always" and not always_label else reply)

    async def ask_user(self, questions: list[dict]) -> list[tuple[str, str]] | None:
        """The questions on stderr, one line per answer on stdin (D13); None when stdin closes first.

        A number (or `1,3`) picks options, anything else is free text, empty is `(no answer)`. Each line is
        the submission for its question: a terminal reading one line has no way back, so there is no submit
        row here. Ctrl+C lands in the main thread and ends the whole run, as for approvals; an EOF reaches
        the worker and cancels the call.
        """
        if not self._interactive:
            raise NoUserAvailable()
        pairs: list[tuple[str, str]] = []
        async with self._ask_lock:
            for question in questions:
                text = one_line(question.get("question", ""))
                options = question.get("options") or []
                if not self._quiet:
                    _err(f"⚠ {text}")
                    for i, option in enumerate(options, 1):
                        label = one_line(option.get("label", ""))
                        description = one_line(option.get("description", ""))
                        _err(f"  {i}. {label}" + (f"  {description}" if description else ""))
                prompt = f"Answer ({'1-%d or text' % len(options) if options else 'text'}): "
                try:
                    line = await asyncio.to_thread(input, prompt)
                except EOFError:
                    return None  # stdin closed (Ctrl+Z/Ctrl+D): cancelled, like Esc in the TUI
                pairs.append((text, _parse_ask_line(line, options)))
        return pairs


async def _connect_mcp(mcp_client: Any, servers: dict[str, Any], adapter: HeadlessAdapter) -> None:
    async def one(name: str, cfg: dict[str, Any]) -> None:
        timeout = float(cfg.get("connect_timeout") or MCP_CONNECT_TIMEOUT_S)
        try:
            await asyncio.wait_for(
                mcp_client.connect_to_server(
                    server_name=name, transport=cfg.get("transport", "stdio"), command=cfg.get("command"),
                    args=cfg.get("args", []), env=cfg.get("env"), url=cfg.get("url"), headers=cfg.get("headers"),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            adapter.emit("error", f"MCP {name}: connection timed out after {timeout:.0f}s")
        except Exception as e:
            adapter.emit("error", f"MCP {name}: {e}")
    await asyncio.gather(*(one(name, cfg) for name, cfg in servers.items()))


async def _stop_jobs(adapter: HeadlessAdapter) -> None:
    """Kill every job still running; report each (stderr line, or a stream-json `job` event)."""
    from dostuff.tools.bash.jobs import get_registry
    try:
        killed = await get_registry().kill_all()
    except Exception as e:
        _err(f"Could not stop background jobs: {e}")
        return
    for job in killed:
        if adapter._stream is not None:
            adapter.emit("job", {"id": job.id, "status": job.status, "code": job.exit_code})
        else:
            adapter._progress(f'⚙ stopped background job "{job.id}" (pid {job.pid}) at the end of the run')


def _agent_events(adapter: HeadlessAdapter, session_id: str) -> Callable[[Any, str], None]:
    """A registry listener: an `agent` event when one of this session's runs starts and when it ends
    (stream-json), and the start / end lines of background runs on stderr (subagents spec D17)."""
    from dostuff.helpers.agent.notices import agent_notice_line, fmt_agent_runtime
    started: set[str] = set()

    def on_event(run: Any, event: str) -> None:
        if run.owner != session_id:
            return
        if event == "status" and run.status == "running" and run.id not in started:
            started.add(run.id)
            line = f'⚙ agent "{run.description}" ({run.agent_type}) started in the background'
        elif event == "end":
            started.discard(run.id)  # a send_message resume is a new run on the same id
            runtime = fmt_agent_runtime((run.ended or time.time()) - (run.started or time.time()))
            line = agent_notice_line(run.description, run.agent_type, run.status, runtime)
        else:
            return
        if adapter._stream is not None:
            adapter.emit("agent", {"id": run.id, "agent_type": run.agent_type, "description": run.description,
                                   "status": run.status, "background": run.background})
        elif run.background:
            adapter._progress(line)
    return on_event


async def _agent_notices(session_id: str) -> bool:
    """Wait while this session has agents running and no agent notice is queued; True once one is
    (the next turn drains it), False when no agent is left to wait for (subagents spec D17)."""
    from dostuff.agent.subagents.registry import ENDED_STATUSES, get_agent_registry
    from dostuff.helpers.agent.notices import get_notice_queue
    while True:
        if get_notice_queue().pending(session_id, "agent"):
            return True
        waiting = [r.done for r in get_agent_registry().runs(session_id)
                   if r.status not in ENDED_STATUSES and r.done is not None and not r.done.done()]
        if not waiting:
            return False
        await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)


def _agents_pending(session_id: str) -> bool:
    """Any of this session's agents still running, or an agent notice not yet read."""
    from dostuff.agent.subagents.registry import ENDED_STATUSES, get_agent_registry
    from dostuff.helpers.agent.notices import get_notice_queue
    return bool(get_notice_queue().pending(session_id, "agent")) or any(
        r.status not in ENDED_STATUSES for r in get_agent_registry().runs(session_id))


async def _stop_agents(adapter: HeadlessAdapter) -> None:
    """Stop every agent still running; each is named on stderr (stream-json: its `agent` end event)."""
    from dostuff.agent.subagents.registry import ENDED_STATUSES, get_agent_registry
    registry = get_agent_registry()
    running = [r for r in registry.runs() if r.status not in ENDED_STATUSES]
    if not running:
        return
    try:
        await registry.stop_all("session_end")
    except Exception as e:
        _err(f"Could not stop agents: {e}")
        return
    if adapter._stream is None:
        for run in running:
            adapter._progress(f'⚙ stopped agent "{run.description}" ({run.id}, {run.agent_type}) at the end of the run')


async def run_headless(
    prompt: str,
    session_id: str | None = None,
    permission_mode: PermissionMode | None = None,
    output_format: OutputFormat = "text",
    quiet: bool = False,
    interactive: bool = True,
    max_turns: int | None = None,
    allowed_tools: list[str] | None = None,
    trust_project_hooks: bool = False,
) -> int:
    """Run one turn and print the answer; returns the process exit code.

    max_turns caps the model calls (loop.max_iterations otherwise); allowed_tools are patterns
    that run without asking in any mode (see SessionPermissions.allow). trust_project_hooks runs the
    project's hooks without a trust record (hooks spec D4).
    """
    # Heavy imports here so `dostuff --help` stays fast
    from dostuff.agent.instructions import build_base_instructions, build_system_instructions, build_turn_context
    from dostuff.agent.loop import loop as agent_loop
    from dostuff.config import get_config
    from dostuff.helpers.agent.append_step import append_step
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    from dostuff.helpers.agent.turn_context import make_context_step
    from dostuff.helpers.mcp.load_mcp_config import load_mcp_config
    from dostuff.helpers.skills.custom_commands import discover_commands
    from dostuff.helpers.skills.skill_command import (
        mention_line, parse_skill_command, skill_commands, skill_mentions, skill_prompt, with_mention_note,
    )
    from dostuff.helpers.ui.emit import set_adapter
    from dostuff.lib.mcp.mcp_client import MCPClient
    from dostuff.lib.mcp.mcp_client_registration_store import MCPClientRegistrationStore
    from dostuff.lib.memory.session_store import SQLiteSessionStore
    from dostuff.lib.memory.stores import get_episodic_store, get_semantic_store, make_tool_registry_store
    from dostuff.lib.tracing import session_id_var, turn_id_var

    config = get_config()

    def stream(event: dict[str, Any]) -> None:
        print(json.dumps(event, ensure_ascii=False), flush=True)

    adapter = HeadlessAdapter(interactive=interactive, quiet=quiet,
                              stream=stream if output_format == "stream-json" else None)
    set_adapter(None)  # loader chatter ("[MCP] Loaded config ...") isn't useful here; errors come via adapter
    session_id = session_id or uuid.uuid4().hex[:16]
    turn_id = uuid.uuid4().hex
    session_id_var.set(session_id)
    turn_id_var.set(turn_id)
    data_dir = config.get_data_dir()
    store = SQLiteSessionStore(db_path=str(data_dir / "sessions.db"))
    from dostuff.agent.subagents.registry import get_agent_registry
    from dostuff.helpers.agent.notices import WAKE_LIMIT
    get_agent_registry().attach_store(store)  # agent rows and transcripts (subagents D14)
    on_agent_event = _agent_events(adapter, session_id)
    get_agent_registry().add_listener(on_agent_event)
    mcp_client = MCPClient(
        registration_store=MCPClientRegistrationStore(path=str(data_dir / "mcp_client_registrations.json")),
        tool_registry_store=make_tool_registry_store(),
    )
    started = time.time()
    result = ""
    code = 0
    try:
        # Nobody can answer the trust prompt here, so it is settled before anything runs
        refusal, warning = trust.headless_decision(config.hooks_project_trust, config.hooks, trust_project_hooks)
        if refusal:
            raise _Refused(refusal)
        if warning:
            _err(warning)
            untrusted_notice_given(session_id)
        if output_format == "stream-json":
            from dostuff.lib.model import get_model
            stream({"type": "init", "session_id": session_id, "model": get_model(), "cwd": str(Path.cwd()),
                    "permission_mode": permission_mode or config.permission_mode})
        servers = load_mcp_config()
        if servers:
            await _connect_mcp(mcp_client, servers, adapter)
        base, skills = build_base_instructions()
        meta = await store.get_session_meta(session_id) or {}
        if meta:
            await get_agent_registry().load(session_id, store)  # continuable with send_message (subagents D14)
        steps_history = await store.load(session_id)
        working_history = await store.load_working_history(session_id)
        current_session_history: list = []
        commands = skill_commands(skills, discover_commands())
        skill_run = parse_skill_command(prompt, commands)  # `-p "/<name> args"`
        mentions = skill_mentions(prompt, commands)  # what was typed, before anything expands
        expansion = await asyncio.to_thread(expand_file_mentions, skill_run[1] if skill_run else prompt,
                                            config.max_tool_output_chars, file_lines=config.paste_file_lines)
        if expansion.attached:
            adapter._progress("Attached " + ", ".join(expansion.attached))
        if expansion.skipped:
            adapter.emit("error", "Not attached: " + "; ".join(expansion.skipped))
        if expansion.refused:
            raise _NotAttached("Not attached: " + "; ".join(expansion.refused))
        text = await asyncio.to_thread(skill_prompt, skill_run[0], expansion.content) if skill_run else expansion.content
        text = with_mention_note(text, mentions)
        if mentions:
            adapter._progress(mention_line(mentions))
        # Text first, then the images (multimodal D10); a plain string when nothing visual is attached (CON-007)
        content: str | list[dict[str, Any]] = [{"type": "text", "text": text}, *expansion.parts] if expansion.parts else text
        # SessionStart, then UserPromptSubmit: both before the user step is appended (hooks spec D1)
        session_hook_context: dict[str, str] = {}
        await session_start("resume" if meta else "startup", session_hook_context, session_id=session_id,
                            turn_id=turn_id, adapter=adapter)
        blocked, prompt_context = await user_prompt_submit(content, session_id=session_id, turn_id=turn_id,
                                                           adapter=adapter)
        if blocked is not None:
            raise _Refused(blocked_prompt_message(blocked))
        # The turn's context rides ahead of the prompt as a hidden step, never in the system prompt (D5)
        bundle = await build_turn_context(prompt, memory_store=get_semantic_store(), episodic_store=get_episodic_store(),
                                          user_id=config.get_user_id(), prompt_context=prompt_context)
        if bundle:
            await append_step(make_context_step(bundle), steps_history, working_history,
                              current_session_history, session_id, store, "interactive_loop")
        await append_step({"role": "user", "content": content}, steps_history, working_history,
                          current_session_history, session_id, store, "interactive_loop")
        turn_instructions = build_system_instructions(base)
        permissions = SessionPermissions(mode=permission_mode, allow=list(allowed_tools or []))

        async def turn(text: str, instructions: str, meta: dict[str, Any]) -> str:
            return await agent_loop(
                session_id, turn_id, text, instructions, mcp_client, working_history, "interactive_loop",
                current_session_history, steps_history, store,
                adapter=adapter,
                compaction_notes=meta.get("compaction_notes") or "",
                session_prompt_tokens=meta.get("prompt_tokens") or 0,
                session_completion_tokens=meta.get("completion_tokens") or 0,
                session_total_tokens=meta.get("total_tokens") or 0,
                last_input_tokens=meta.get("last_input_tokens") or 0,
                stream=True,
                max_iterations=max_turns or config.max_iterations,
                permissions=permissions,
                session_hook_context=session_hook_context,
            )
        result = await turn(prompt, turn_instructions, meta)
        # Background agents are waited for: each batch of their notices gets a turn, like the TUI's wake-up
        # (subagents spec D17); past WAKE_LIMIT the rest are stopped in `finally`
        wake_turns = 0
        while not result.startswith(MAX_ITERATIONS_PREFIX):
            if wake_turns >= WAKE_LIMIT:  # checked before waiting: an agent may never end on its own
                if _agents_pending(session_id):
                    _err(WAKE_LIMIT_MESSAGE.format(limit=WAKE_LIMIT))
                break
            if not await _agent_notices(session_id):
                break
            wake_turns += 1
            turn_id = uuid.uuid4().hex
            turn_id_var.set(turn_id)
            meta = await store.get_session_meta(session_id) or {}
            # No user step and no UserPromptSubmit: the loop drains the notices into history itself
            result = await turn("", turn_instructions, meta)
        if result.startswith(MAX_ITERATIONS_PREFIX):
            code = 1
    except _Refused as e:
        _err(str(e))
        code = 2
    except _NotAttached as e:
        _err(str(e))
        code = 1
    except Exception as e:
        _err(f"Error: {e}")
        code = 1
    finally:
        await _stop_agents(adapter)  # first, so their own jobs die with them; also on Ctrl+C (subagents D17)
        get_agent_registry().remove_listener(on_agent_event)
        await _stop_jobs(adapter)  # also on Ctrl+C: nothing outlives the run (background shell spec D8)
        try:
            await asyncio.wait_for(mcp_client.cleanup(), timeout=5.0)
        except Exception:
            pass
        await get_agent_registry().flush()
        await store.close()

    if adapter.denied_without_terminal:
        _err(f"{adapter.denied_without_terminal} call(s) needed approval and were denied. "
             "Allow them with --permission-mode allowlist (+ permissions.allow) or auto.")
    if output_format != "text":
        summary = {
            "session_id": session_id,
            "result": result,
            "is_error": code != 0,
            "duration_s": round(time.time() - started, 2),
            "usage": adapter.usage,
            "cost_usd": round(adapter.cost_usd, 6) if adapter.cost_usd is not None else None,
        }
        if output_format == "stream-json":
            stream({"type": "result", **summary})
        else:
            print(json.dumps(summary, ensure_ascii=False))
    elif result:
        print(result)
    if not quiet:
        _err(f"(session {session_id}; continue with: dostuff -p \"...\" --session {session_id})")
    return code
