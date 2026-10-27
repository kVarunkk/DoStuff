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

PREVIEW_LINES = 4
MCP_CONNECT_TIMEOUT_S = 150.0
OutputFormat = Literal["text", "json", "stream-json"]
OUTPUT_FORMATS: tuple[str, ...] = ("text", "json", "stream-json")
MAX_ITERATIONS_PREFIX = "Reached maximum iterations"  # loop()'s answer when it hits the cap


def _err(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


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

    async def ask(self, message: str, always_label: str = "") -> ApprovalReply:
        if not self._interactive:
            self.denied_without_terminal += 1
            self._progress(f"✗ Denied (no terminal to ask): {message.splitlines()[0] if message else ''}")
            return ApprovalReply("no", feedback="Running non-interactively (dostuff -p); nobody can approve this call.")
        _err(f"⚠ {message}")
        choices = "[y]es / [a]lways" if always_label else "[y]es"
        answer = await asyncio.to_thread(input, f"Allow? {choices} / [N]o: ")
        reply = parse_answer(answer)
        return ApprovalReply("yes" if reply == "always" and not always_label else reply)


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


async def run_headless(
    prompt: str,
    session_id: str | None = None,
    permission_mode: PermissionMode | None = None,
    output_format: OutputFormat = "text",
    quiet: bool = False,
    interactive: bool = True,
    max_turns: int | None = None,
    allowed_tools: list[str] | None = None,
) -> int:
    """Run one turn and print the answer; returns the process exit code.

    max_turns caps the model calls (loop.max_iterations otherwise); allowed_tools are patterns
    that run without asking in any mode (see SessionPermissions.allow).
    """
    # Heavy imports here so `dostuff --help` stays fast
    from dostuff.agent.instructions import build_base_instructions, build_turn_instructions
    from dostuff.agent.loop import loop as agent_loop
    from dostuff.config import get_config
    from dostuff.helpers.agent.append_step import append_step
    from dostuff.helpers.agent.file_mentions import expand_file_mentions
    from dostuff.helpers.mcp.load_mcp_config import load_mcp_config
    from dostuff.helpers.skills.custom_commands import discover_commands
    from dostuff.helpers.skills.skill_command import parse_skill_command, skill_commands, skill_prompt
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
    mcp_client = MCPClient(
        registration_store=MCPClientRegistrationStore(path=str(data_dir / "mcp_client_registrations.json")),
        tool_registry_store=make_tool_registry_store(),
    )
    started = time.time()
    result = ""
    code = 0
    try:
        if output_format == "stream-json":
            from dostuff.lib.model import get_model
            stream({"type": "init", "session_id": session_id, "model": get_model(), "cwd": str(Path.cwd()),
                    "permission_mode": permission_mode or config.permission_mode})
        servers = load_mcp_config()
        if servers:
            await _connect_mcp(mcp_client, servers, adapter)
        base, skills = build_base_instructions()
        meta = await store.get_session_meta(session_id) or {}
        steps_history = await store.load(session_id)
        working_history = await store.load_working_history(session_id)
        current_session_history: list = []
        skill_run = parse_skill_command(prompt, skill_commands(skills, discover_commands()))  # `-p "/<name> args"`
        expansion = await asyncio.to_thread(expand_file_mentions, skill_run[1] if skill_run else prompt,
                                            config.max_tool_output_chars)
        if expansion.attached:
            adapter._progress("Attached " + ", ".join(expansion.attached))
        if expansion.skipped:
            adapter.emit("error", "Not attached: " + "; ".join(expansion.skipped))
        content = await asyncio.to_thread(skill_prompt, skill_run[0], expansion.content) if skill_run else expansion.content
        await append_step({"role": "user", "content": content}, steps_history, working_history,
                          current_session_history, session_id, store, "interactive_loop")
        instructions = await build_turn_instructions(base, prompt, get_semantic_store(), get_episodic_store(), config.get_user_id())
        result = await agent_loop(
            session_id, turn_id, prompt, instructions, mcp_client, working_history, "interactive_loop",
            current_session_history, steps_history, store,
            adapter=adapter,
            compaction_notes=meta.get("compaction_notes") or "",
            session_prompt_tokens=meta.get("prompt_tokens") or 0,
            session_completion_tokens=meta.get("completion_tokens") or 0,
            session_total_tokens=meta.get("total_tokens") or 0,
            last_input_tokens=meta.get("last_input_tokens") or 0,
            stream=True,
            max_iterations=max_turns or config.max_iterations,
            permissions=SessionPermissions(mode=permission_mode, allow=list(allowed_tools or [])),
        )
        if result.startswith(MAX_ITERATIONS_PREFIX):
            code = 1
    except Exception as e:
        _err(f"Error: {e}")
        code = 1
    finally:
        try:
            await asyncio.wait_for(mcp_client.cleanup(), timeout=5.0)
        except Exception:
            pass
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
