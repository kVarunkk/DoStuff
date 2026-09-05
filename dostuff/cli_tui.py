"""Real TUI for dostuff — full agent experience via textual, not skeleton."""
import asyncio
import copy
import os
import uuid
from pathlib import Path
from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import Static, TextArea
from textual.containers import Vertical, VerticalScroll
from textual.binding import Binding
from textual.events import Key

from dostuff.config import Config
from dostuff.helpers.agent.load_identity import load_identity
from dostuff.helpers.agent.load_project_instructions import load_project_instructions
from dostuff.helpers.skills.discover_skills import discover_skills
from dostuff.helpers.agent.constants import SYSTEM_INSTRUCTIONS
from dostuff.lib.memory.session_store import SQLiteSessionStore
from dostuff.lib.memory.semantic_memory_store import SemanticMemoryStore
from dostuff.lib.memory.episodic_memory_store import EpisodicMemoryStore
from dostuff.lib.mcp.mcp_client import MCPClient
from dostuff.lib.mcp.mcp_client_registration_store import MCPClientRegistrationStore
from dostuff.lib.mcp.mcp_tool_registry_store import MCPToolRegistryStore
from dostuff.lib.model import MODEL
from dostuff.lib.tracing import session_id_var, turn_id_var
from dostuff.agent.loop import loop as agent_loop
from dostuff.helpers.agent.append_step import append_step
from dostuff.helpers.agent.save_memories_and_exit import save_memories_and_exit
from dostuff.helpers.mcp.load_mcp_config import load_mcp_config


class EnterSubmits(TextArea):
    """TextArea where Enter submits and Shift+Enter inserts a newline."""

    class Submitted(Message):
        """Sent when user presses Enter (without Shift)."""
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    async def _on_key(self, event: Key) -> None:
        if event.key == "ctrl+j":
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return
        await super()._on_key(event)


class TuiAdapter:
    """
    Translates loop() output to TUI widget updates, and TUI input back to loop().

    loop() calls:
      adapter.emit(event_type, data)    → TUI shows message/tool result
      adapter.ask(message, resume_args)  → TUI prompts user; returns y/n

    TUI calls (from input handler):
      adapter.got_input(text)           → feeds user text back to waiting ask()
    """

    def __init__(self, app):
        self._app = app
        self._waiting_for_confirm = False
        self._confirm_event: asyncio.Event | None = None
        self._confirm_args: dict | None = None
        self._confirm_resume: bool | None = None
        self._confirm_loop: asyncio.AbstractEventLoop | None = None
        self._last_turn_usage: dict = {}
        self._session_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._cancelled: bool = False
    # ── Called by loop() ───────────────────────────────────────────────────────

    def emit(self, event_type: str, data) -> None:
        if event_type == "usage" and isinstance(data, dict):
            self._last_turn_usage = data
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self._session_usage[k] = self._session_usage.get(k, 0) + (data.get(k) or 0)
            self._app.call_from_thread(self._app._update_token_display)
            return
        # Map event_type to msg_type for CSS styling
        type_to_msg = {
            "tool_call": "tool",
            "tool_result": "tool-result",
            "confirm": "confirm",
            "system": "system",
            "usage": "system",
            "error": "error",
        }
        msg_type = type_to_msg.get(event_type, "agent")
        label = event_type.upper().replace("_", " ")
        # Truncate long tool results for clean TUI display
        text = str(data)
        MAX = 600
        if len(text) > MAX:
            text = text[:MAX].rstrip() + f"\n… [+{len(str(data)) - MAX} chars truncated]"
        self._app.call_from_thread(
            lambda: self._app._append(f"[{label}] {text}", msg_type=msg_type)
        )

    async def ask(self, message: str, resume_args: dict) -> bool:
        self._waiting_for_confirm = True
        self._confirm_resume = None
        evt = asyncio.Event()
        self._confirm_event = evt
        self._confirm_loop = asyncio.get_running_loop()
        self._app.call_from_thread(
            lambda: self._app._append(f"\n⚠ {message}\n  (y) allow  (n) deny: ", msg_type="confirm")
        )
        await evt.wait()
        self._waiting_for_confirm = False
        res = self._confirm_resume is True
        self._confirm_resume = None
        self._confirm_event = None
        self._confirm_loop = None
        return res

    # ── Called by TUI input handler ─────────────────────────────────────────────

    def got_input(self, text: str) -> None:
        if self._waiting_for_confirm and self._confirm_event is not None:
            self._confirm_resume = text.strip().lower() == "y"
            if self._confirm_loop is not None:
                self._confirm_loop.call_soon_threadsafe(self._confirm_event.set)

    def has_pending_confirm(self) -> bool:
        return self._waiting_for_confirm


class DostuffTUI(App):
    CSS = """
    Screen { background: transparent; }
    #messages { height: 4fr; background: transparent; }
    #msg-container { height: auto; padding: 0 0; }
    .msg { width: 100%; height: auto; padding: 1 2; }
    .msg-user { background: #4a6fa5; color: white;  }
    .msg-confirm { background: #d4a017; color: #111;  }
    .msg-error { background: #c0392b; color: white;  }
    .msg-system { background: #3a3a3a; color: #ccc; }
    .msg-spacer { height: 2; background: transparent; }
    .msg-tool { background: #1b6e5a; color: white;  }
    .msg-tool-result { background: #2a8c71; color: white;  }
    .msg-agent { color: white; background: transparent; }
    #input { height: 1fr; max-height: 20%; border-top: solid white; border-bottom: solid white; padding: 1; }
    #status { height: auto; color: #888888; text-align: center; padding: 0 1; }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("esc", "cancel", "Cancel", priority=True),
    ]

    # ── Setup (mirrors core/agent.py run() + run_agent()) ──────────────────────

    def __init__(
        self,
        session_id: str,
        **kwargs,
    ):
        print("⏳ Loading session...", flush=True)
        super().__init__(**kwargs)
        self.session_id = session_id
        self.adapter = TuiAdapter(self)
        from dostuff.helpers.ui.emit import set_adapter
        set_adapter(self.adapter)

        # Cwd already switched in cli.py before TUI launched; Config now reads from correct cwd
        print("⏳ Loading config...", flush=True)
        self.config = Config()
        print("⏳ Initializing session store...", flush=True)
        self.store = SQLiteSessionStore(
            db_path=str(self.config.get_data_dir() / "sessions.db")
        )

        self._is_resumed = False
        persist_path = str(self.config.get_data_dir() / "chroma")
        print("⏳ Initializing memory stores...", flush=True)
        self.memory_store = SemanticMemoryStore(persist_path=persist_path)
        self.episodic_store = EpisodicMemoryStore(persist_path=persist_path)
        print("⏳ Initializing MCP client...", flush=True)
        self.reg_store = MCPClientRegistrationStore(
            path=str(self.config.get_data_dir() / "mcp_client_registrations.json")
        )
        self.tool_store = MCPToolRegistryStore(persist_path=persist_path)
        self.mcp_client = MCPClient(
            registration_store=self.reg_store,
            tool_registry_store=self.tool_store,
        )

        # Session state
        self.steps_history: list = []
        self.current_session_history: list = []
        self._setup_done = False
        self._system_instructions: str = ""

    # ── Compose UI ─────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="messages"):
            yield Vertical(id="msg-container")
        yield EnterSubmits(id="input", language="")
        yield Static(id="status")

    # ── On ready: bootstrap session (like run() + start of run_agent()) ────────

    async def on_mount(self) -> None:
        if self._setup_done:
            return
        self._setup_done = True

        # Model env
        if MODEL:
            os.environ.setdefault("MODEL", MODEL)

        # Skills + identity
        skills = discover_skills()
        skills_summary = "\n\n".join(
            f"- name: {s['name']}\n  description: {s['description']}\n  location: {s['location']}"
            for s in skills
        )
        identity = load_identity()
        project_instructions = load_project_instructions()
        self._system_instructions = SYSTEM_INSTRUCTIONS.format(
            dostuff_identity=identity,
            skills_summary=skills_summary or "(none available)",
        )
        if project_instructions:
            self._system_instructions += "\n\n## Project Instructions\n" + project_instructions
        self._pending_workers: list = []  # graceful-shutdown worker tracking

        # Session ID + user ID (single-user: always from Config)
        self.user_id = self.config.get_user_id()
        session_id_var.set(self.session_id)

        # Load history
        self.steps_history = await self.store.load(self.session_id)

        # Save current cwd to session meta for future resumes
        existing_meta = await self.store.get_session_meta(self.session_id)
        wd_to_save = str(Path.cwd())
        await self.store.save_session_meta(self.session_id, wd_to_save)

        # Product branding header
        self._append(
            r"""
▓▓▓▓   ▓▓▓   ▓▓▓▓ ▓▓▓▓▓ ▓   ▓ ▓▓▓▓▓ ▓▓▓▓▓   
▓░░░▓ ▓ ░░▓ ▓ ░░░░ ░▓░░░▓░  ▓░▓░░░░░▓░░░░░  
▓░░░▓░▓░ ░▓░ ▓▓▓░░░ ▓░░░▓░░ ▓░▓▓▓▓░░▓▓▓▓░░░ 
▓░░ ▓░▓░░ ▓░░ ░░▓   ▓░░ ▓░░ ▓░▓░░░░ ▓░░░░   
▓▓▓▓ ░░▓▓▓ ░▓▓▓▓░░  ▓░░  ▓▓▓ ░▓░░░░░▓░░░░░  
 ░░░░ ░ ░░░ ░░░░░ ░  ░░   ░░░ ░░░    ░░     
  ░░░░   ░░░  ░░░░    ░    ░░░  ░     ░     
                                    v0.1.2
            """,
            msg_type="system",
        )
        self._append("", msg_type="spacer")

        if existing_meta:
            self._append(f"▶ Resumed session '{self.session_id}' ({len(self.steps_history)} step(s)).")
            # Render previous steps to Messages (3.1) — preserve tool role with green bg
            for step in self.steps_history:
                role = step.get("role", "?")
                content = step.get("content", "")
                tool_calls = step.get("tool_calls")
                # Match live TuiAdapter.emit() label format: [TOOL CALL] name(args) / [TOOL RESULT] result
                is_tool_call = not content and tool_calls
                is_tool_result = role == "tool"
                if is_tool_call:
                    tc = tool_calls[0].get("function", {})
                    name = tc.get("name", "?")
                    args = tc.get("arguments", "")
                    content = f"[TOOL CALL] {name}({args})"
                    mt = "tool"
                elif is_tool_result:
                    content = f"[TOOL RESULT] {content}"
                    mt = "tool-result"
                elif role == "user":
                    mt = "user"
                    content = content
                else:
                    mt = "agent"
                if content:
                    # Truncate historical tool/agent content the same as live
                    if mt in ("agent", "tool", "tool-result") and len(content) > 600:
                        content = content[:600].rstrip() + f"\n… [+{len(content) - 600} chars truncated]"
                    self._append(content, msg_type=mt)
                    # No spacer in history to keep tool call/result pairs tight
        else:
            self._append(f"▶ New session '{self.session_id}'.")

        # Skills loaded (announced like MCP servers)
        skills = discover_skills()
        if skills:
            skill_names = ", ".join(s["name"] for s in skills)
            self._append(f"Skills loaded: {len(skills)} — {skill_names}", msg_type="system")
        else:
            self._append("Skills loaded: 0 (add to .dostuff/skills/ or ~/.agents/skills/)", msg_type="system")

        # Graceful startup check — bare minimum must be configured
        if not MODEL or "/" not in MODEL:
            self._append(
                "❌ No LLM configured. Set in config.yaml:\n"
                "   model.name = 'openai/gpt-4o-mini'\n"
                "   model.api_key_env = 'OPENAI_API_KEY'\n"
                "Add API key to ~/.dostuff/.env:\n"
                "   OPENAI_API_KEY=sk-...\n"
                "Or: export MODEL=openai/gpt-4o-mini\n"
                "Model name format: provider/model-name\n"
                "Docs: https://docs.litellm.ai/docs/providers\n"
                "See README.md 'Models' and 'Quick Start'.",
                msg_type="error",
            )
            self._update_status(is_resumed=bool(existing_meta))
            return  # don't crash; user can /exit

        self._update_status(is_resumed=bool(existing_meta))

        # MCP servers
        # MCP servers — run in background so TUI renders immediately (fix blank screen)
        mcp_servers = load_mcp_config()
        if mcp_servers:
            self._append(f"Connecting to {len(mcp_servers)} MCP server(s)...", msg_type="system")
            self._pending_workers.append(self.run_worker(
            self._connect_mcp_servers(mcp_servers), name="mcp_connect"
        ))
        else:
            self._append("Running with local tools only.", msg_type="system")

        self._append("Type a message or /help for commands.")

    # ── Input handler (mirrors run_agent() while loop + commands) ───────────────

    def on_enter_submits_submitted(self, event: EnterSubmits.Submitted | None) -> None:
        # Use event.text (captured before any clearing) so confirmation works
        text = event.text if event is not None else ""
        if not text:
            return

        # Check for pending HITL confirmation (4.3) — read text BEFORE clearing
        if self.adapter.has_pending_confirm():
            self.adapter.got_input(text)
            self._clear_input()
            return

        # Clear input immediately (3.3)
        self._clear_input()

        cmd = text.strip().lower()

        # Built-in commands
        if cmd == "/exit":
            self.run_worker(self.action_quit(), name="exit_worker", thread=True)
            return
        # if cmd == "/history":
        #     print_history(self.steps_history)
        #     self._append(f"(Printed {len(self.steps_history)} history step(s))")
        #     return
        if cmd == "/clear":
            self.steps_history = []
            asyncio.create_task(self.store.save(self.session_id, []))
            self._append("History cleared.")
            return
        if cmd == "/help":
            from dostuff.helpers.agent.constants import COMMANDS
            self._append(f"Commands: {', '.join(sorted(COMMANDS))}")
            return

        # User message — no spacer before/after to keep tight with tool/agent messages
        self._append(text, msg_type="user")

        # Normal turn — run via worker (tracked for graceful shutdown)
        self._pending_workers.append(self.run_worker(
            self._run_turn(text),
            name=f"turn_{uuid.uuid4().hex[:6]}",
            thread=True,
        ))

    # ── One agent turn (same pre/post logic as run_agent()) ────────────────────

    async def _run_turn(self, user_text: str) -> None:
        loader_text: str = ""
        loader_widget: Static | None = None
        try:
            turn_id = uuid.uuid4().hex
            turn_id_var.set(turn_id)
            working_history = copy.deepcopy(self.steps_history)

            # Append step
            user_step = {"role": "user", "content": user_text}
            await append_step(
                user_step,
                self.steps_history,
                working_history,
                self.current_session_history,
                self.session_id,
                self.store,
                "interactive_loop",
            )

            # Memory query
            memories = await self.memory_store.query(self.user_id, query_text=user_text, top_k=5)
            memory_text = "\n".join(f"- {m['key']}: {m['value']}" for m in memories)

            episodes = await self.episodic_store.query(
                user_id=self.user_id,
                session_id=self.session_id,
                query_text=user_text,
                top_k=3,
            )
            episodic_text = ""
            if episodes:
                episodic_text = "\n".join(
                    f"  • [{ep['event_type']}] {ep['anchor_event']}\n"
                    f"    Summary: {ep['summary']}\n"
                    f"    Date: {ep['created_at_iso']}"
                    for ep in episodes
                )

            dynamic_instructions = self._system_instructions
            if memory_text:
                dynamic_instructions += f"\n\n<relevant_user_facts>\n{memory_text}\n</relevant_user_facts>"
            if episodic_text:
                dynamic_instructions += f"\n\n<past_episodes>\n{episodic_text}\n</past_episodes>"

            loader_widget = self._show_loader("Working...")
            self._last_turn_usage: dict = {}
            result = await agent_loop(
                self.session_id,
                turn_id,
                user_text,
                dynamic_instructions,
                self.mcp_client,
                working_history,
                "interactive_loop",
                self.current_session_history,
                self.steps_history,
                self.store,
                adapter=self.adapter,
            )

            # Capture agent response for display (real integration — not placeholder)
            agent_text = result or "(no response)"

            def _fmt(n: int) -> str:
                if n < 1000:
                    return str(n)
                if n < 1_000_000:
                    return f"{n/1000:.1f}k"
                return f"{n/1_000_000:.1f}M"

            # Save updated history
            await self.store.save(self.session_id, self.steps_history)

            turn_u = self._last_turn_usage
            tokens_suffix = ""
            if turn_u and (turn_u.get("prompt_tokens") or turn_u.get("completion_tokens")):
                tokens_suffix = f"  [on #2c2c2c] ↑{_fmt(turn_u.get('prompt_tokens', 0))} ↓{_fmt(turn_u.get('completion_tokens', 0))} [/on #2c2c2c]"

            def _show():
                self._append(agent_text, msg_type="agent")
                if tokens_suffix:
                    self._append(tokens_suffix, msg_type="system")
                try:
                    self._stop_loader(loader_widget)
                except Exception:
                    pass
            self._call_from_thread(_show)

        except Exception as e:
            try:
                self._stop_loader(loader_widget)
            except Exception:
                pass
            self._call_from_thread(
                lambda: self._append(f"Turn error: {e}", msg_type="error")
            )

    # ── Loader helpers (fix 4: response/network loader) ─────────────────────────

    def _show_loader(self, label: str = "Working...") -> Static:
        """Show loader message in conversation. Returns the Static widget so it can be cleared."""
        loader_text = f"⏳ {label}"
        loader_widget = Static(loader_text, classes="msg msg-system", markup=False)
        msg_container = self.query_one("#msg-container", Vertical)
        msg_container.mount(loader_widget)
        self._loader_widget = loader_widget
        self._scroll_after_layout()
        return loader_widget

    def _stop_loader(self, loader_widget: Static | None) -> None:
        """Hide loader widget from conversation stream."""
        if loader_widget is None:
            return
        try:
            loader_widget.update("")
            loader_widget.styles.display = "none"
        except Exception:
            pass
        if getattr(self, "_loader_widget", None) is loader_widget:
            self._loader_widget = None

    async def action_cancel(self) -> None:
        """Cancel current agent turn (ESC) — sets adapter flag; loop can check."""
        self.adapter._cancelled = True
        # Also show in status bar
        self.call_from_thread(lambda: self._update_status(working=False, loader="⏹ Cancelled"))
        self._append("Cancelled by user (ESC).", msg_type="system")

    async def action_quit(self) -> None:
        """Threaded worker — _cleanup uses call_from_thread to render messages."""
        self.run_worker(
            self._run_cleanup_and_exit(),
            name="cleanup_worker",
            thread=True,
        )

    async def _run_cleanup_and_exit(self) -> None:
        # Show save message (threaded worker → call_from_thread renders correctly)
        self.call_from_thread(lambda: self._append("Saving session..."))
        try:
            await self._cleanup_and_save_session()
        except Exception:
            pass
        # Schedule exit() on event loop from thread
        self.call_from_thread(self.exit)

    # ── Exit (save memories + cleanup, like run() exit block) ─────────────────

    async def _cleanup_and_save_session(self) -> None:
        """Cancel pending work, save, cleanup, with bounded timeouts."""
        # 1. Cancel any in-flight workers (turn, MCP connect, etc.)
        cancelled = 0
        for w in self._pending_workers:
            if w is not None and not w.is_finished:
                w.cancel()
                cancelled += 1
        self._pending_workers.clear()
        # 2. Wait briefly for cancellations to settle
        if cancelled:
            try:
                await asyncio.sleep(0.2)
            except Exception:
                pass
        # 3. Save memories + sessions
        try:
            await save_memories_and_exit(
                self.current_session_history,
                self.user_id,
                self.session_id,
                self.memory_store,
                self.episodic_store,
            )
            self._append(f"Session saved. Goodbye! (cancelled {cancelled} pending task(s))")
        except Exception as e:
            self._append(f"Save error: {e}")
        finally:
            # 4. MCP cleanup with strict timeout
            try:
                await asyncio.wait_for(self.mcp_client.cleanup(), timeout=3.0)
            except Exception:
                pass
            # 5. Force-stop any lingering litellm logging workers (background threads)
            try:
                import litellm
                if hasattr(litellm, "logging_worker"):
                    worker = litellm.logging_worker
                    if hasattr(worker, "stop"):
                        worker.stop()
            except Exception:
                pass


    # ── Helpers ────────────────────────────────────────────────────────────────

    def _append(self, line: str, msg_type: str = "agent") -> None:
        """Add a full-width message widget with background spanning entire width."""
        msg_container = self.query_one("#msg-container", Vertical)
        css_class = {
            "agent": "msg msg-agent",
            "user": "msg msg-user",
            "confirm": "msg msg-confirm",
            "system": "msg msg-system",
            "error": "msg msg-error",
            "spacer": "msg-spacer",
            "tool": "msg msg-tool",
            "tool-result": "msg msg-tool-result",
        }.get(msg_type, "msg msg-agent")
        msg = Static(line, classes=css_class, markup=False)
        # Mount before loader (so loader stays at bottom of stream)
        loader = getattr(self, "_loader_widget", None)
        if loader is not None and loader.is_mounted:
            msg_container.mount(msg, before=loader)
        else:
            msg_container.mount(msg)
        # Defer scroll until after this mount refreshes so wrapped messages have final height.
        self.call_after_refresh(self._scroll_after_layout)

    def _scroll_after_layout(self) -> None:
        """Scroll the last message widget into full view.

        scroll_end() on VerticalScroll can leave trailing space with auto-height
        content. Scrolling the last widget directly to the bottom edge of the
        scroll view is more reliable.
        """
        try:
            msg_scroll = self.query_one("#messages", VerticalScroll)
            container = self.query_one("#msg-container", Vertical)
            widgets = list(container.query("Static"))
            if widgets:
                widgets[-1].scroll_visible(animate=True)
            else:
                msg_scroll.scroll_end(animate=True)
        except Exception:
            return

    async def _connect_mcp_servers(self, mcp_servers: dict) -> None:
        """Background MCP connection so TUI never blanks."""
        self._append("  ⏳ connecting...", msg_type="system")
        for name, cfg in mcp_servers.items():
            transport = cfg.get("transport", "stdio")
            try:
                await self.mcp_client.connect_to_server(
                    server_name=name,
                    transport=transport,
                    command=cfg.get("command"),
                    args=cfg.get("args", []),
                    env=cfg.get("env"),
                    url=cfg.get("url"),
                    headers=cfg.get("headers"),
                )
                self._call_from_thread(lambda n=name: self._append(f"  ✓ {n}", msg_type="system"))
            except Exception as e:
                self._call_from_thread(lambda n=name, err=str(e): self._append(f"  ✗ {n}: {err}", msg_type="error"))
        self._call_from_thread(lambda: self._append(f"MCP ready ({len(self.mcp_client.servers)} server(s)).", msg_type="system"))

    def _update_token_display(self, working: bool = False, loader: str = "") -> None:
        """Update the status bar with cumulative session tokens (and optional loader)."""
        def _fmt(n: int) -> str:
            if n < 1000:
                return str(n)
            if n < 1_000_000:
                return f"{n/1000:.1f}k"
            return f"{n/1_000_000:.1f}M"
        su = self.adapter._session_usage
        p, c = _fmt(su.get("prompt_tokens", 0)), _fmt(su.get("completion_tokens", 0))
        is_resumed = self._is_resumed
        cwd_str = os.getcwd()
        cwd_display = cwd_str if len(cwd_str) < 36 else "..." + cwd_str[-33:]
        sid_short = self.session_id[:8]
        loader_str = f"  •  {loader}" if (working and loader) else ""
        status_str = f"📁 {cwd_display}  •  sess: {sid_short}  •  {'resumed' if is_resumed else 'new'}  •  ↑{p} ↓{c}{loader_str}"
        self.query_one("#status", Static).update(status_str)

    def _update_status(self, is_resumed: bool = False, working: bool = False, loader: str = "") -> None:
        self._is_resumed = is_resumed
        self._update_token_display(working=working, loader=loader)

    def _call_from_thread(self, fn) -> None:
        """Schedule fn on the main Textual event loop. Safe if already on main thread."""
        import threading
        if threading.current_thread() is threading.main_thread() and threading.get_ident() == threading.current_thread().ident:
            # Already on main thread — call directly (or use call_later to avoid blocking)
            try:
                fn()
            except Exception:
                pass
        else:
            try:
                self.call_from_thread(fn)
            except RuntimeError:
                fn()

    def _clear_input(self) -> None:
        try:
            ta = self.query_one("#input", EnterSubmits)
            ta.text = ""
        except Exception:
            pass

    
    

   

