"""Real TUI for dostuff — full agent experience via textual, not skeleton.

Every turn, /compact, /clear and exit cleanup runs as an *async* worker on Textual's event
loop, so widgets are only touched from the UI thread and MCP sessions live on one loop.
Tools that run in threads (via asyncio.to_thread) reach the UI through TuiAdapter.emit,
which hops back to the UI thread when needed.
"""
import asyncio
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable
from textual.app import App, ComposeResult
from textual.message import Message
from textual.timer import Timer
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option
from textual.containers import Vertical, VerticalScroll
from textual.binding import Binding
from textual.worker import Worker, WorkerState
from rich.text import Text
from rich.markdown import Markdown as RichMarkdown
from textual.events import Key
from dostuff import __version__
from dostuff.config import get_config
from dostuff.agent.permissions import PERMISSION_MODES, ApprovalReply, SessionPermissions, parse_answer
from dostuff.agent.instructions import build_base_instructions, build_turn_instructions
from dostuff.helpers.agent.constants import COMMAND_DESCRIPTIONS
from dostuff.helpers.memory.export_session import session_to_markdown
from dostuff.helpers.memory.shell_step import format_shell_step as _format_shell_step, parse_shell_step as _parse_shell_step
from dostuff.tools.bash.bash_command import run_shell
from dostuff.lib.memory.session_store import SQLiteSessionStore
from dostuff.lib.memory.stores import get_episodic_store, get_semantic_store
from dostuff.lib.mcp.mcp_client import MCPClient
from dostuff.lib.mcp.mcp_client_registration_store import MCPClientRegistrationStore
from dostuff.lib.mcp.mcp_tool_registry_store import MCPToolRegistryStore
from dostuff.lib.checkpoints import UndoResult, get_checkpoint_store
from dostuff.lib.model import get_model
from dostuff.lib.tracing import session_id_var, turn_id_var
from dostuff.agent.loop import EMPTY_RESPONSE_NUDGE, loop as agent_loop, truncate_tool_output
from dostuff.helpers.agent.append_step import append_step
from dostuff.helpers.mcp.load_mcp_config import load_mcp_config
from dostuff.setup_wizard import PROVIDERS, missing_api_keys, read_template
from dostuff.helpers.models.add_model_entry import ModelConfigError, add_model_entry
from dostuff.helpers.models.list_models import ModelListError, provider_models
from dostuff.helpers.agent.get_model_token_limit import get_model_token_limit
from dostuff.helpers.agent.manage_context import compact_context
from dostuff.helpers.ui.emit import set_adapter

MCP_CONNECT_TIMEOUT_S = 150.0  # covers the 120 s OAuth browser callback
SHELL_TIMEOUT_S = 110.0  # `!command` typed by the user
TOOL_RESULT_PREVIEW_LINES = 8  # tool output shown in the conversation; the model gets all of it
SHELL_PREVIEW_LINES = 40
PICKER_MAX_SESSIONS = 9  # /resume picker: number keys reach every option
PICKER_MAX_VISIBLE = 10  # longer option lists scroll with the cursor
TODO_RESULT_PREFIX = "todo_write: "  # loop() emits results as "<tool>: <result>"
# Gemini sends reasoning as whole summary paragraphs ~1 s apart; each new chunk is typed out
# over THOUGHT_REVEAL_TICKS ticks so it reads like streaming instead of popping in as a block
THOUGHT_REVEAL_INTERVAL_S = 0.03
THOUGHT_REVEAL_TICKS = 25


def _preview(text: str, max_lines: int) -> str:
    """First max_lines lines, with a '… +N lines' marker for the rest."""
    lines = text.rstrip("\n").splitlines() or [""]
    extra = len(lines) - max_lines
    if extra > 0:
        lines = lines[:max_lines] + [f"… +{extra} lines"]
    return "\n".join(lines)


def _tool_result_line(text: str, max_lines: int = TOOL_RESULT_PREVIEW_LINES) -> str:
    """Tool output under its call, Claude Code style: '  ⎿  first line' + indented rest."""
    return "  ⎿  " + _preview(text, max_lines).replace("\n", "\n     ")


def _thought_text(text: str) -> Text:
    """Reasoning text with blank-line runs collapsed and `**Heading**` lines (Gemini's thought
    summaries) in bold, a blank line before each heading but the first."""
    out = Text()
    lines = [line.rstrip() for line in text.strip().splitlines()]
    prev_blank = True
    for line in lines:
        if not line:
            prev_blank = True
            continue
        is_heading = line.startswith("**")  # may still be unclosed while being revealed
        if out:
            out.append("\n\n" if is_heading or prev_blank else "\n")
        if is_heading:
            out.append(line.strip("*").strip(), style="bold")
        else:
            out.append(line)
        prev_blank = False
    return out


def _rel_path(path: str) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:  # another drive on Windows
        return path


def _format_undo(result: UndoResult) -> str:
    """The /undo report: one line per file, then what's left for `/undo force`."""
    rows = (
        [f"  restored  {_rel_path(p)}" for p in result.restored]
        + [f"  deleted   {_rel_path(p)} (the agent created it)" for p in result.removed]
        + [f"  kept      {_rel_path(p)} (changed since the agent edited it)" for p in result.conflicts]
        + [f"  skipped   {_rel_path(p)} (too large to snapshot)" for p in result.skipped]
    )
    text = "Undo:\n" + ("\n".join(rows) if rows else "  nothing to change")
    if result.remaining:
        text += "\nRun /undo force to revert the kept files too."
    text += "\n(Changes made by shell commands are not tracked.)"
    return text


def _fmt_time(iso: str) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.strptime(iso[:16], "%Y-%m-%dT%H:%M"))
    except ValueError:
        return iso


class CommandMenu(OptionList):
    """Slash-command suggestions shown above the input while it starts with '/'."""

    can_focus = False

    def show_matches(self, prefix: str) -> bool:
        """List commands starting with prefix; returns False (and hides) if none match."""
        matches = [c for c in COMMAND_DESCRIPTIONS if c.startswith(prefix.lower())]
        self.clear_options()
        if not matches:
            self.display = False
            return False
        width = max(len(c) for c in COMMAND_DESCRIPTIONS) + 4
        self.add_options(
            Option(Text.assemble((c.ljust(width), "bold"), (COMMAND_DESCRIPTIONS[c], "dim")), id=c)
            for c in matches
        )
        self.highlighted = 0
        self.display = True
        return True

    def selected_command(self) -> str | None:
        if not self.display or self.highlighted is None:
            return None
        return self.get_option_at_index(self.highlighted).id


def _confirm_text(details: str) -> Text:
    """Approval details in the confirm color, with diff lines colored like the diff view."""
    text = Text()
    for i, line in enumerate(details.split("\n")):
        if i:
            text.append("\n")
        if line.startswith("+") and not line.startswith("+++"):
            style = "green"
        elif line.startswith("-") and not line.startswith("---"):
            style = "red"
        elif line.startswith("@@"):
            style = "cyan"
        else:
            style = "#e5c07b"
        text.append(line, style=style)
    return text


class PermissionPrompt(Static, can_focus=True):
    """Approval selector shown in place of the input, Claude Code style; also the /resume and /model picker.

    Up/Down (or j/k) move, Enter picks, a number picks that option at once; y / a / n are
    shortcuts for yes / always / no. ESC is the app's cancel binding (deny + cancel the turn,
    or close the picker).

    With `filterable`, typing narrows the options (letters and digits go to the filter, not
    shortcuts), and Enter with no match picks the typed text itself.
    """

    SHORTCUTS = {"y": "yes", "a": "always", "n": "no"}
    APPROVAL_HINT = "↑/↓ select · Enter confirm · Esc deny and stop"

    class Chosen(Message):
        def __init__(self, value: str, source_id: str = "") -> None:
            super().__init__()
            self.value = value
            self.source_id = source_id  # "permission-prompt" or "picker"

    def __init__(self, question: str, options: list[tuple[str, str]], hint: str = APPROVAL_HINT, cursor: int = 0,
                 filterable: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._question = question
        self._all_options = options  # (value, label)
        self._options = options  # the ones matching the filter
        self._hint = hint
        self._cursor = cursor
        self._chosen = False
        self._filterable = filterable
        self._filter = ""

    def on_mount(self) -> None:
        self._render_options()

    def _render_options(self) -> None:
        text = Text(self._question, style="bold")
        if self._filterable:
            text.append(f"\n  Filter: {self._filter}▏", style="#8ab4f8")
            if not self._options:
                typed = self._filter.strip()
                text.append("\n  No match" + (f"; Enter uses '{typed}' as typed" if typed else ""), style="dim")
        count = len(self._options)
        start = max(0, min(self._cursor - PICKER_MAX_VISIBLE // 2, count - PICKER_MAX_VISIBLE))
        end = min(start + PICKER_MAX_VISIBLE, count)
        if start:
            text.append(f"\n  ↑ {start} more", style="dim")
        for i in range(start, end):
            label = self._options[i][1]
            if i == self._cursor:
                text.append(f"\n❯ {i + 1}. {label}", style="bold #8ab4f8")
            else:
                text.append(f"\n  {i + 1}. {label}")
        if end < count:
            text.append(f"\n  ↓ {count - end} more", style="dim")
        text.append("\n" + self._hint, style="dim")
        self.update(text)

    def _choose(self, index: int) -> None:
        if self._chosen:
            return
        self._chosen = True
        self.post_message(self.Chosen(self._options[index][0], source_id=self.id or ""))

    def _set_filter(self, text: str) -> None:
        self._filter = text
        query = text.strip().lower()
        self._options = [o for o in self._all_options if query in o[1].lower()] if query else self._all_options
        self._cursor = 0
        self._render_options()

    def _on_filter_key(self, event: Key) -> bool:
        """Filter mode keys; False when the key isn't one."""
        if event.key == "enter" and not self._options:
            if self._filter.strip() and not self._chosen:
                self._chosen = True
                self.post_message(self.Chosen(self._filter.strip(), source_id=self.id or ""))
        elif event.key == "backspace":
            self._set_filter(self._filter[:-1])
        elif event.is_printable and event.character:
            self._set_filter(self._filter + event.character)
        else:
            return False
        return True

    def on_key(self, event: Key) -> None:
        key, count = event.key, len(self._options)
        values = [value for value, _ in self._options]
        if self._filterable and key not in ("up", "down", "tab") and (key != "enter" or not count):
            if self._on_filter_key(event):
                event.stop()
                event.prevent_default()
            return
        if not count:
            return
        if key in ("up", "k"):
            self._cursor = (self._cursor - 1) % count
            self._render_options()
        elif key in ("down", "j", "tab"):
            self._cursor = (self._cursor + 1) % count
            self._render_options()
        elif key == "enter":
            self._choose(self._cursor)
        elif key.isdigit() and 1 <= int(key) <= count:
            self._choose(int(key) - 1)
        elif self.SHORTCUTS.get(key) in values:
            self._choose(values.index(self.SHORTCUTS[key]))
        else:
            return
        event.stop()
        event.prevent_default()


class EnterSubmits(TextArea):
    """TextArea where Enter submits and Ctrl+J inserts a newline.

    While the "/" menu is open, Up/Down move through it, Tab completes and Enter runs the
    highlighted command.
    """

    class Submitted(Message):
        """Sent when user presses Enter."""
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def _menu(self) -> CommandMenu | None:
        try:
            menu = self.app.query_one("#command-menu", CommandMenu)
        except Exception:
            return None
        return menu if menu.display else None

    async def _on_key(self, event: Key) -> None:
        menu = self._menu()
        if menu is not None and event.key in ("up", "down", "tab", "enter"):
            event.stop()
            event.prevent_default()
            if event.key == "up":
                menu.action_cursor_up()
            elif event.key == "down":
                menu.action_cursor_down()
            else:
                command = menu.selected_command()
                menu.display = False
                if command and event.key == "tab":
                    self.text = command
                    self.move_cursor(self.document.end)
                elif command:
                    self.post_message(self.Submitted(command))
            return
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
      adapter.emit(event_type, data)    → TUI shows message/tool result (any thread)
      adapter.ask(message)              → TUI shows a selector; returns an ApprovalReply
                                          ("yes" | "always" | "no", plus feedback for "no")
      adapter.ask_batch(messages)       → one selector for several calls; adds "each"

    TUI calls:
      adapter.choose(value)             → an option was picked in the selector
      adapter.got_input(text)           → typed reply (feedback after "No", or y/a/n)
    """

    def __init__(self, app: "DostuffTUI"):
        self._app = app
        self._waiting_for_confirm = False
        self._confirm_event: asyncio.Event | None = None
        self._confirm_reply: ApprovalReply | None = None
        self._feedback_mode = False  # "No" was picked; the input now takes feedback for the model
        self._ask_lock = asyncio.Lock()  # one prompt at a time
        self._last_turn_usage: dict = {}
        self._session_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._cancelled: bool = False
        self._cancel_event: asyncio.Event = asyncio.Event()
        self._last_context_window_percent = ""
        self._token_display = ""

    def reset_usage(self) -> None:
        self._last_turn_usage = {}
        self._session_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._token_display = " ↑0 ↓0"
        self._last_context_window_percent = ""

    # ── Called by loop() ───────────────────────────────────────────────────────

    def emit(self, event_type: str, data) -> None:
        self._app.ui(lambda: self._emit_on_ui(event_type, data))

    def _emit_on_ui(self, event_type: str, data) -> None:
        app = self._app
        if event_type == "usage" and isinstance(data, dict):
            # Display-only cache for the status bar; loop() gets its token totals from session meta.
            self._last_turn_usage = data
            for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
                self._session_usage[k] = self._session_usage.get(k, 0) + (data.get(k) or 0)
            su = self._session_usage
            if su.get("prompt_tokens", 0) or su.get("completion_tokens", 0):
                cached = f" ({app._fmt_tokens(su['cached_tokens'])} cached)" if su.get("cached_tokens") else ""
                self._token_display = f" ↑{app._fmt_tokens(su.get('prompt_tokens', 0))}{cached} ↓{app._fmt_tokens(su.get('completion_tokens', 0))}"
            app._update_status(context_window_percent=self._last_context_window_percent)
            return
        if event_type == "status" and isinstance(data, dict):
            percent_str_raw = data.get("context_window_percent", "")
            limit_raw = data.get("token_limit", 0)
            limit_str = app._fmt_tokens(limit_raw) if isinstance(limit_raw, int) and limit_raw else ""
            self._last_context_window_percent = f"{percent_str_raw}/{limit_str}" if limit_str else percent_str_raw
            app._update_status(context_window_percent=self._last_context_window_percent)
            return
        if event_type == "partial":
            app._update_stream_widget("answer", str(data))
            return
        if event_type == "partial_thought":
            app._update_stream_widget("thought", str(data))
            return
        if event_type == "stream_end":
            app._finish_stream_widgets()
            return
        if event_type == "stream_interrupted":
            app._finish_stream_widgets(interrupted=True)
            return
        if event_type == "diff":
            app._append(str(data), msg_type="diff")
            return
        if event_type == "tool_call":
            app._append(f"● {data}", msg_type="tool")
            return
        if event_type == "tool_result":
            text = str(data)
            if text.startswith(TODO_RESULT_PREFIX):  # the plan is short and worth seeing in full
                app._append(_tool_result_line(text.removeprefix(TODO_RESULT_PREFIX), max_lines=100), msg_type="tool-result")
                return
            app._append(_tool_result_line(text), msg_type="tool-result")
            return
        if event_type in ("confirm", "system", "error"):
            app._append(str(data), msg_type=event_type)
            return
        label = event_type.upper().replace("_", " ")
        app._append(f"[{label}] {data}", msg_type="agent")

    async def ask(self, message: str, always_label: str = "") -> ApprovalReply:
        options = [("yes", "Yes")]
        if always_label:
            options.append(("always", f"Yes, and don't ask again this session for {always_label}"))
        options.append(("no", "No, and tell the model what to do differently"))
        return await self._prompt(f"⚠ {message}", "Do you want to proceed?", options)

    async def ask_batch(self, messages: list[str], always_label: str = "") -> ApprovalReply:
        count = len(messages)
        details = f"⚠ The model wants to run {count} actions that need approval:"
        for i, message in enumerate(messages, 1):
            details += f"\n\n[{i}/{count}] {message}"
        options = [("yes", f"Yes, allow all {count}")]
        if always_label:
            options.append(("always", f"Yes to all, and don't ask again this session for {always_label}"))
        options += [("each", "Let me review each one"), ("no", "No to all, and tell the model what to do differently")]
        return await self._prompt(details, f"Allow these {count} actions?", options)

    async def _prompt(self, details: str, question: str, options: list[tuple[str, str]]) -> ApprovalReply:
        # Runs on the UI loop (turns are async workers), so widgets can be touched directly
        async with self._ask_lock:
            self._waiting_for_confirm = True
            self._feedback_mode = False
            self._confirm_reply = None
            evt = asyncio.Event()
            self._confirm_event = evt
            self._app._mount_message(Static(_confirm_text(details), classes="msg msg-confirm", markup=False))
            self._app._show_permission_prompt(question, options)
            try:
                await evt.wait()
            finally:
                self._waiting_for_confirm = False
                self._feedback_mode = False
                self._confirm_event = None
                self._app._hide_permission_prompt()
            reply = self._confirm_reply or ApprovalReply("no")
            self._confirm_reply = None
            return reply

    def _resolve(self, reply: ApprovalReply) -> None:
        if self._waiting_for_confirm and self._confirm_event is not None:
            self._confirm_reply = reply
            self._confirm_event.set()

    # ── Called by the TUI ───────────────────────────────────────────────────────

    def choose(self, value: str) -> None:
        """An option was picked; "no" first asks for (optional) feedback in the input."""
        if not self._waiting_for_confirm:
            return
        if value == "no":
            self._feedback_mode = True
            self._app._ask_denial_feedback()
            return
        self._resolve(ApprovalReply(value))  # type: ignore[arg-type]

    def got_input(self, text: str) -> None:
        if self._feedback_mode:
            self._resolve(ApprovalReply("no", feedback=text))
        else:
            self._resolve(ApprovalReply(parse_answer(text)))

    def deny_pending_confirm(self) -> None:
        """ESC while a prompt is open: resolve it as "no"."""
        self._resolve(ApprovalReply("no"))

    def has_pending_confirm(self) -> bool:
        return self._waiting_for_confirm


class DostuffTUI(App):
    # Message kinds differ by prefix (">", "●", "⎿", "!") and text color; only user messages get a
    # grey band (like Claude Code) so turns are easy to find when scrolling.
    CSS = """
    Screen { background: transparent; }
    #messages { height: 1fr; background: transparent; scrollbar-size-vertical: 1; }
    #msg-container { height: auto; min-height: 100%; }
    .msg { width: 100%; height: auto; padding: 0 1; margin-top: 1; background: transparent; }
    .msg-user { color: #d0d0d0; background: #373737; }
    .msg-confirm { color: #e5c07b; }
    #permission-prompt {
        height: auto; margin-top: 1; padding: 0 1; background: transparent; border: round #e5c07b;
    }
    #picker {
        height: auto; margin-top: 1; padding: 0 1; background: transparent; border: round #8ab4f8;
    }
    .msg-error { color: #e06c75; }
    .msg-system { color: #8a8a8a; }
    .msg-thought { color: #8a8a8a; text-style: italic; }
    .msg-tool { }
    .msg-tool-result { color: #8a8a8a; margin-top: 0; }
    .msg-diff { margin-top: 0; }
    .msg-shell { color: #d7769f; }
    .msg-shell-output { color: #8a8a8a; margin-top: 0; }
    #command-menu {
        height: auto; max-height: 8; display: none; background: transparent;
        border: none; padding: 0 1; scrollbar-size-vertical: 1;
    }
    #command-menu > .option-list--option-highlighted { background: transparent; color: #8ab4f8; }
    #command-menu > .option-list--option-hover { background: transparent; }
    #input {
        height: auto; min-height: 3; max-height: 12; margin-top: 1;
        background: transparent; border: round #4a4a4a; padding: 0 1;
    }
    #input:focus { border: round #7a7a7a; }
    #input.shell-mode { border: round #d7769f; }
    #input.feedback-mode { border: round #e5c07b; }
    #input > .text-area--cursor-line { background: transparent; }
    #status { height: 1; color: #8a8a8a; padding: 0 1; }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
    ]

    # ── Setup ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        session_id: str,
        **kwargs,
    ):
        print(" Loading session...", flush=True)
        # ansi_color + an ANSI theme make "transparent" mean the terminal's own background;
        # without them Textual paints the theme's RGB background (black) behind everything.
        kwargs.setdefault("ansi_color", True)
        super().__init__(**kwargs)
        self.theme = "ansi-dark"
        self.session_id = session_id
        self.adapter = TuiAdapter(self)
        set_adapter(self.adapter)

        # Cwd already switched in cli.py before TUI launched; Config now reads from correct cwd
        print(" Loading config...", flush=True)
        self.config = get_config()
        print(" Initializing session store...", flush=True)
        self.store = SQLiteSessionStore(
            db_path=str(self.config.get_data_dir() / "sessions.db")
        )

        persist_path = str(self.config.get_data_dir() / "chroma")
        print(" Initializing memory stores...", flush=True)
        # Same instances the save_memory / save_episode tools write to
        self.memory_store = get_semantic_store()
        self.episodic_store = get_episodic_store()
        print(" Initializing MCP client...", flush=True)
        self.reg_store = MCPClientRegistrationStore(
            path=str(self.config.get_data_dir() / "mcp_client_registrations.json")
        )
        self.tool_store = MCPToolRegistryStore(persist_path=persist_path)
        self.mcp_client = MCPClient(
            registration_store=self.reg_store,
            tool_registry_store=self.tool_store,
        )

        # Session state
        self.user_id = ""
        self.steps_history: list = []
        self.current_session_history: list = []
        self._setup_done = False
        self._system_instructions: str = ""
        self._loader_widget: Static | None = None  # loader widget at bottom of stream
        self._stream_widgets: dict[str, Static] = {}  # "answer" / "thought" → live widget
        self._stream_texts: dict[str, str] = {}
        self._thought_shown = 0  # chars of the live thought text revealed so far
        self._thought_step = 1
        self._thought_timer: Timer | None = None
        self._last_streamed_answer: str = ""
        self._prompt_queue: list = []
        self._pending_workers: list[Worker] = []
        self._turn_in_progress: bool = False
        self._turn_start_time: float | None = None
        self._timer: Timer | None = None
        self._quitting = False
        self.max_iterations = self.config.max_iterations
        # Session-scoped approvals ("always") — reset when the app restarts
        self.permissions = SessionPermissions()
        self._draft_before_feedback = ""  # input text set aside while it takes denial feedback
        self._picker_callback: Callable[[str], Awaitable[None]] | None = None
        self._mcp_configured: list[str] = []

    # ── Compose UI ─────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="messages"):
            yield Vertical(id="msg-container")
        yield CommandMenu(id="command-menu")
        yield EnterSubmits(id="input", language="")
        yield Static(id="status")

    def ui(self, fn: Callable[[], Any]) -> None:
        """Run fn on the UI thread (directly if already there)."""
        if threading.get_ident() == getattr(self, "_thread_id", None):
            fn()
            return
        try:
            self.call_from_thread(fn)
        except RuntimeError:
            pass  # app is not running (shutting down)

    def _start_worker(self, work, name: str) -> Worker:
        worker = self.run_worker(work, name=name, exit_on_error=False)
        self._pending_workers.append(worker)
        return worker

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            if event.worker in self._pending_workers:
                self._pending_workers.remove(event.worker)

    # ── On ready: bootstrap session ─────────────────────────────────────────────

    async def on_mount(self) -> None:
        if self._setup_done:
            return
        self._setup_done = True
        # Anchored: the compositor keeps the view at the bottom in the same layout pass
        # (no flicker); scrolling up releases it, scrolling back to the bottom restores it.
        self.query_one("#messages", VerticalScroll).anchor()
        self.query_one("#input", EnterSubmits).focus()

        self._system_instructions, skills = build_base_instructions()
        self.user_id = self.config.get_user_id()
        existing_meta = await self._load_session(self.session_id)

        self._append(
            r"""
▓▓▓▓   ▓▓▓   ▓▓▓▓ ▓▓▓▓▓ ▓   ▓ ▓▓▓▓▓ ▓▓▓▓▓
▓░░░▓ ▓ ░░▓ ▓ ░░░░ ░▓░░░▓░  ▓░▓░░░░░▓░░░░░
▓░░░▓░▓░ ░▓░ ▓▓▓░░░ ▓░░░▓░░ ▓░▓▓▓▓░░▓▓▓▓░░░
▓░░ ▓░▓░░ ▓░░ ░░▓   ▓░░ ▓░░ ▓░▓░░░░ ▓░░░░
▓▓▓▓ ░░▓▓▓ ░▓▓▓▓░░  ▓░░  ▓▓▓ ░▓░░░░░▓░░░░░
 ░░░░ ░ ░░░ ░░░░░ ░  ░░   ░░░ ░░░    ░░
  ░░░░   ░░░  ░░░░    ░    ░░░  ░     ░
"""
            + f"{'v' + __version__:>42}\n",
            msg_type="system",
        )

        if existing_meta:
            self._append(f" Resumed session '{self.session_id}' ({len(self.steps_history)} step(s)).")
            self._render_history()
        else:
            self._append(f" New session '{self.session_id}'.")

        if skills:
            skill_names = ", ".join(s["name"] for s in skills)
            self._append(f"Skills loaded: {len(skills)} — {skill_names}", msg_type="system")
        else:
            self._append("Skills loaded: 0 (add to .dostuff/skills/ or ~/.agents/skills/)", msg_type="system")

        if self.config.permission_mode == "auto":
            self._append(
                "⚠ permissions.mode is 'auto': every tool call (shell, writes, deletes, MCP) runs WITHOUT asking.",
                msg_type="error",
            )

        self._warn_if_model_unconfigured()
        self._update_status()

        # MCP servers connect in the background so the TUI renders immediately
        mcp_servers = load_mcp_config()
        self._mcp_configured = list(mcp_servers)
        if mcp_servers:
            self._append(f"Connecting to {len(mcp_servers)} MCP server(s)...", msg_type="system")
            self._start_worker(self._connect_mcp_servers(mcp_servers), name="mcp_connect")
        else:
            self._append("Running with local tools only.", msg_type="system")

        self._append("Type a message or /help for commands.")

    async def _load_session(self, session_id: str) -> dict | None:
        """Make session_id current: load its steps and token counters; returns its meta (None if new).

        A stored session gets this cwd and a fresh last_used; a new one gets its DB row with its
        first step, so launches that never send anything don't leave empty sessions behind.
        """
        self.session_id = session_id
        session_id_var.set(session_id)
        self.steps_history = await self.store.load(session_id)
        self.current_session_history = []
        existing_meta = await self.store.get_session_meta(session_id)
        meta = existing_meta or {}
        if existing_meta:
            await self.store.save_session_meta(
                session_id,
                working_dir=str(Path.cwd()),
                prompt_tokens=meta.get("prompt_tokens") or 0,
                completion_tokens=meta.get("completion_tokens") or 0,
                total_tokens=meta.get("total_tokens") or 0,
                compaction_notes=meta.get("compaction_notes") or "",
                last_input_tokens=meta.get("last_input_tokens") or 0,
            )
        self.adapter.reset_usage()
        meta_p = meta.get("prompt_tokens") or 0
        meta_c = meta.get("completion_tokens") or 0
        self.adapter._session_usage["prompt_tokens"] = meta_p
        self.adapter._session_usage["completion_tokens"] = meta_c
        self.adapter._session_usage["total_tokens"] = meta.get("total_tokens") or 0
        self.adapter._token_display = f" ↑{self._fmt_tokens(meta_p)} ↓{self._fmt_tokens(meta_c)}"
        self._refresh_context_status(meta.get("last_input_tokens") or 0)
        return existing_meta

    def _refresh_context_status(self, last_input_tokens: int) -> None:
        token_limit = get_model_token_limit()
        percent = (last_input_tokens / token_limit * 100) if token_limit > 0 else 0
        limit_str = self._fmt_tokens(token_limit) if token_limit else ""
        self._update_status(context_window_percent=f"{percent:.1f}%/{limit_str}" if limit_str else f"{percent:.1f}%")

    def _render_history(self) -> None:
        for step in self.steps_history:
            self._render_history_step(step)

    def _warn_if_model_unconfigured(self) -> None:
        model = get_model()
        missing = missing_api_keys(model)
        if not missing:
            return
        self._append(
            f"❌ No API key found for model '{model}' (missing: {', '.join(missing)}).\n"
            "Quit (/exit) and run `dostuff setup`, or add it to ~/.dostuff/.env:\n"
            f"   {missing[0]}=...\n"
            "Docs: https://docs.litellm.ai/docs/providers",
            msg_type="error",
        )

    def _render_history_step(self, step: dict) -> None:
        """Render a stored step like the live adapter does (● call / ⎿ result)."""
        role = step.get("role", "?")
        content = step.get("content") or ""
        if role == "tool":
            self._append(_tool_result_line(f"{step.get('name', '')}: {content}"), msg_type="tool-result")
            return
        if role == "user":
            shell = _parse_shell_step(content)
            if shell is not None:
                self._append(f"! {shell[0]}", msg_type="shell")
                self._append(_tool_result_line(shell[1], SHELL_PREVIEW_LINES), msg_type="shell-output")
            elif content and content != EMPTY_RESPONSE_NUDGE:
                self._append(content, msg_type="user")
            return
        if content:
            self._append(content, msg_type="agent")
        for tc in step.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            self._append(f"● {fn.get('name', '?')}({fn.get('arguments', '')})", msg_type="tool")

    # ── Commands ──────────────────────────────────────────────────────────────

    async def _compact_session(self) -> None:
        loader_widget = self._show_loader("Compacting...")
        self._update_status(loader="Compacting")
        try:
            meta = await self.store.get_session_meta(self.session_id) or {}
            notes = meta.get("compaction_notes") or ""
            working_history = await self.store.load_working_history(self.session_id)
            budget = min(int(get_model_token_limit() * 0.15), 20000)
            try:
                recent, summary = await asyncio.wait_for(compact_context(working_history, budget, notes), timeout=120)
            except asyncio.TimeoutError:
                self._append("Compact timed out", msg_type="error")
                return
            if summary:
                await self.store.compact_working_history(self.session_id, len(recent), summary)
                self._append(f"Context compacted. Notes: {summary}", msg_type="system")
            else:
                self._append("Nothing to compact yet.", msg_type="system")
        except Exception as e:
            self._append(f"Compact failed: {e}", msg_type="error")
        finally:
            self._stop_loader(loader_widget)
            self._update_status(loader="")

    async def _clear_session(self) -> None:
        """Start a new session (like Claude Code); the old one stays in the DB, resumable with /resume.

        Todos and undo checkpoints are keyed by session id, so they start fresh too.
        Session-scoped approvals ("always") are kept. The new session's DB row is created with
        its first step.
        """
        old_session_id = self.session_id
        new_session_id = uuid.uuid4().hex[:16]
        self.session_id = new_session_id
        self.steps_history = []
        self.current_session_history = []
        self.adapter.reset_usage()
        await self.query_one("#msg-container", Vertical).remove_children()
        token_limit = get_model_token_limit()
        self._update_status(context_window_percent=f"0.0%/{self._fmt_tokens(token_limit)}" if token_limit else "0.0%")
        self._append(
            f" New session '{new_session_id}'. Previous session saved; /resume or `dostuff resume {old_session_id}` brings it back.",
            msg_type="system",
        )

    async def _undo_changes(self, force: bool) -> None:
        """Revert the newest file checkpoint, then tell the model which files went back."""
        try:
            result = await asyncio.to_thread(get_checkpoint_store().undo, self.session_id, force)
        except Exception as e:
            self._append(f"Undo failed: {e}", msg_type="error")
            return
        if result is None:
            self._append("Nothing to undo: the agent hasn't changed any files in this session.", msg_type="system")
            return
        self._append(_format_undo(result), msg_type="system")
        if not result.changed:
            return
        note = (
            "[/undo] The user reverted your file changes from an earlier turn. "
            + "; ".join(part for part in (
                f"Restored: {', '.join(_rel_path(p) for p in result.restored)}" if result.restored else "",
                f"Deleted (files you created): {', '.join(_rel_path(p) for p in result.removed)}" if result.removed else "",
            ) if part)
            + ". Re-read these files before relying on their contents."
        )
        try:
            working_history = await self.store.load_working_history(self.session_id)
            await append_step(
                {"role": "user", "content": note},
                self.steps_history,
                working_history,
                self.current_session_history,
                self.session_id,
                self.store,
                "interactive_loop",
            )
        except Exception as e:
            self._append(f"Files were reverted, but the note for the model could not be saved: {e}", msg_type="error")

    async def _resume_command(self, arg: str) -> None:
        """/resume: pick one of this directory's sessions; /resume <id or prefix> switches directly."""
        try:
            here = await self.store.list_sessions(working_dir=os.getcwd())
            if arg:
                rows = await self.store.list_sessions()
                matches = [r for r in rows if r["session_id"] == arg] or [r for r in rows if r["session_id"].startswith(arg)]
                if len(matches) != 1:
                    self._append(f"No session matches '{arg}'." if not matches else f"'{arg}' matches {len(matches)} sessions; give more characters.", msg_type="error")
                    return
                row = matches[0]
                if row not in here:
                    self._append(
                        f"Session {row['session_id']} is in {row['working_dir']}. Quit and run `dostuff resume {row['session_id']}` to open it there.",
                        msg_type="error",
                    )
                    return
                await self._switch_session(row["session_id"])
                return
        except Exception as e:
            self._append(f"Could not list sessions: {e}", msg_type="error")
            return
        others = [r for r in here if r["session_id"] != self.session_id][:PICKER_MAX_SESSIONS]
        if not others:
            self._append("No other sessions in this directory. `dostuff sessions --all` lists every directory.", msg_type="system")
            return
        options = [
            (r["session_id"], f"{_fmt_time(r['last_used'])}  {r['steps']:>4} steps  {(r['title'] or '(no prompt)')[:70]}")
            for r in others
        ]
        self._show_picker("Resume which session?", options, self._switch_session)

    async def _switch_session(self, session_id: str) -> None:
        if self._turn_in_progress:
            self._append("Can't switch sessions while a turn is running (press ESC to cancel it first).", msg_type="error")
            return
        if session_id == self.session_id:
            self._append("That's the current session.", msg_type="system")
            return
        old_session_id, had_steps = self.session_id, bool(self.steps_history)
        try:
            await self._load_session(session_id)
        except Exception as e:
            self._append(f"Resume failed: {e}", msg_type="error")
            return
        await self.query_one("#msg-container", Vertical).remove_children()
        self._append(f" Resumed session '{session_id}' ({len(self.steps_history)} step(s)).", msg_type="system")
        self._render_history()
        if had_steps:
            self._append(f" Previous session '{old_session_id}' saved; /resume brings it back.", msg_type="system")

    _BROWSE_PREFIX = "browse:"  # picker value for "pick a model from this provider's list"

    def _model_command(self, arg: str) -> None:
        """/model [<name>] [--active]: one picker for saved and new models.

        With no name: saved models (config.yaml's `models:`) first, then each provider's model
        list; type to filter, or type any provider/model. A model that isn't saved yet is added to
        ~/.dostuff/config.yaml and switched to; --active also makes it the default.
        (`/model add ...` still works, as before.)
        """
        args = arg.split()
        if args[:1] == ["add"]:
            args = args[1:]
        active = "--active" in args
        names = [a for a in args if a != "--active"]
        if not names:
            self._show_model_picker(active)
            return
        if len(names) != 1 or names[0].startswith("-"):
            self._append(
                "Usage: /model [<provider/model>] [--active], e.g. /model anthropic/claude-sonnet-5-5\n"
                "With no name you pick from a list. --active makes it the default; "
                "`dostuff models add --help` has more options (e.g. --api-key-env).",
                msg_type="error",
            )
            return
        self._start_worker(self._choose_model(names[0], active), name="model")

    def _show_model_picker(self, active: bool) -> None:
        current = get_model()
        options = [
            (m["name"], m["name"] + ("  (current)" if m["name"] in (current, current.split("/", 1)[-1]) else "")
             + ("  (default)" if m["active"] else ""))
            for m in self.config.configured_models()
        ]
        cursor = next((i for i, (_, label) in enumerate(options) if "(current)" in label), 0)
        options += [(self._BROWSE_PREFIX + p, f"More {p} models…") for p in PROVIDERS]
        self._show_picker(
            f"Model: {current}. Switch to? (or type any provider/model)",
            options, lambda value: self._choose_model(value, active), cursor=cursor, filterable=True,
        )

    async def _choose_model(self, value: str, active: bool) -> None:
        """A /model pick: browse a provider, switch to a saved model, or add a new one and switch."""
        if value.startswith(self._BROWSE_PREFIX):
            await self._pick_model_to_add(value[len(self._BROWSE_PREFIX):], active)
            return
        models = self.config.configured_models()
        saved = any(value in (m["name"], m["name"].split("/", 1)[-1]) for m in models)
        if saved and not active:
            await self._switch_model(value)
        elif "/" in value:
            await self._add_model(value, active)
        elif saved:
            entry = next(m for m in models if value in (m["name"], m["name"].split("/", 1)[-1]))
            await self._add_model(entry["name"], active)
        else:
            known = ", ".join(m["name"] for m in models) or "none"
            self._append(f"Unknown model '{value}'. Saved: {known}. Or give a full litellm name like openai/gpt-4o.",
                         msg_type="error")

    async def _pick_model_to_add(self, provider: str, active: bool) -> None:
        """Second step of /model: the provider's models, filterable; a typed name is used as given."""
        if provider in ("ollama", "openrouter"):
            self._append(f"Fetching {provider} models...", msg_type="system")
        try:
            models = await asyncio.to_thread(provider_models, provider)
        except ModelListError as e:
            models = []
            self._append(f"No model list: {e}. Type the model name instead.", msg_type="error")
        default = PROVIDERS[provider][0]
        cursor = models.index(default) if default in models else 0
        self._show_picker(
            f"Which {provider} model? ({len(models)} known; or type any name)",
            [(m, m) for m in models],
            lambda name: self._add_model(name if name.startswith(provider + "/") else f"{provider}/{name}", active),
            cursor=cursor, filterable=True,
        )

    async def _add_model(self, name: str, active: bool) -> None:
        if not active and name in [m["name"] for m in self.config.configured_models()]:
            await self._switch_model(name)  # already listed: just switch
            return
        provider = name.split("/", 1)[0]
        path = self.config.global_path
        try:
            is_new = add_model_entry(path, name, provider, active=active, template=read_template("config.example.yaml"))
        except (ModelConfigError, OSError) as e:
            self._append(f"Could not add '{name}': {e}", msg_type="error")
            return
        self.config.load()  # the picker reads the new list
        entry = next((m for m in self.config.configured_models() if m["name"] == name), None)
        is_default = active or bool(entry and entry["active"])
        self.config.use_model(name, provider, entry["api_key_env"] if entry else "")
        saved = f"Added '{name}' to {path}" if is_new else f"'{name}' is in {path}"
        message = f"{saved}{' as the default model' if is_default else ''}. Model: {get_model()}" + (
            "." if is_default else " for this session (pick it again any time with /model).")
        if not entry:
            message += "\nNote: ./.dostuff/config.yaml has its own `models:` list, so /model won't show it as saved in this directory."
        self._append(message, msg_type="system")
        self._warn_if_model_unconfigured()

    async def _switch_model(self, name: str) -> None:
        if self._turn_in_progress:
            self._append("Can't switch models while a turn is running (press ESC to cancel it first).", msg_type="error")
            return
        models = self.config.configured_models()
        entry = next((m for m in models if m["name"] == name), None) or next(
            (m for m in models if m["name"].split("/", 1)[-1] == name), None)
        if entry is None and "/" not in name:
            known = ", ".join(m["name"] for m in models) or "none"
            self._append(f"Unknown model '{name}'. Configured: {known}. Or give a full litellm name like openai/gpt-4o.", msg_type="error")
            return
        entry = entry or {"name": name, "provider": "", "api_key_env": ""}
        self.config.use_model(entry["name"], entry["provider"], entry["api_key_env"])
        self._append(
            f"Model: {get_model()} (this session only; /model {entry['name']} --active makes it the default).",
            msg_type="system",
        )
        self._warn_if_model_unconfigured()
        meta = await self.store.get_session_meta(self.session_id) or {}
        self._refresh_context_status(meta.get("last_input_tokens") or 0)

    _PERMISSION_MODE_LABELS = {
        "ask": "ask        every write / delete / shell / MCP call asks",
        "allowlist": "allowlist  as ask, plus config's permissions.allow patterns run without asking",
        "auto": "auto       nothing asks (not recommended)",
    }

    def _permissions_command(self, arg: str) -> None:
        """/permissions: pick a permission mode; /permissions <mode> switches directly (session only)."""
        current = self.permissions.mode or self.config.permission_mode
        if not arg:
            options = [(m, self._PERMISSION_MODE_LABELS[m] + ("  (current)" if m == current else "")) for m in PERMISSION_MODES]
            self._show_picker("Permission mode (this session only)", options, self._switch_permission_mode,
                              cursor=PERMISSION_MODES.index(current))
            return
        self._start_worker(self._switch_permission_mode(arg.lower()), name="permissions")

    async def _switch_permission_mode(self, mode: str) -> None:
        if mode not in PERMISSION_MODES:
            self._append(f"Unknown permission mode '{mode}'. Use one of: {', '.join(PERMISSION_MODES)}.", msg_type="error")
            return
        self.permissions.mode = mode  # type: ignore[assignment]
        self._update_status()
        self._append(
            f"Permissions: {mode} (this session only; set permissions.mode in config.yaml to change the default).",
            msg_type="system",
        )
        if mode == "auto":
            self._append("⚠ Every tool call (shell, writes, deletes, MCP) now runs WITHOUT asking.", msg_type="error")
        elif mode == "allowlist" and not self.config.permission_allow:
            self._append("permissions.allow in config.yaml is empty, so allowlist behaves like ask.", msg_type="system")

    async def _show_status(self) -> None:
        meta = await self.store.get_session_meta(self.session_id) or {}
        usage = self.adapter._session_usage
        token_limit = get_model_token_limit()
        last_input = meta.get("last_input_tokens") or 0
        percent = f" ({last_input / token_limit * 100:.1f}%)" if token_limit else ""
        missing = missing_api_keys(get_model())
        connected = list(self.mcp_client.servers)
        mode = self.permissions.mode or self.config.permission_mode
        rows = [
            ("Session", self.session_id + (f'  "{meta["title"]}"' if meta.get("title") else "") + ("" if meta else "  (not saved until the first message)")),
            ("Directory", os.getcwd()),
            ("Model", get_model() + (f"  ⚠ missing {', '.join(missing)}" if missing else "")),
            ("Permissions", mode + (f"  ({len(self.permissions.approved)} 'always' approval(s) this session)" if self.permissions.approved else "")),
            ("Tokens", f"↑{self._fmt_tokens(usage.get('prompt_tokens', 0))} ↓{self._fmt_tokens(usage.get('completion_tokens', 0))}"
                       + (f"  ({self._fmt_tokens(usage['cached_tokens'])} cached)" if usage.get("cached_tokens") else "")),
            ("Context", f"{self._fmt_tokens(last_input)} / {self._fmt_tokens(token_limit)}{percent}"),
            ("Steps", str(len(self.steps_history))),
            ("MCP", f"{len(connected)}/{len(self._mcp_configured)} connected" + (f": {', '.join(connected)}" if connected else "")),
            ("Version", __version__),
        ]
        width = max(len(k) for k, _ in rows) + 2
        self._append("\n".join(f"  {k.ljust(width)}{v}" for k, v in rows), msg_type="system")

    async def _rename_session(self, title: str) -> None:
        try:
            await self.store.set_title(self.session_id, title)
        except Exception as e:
            self._append(f"Rename failed: {e}", msg_type="error")
            return
        self._append(f"Session renamed to: {title}", msg_type="system")

    async def _export_session(self, arg: str) -> None:
        path = Path(arg).expanduser() if arg else Path.cwd() / f"dostuff-session-{self.session_id}.md"
        try:
            steps = await self.store.load(self.session_id)
            if not steps:
                self._append("Nothing to export yet.", msg_type="system")
                return
            meta = await self.store.get_session_meta(self.session_id)
            markdown = session_to_markdown(steps, meta, self.session_id)
            await asyncio.to_thread(path.write_text, markdown, encoding="utf-8")
        except Exception as e:
            self._append(f"Export failed: {e}", msg_type="error")
            return
        self._append(f"Exported {len(steps)} step(s) to {_rel_path(str(path))}", msg_type="system")

    async def _show_memories(self) -> None:
        try:
            facts = await self.memory_store.list_all(self.user_id)
        except Exception as e:
            self._append(f"Could not load memories: {e}", msg_type="error")
            return
        if not facts:
            self._append("No saved memories.", msg_type="system")
            return
        lines = [f"- {f['key']}: {f['value']}  ({f['source']})" for f in facts]
        self._append(f"Saved memories ({len(facts)}):\n" + "\n".join(lines), msg_type="system")

    def on_enter_submits_submitted(self, event: EnterSubmits.Submitted) -> None:
        text = event.text
        # A pending approval prompt consumes the input (empty Enter after "No" = deny without feedback)
        if self.adapter.has_pending_confirm():
            self.adapter.got_input(text)
            self._clear_input()
            return
        if not text:
            return

        self._clear_input()
        cmd = text.strip().lower()

        if text.lstrip().startswith("!"):
            command = text.lstrip()[1:].strip()
            if not command:
                return
            if self._turn_in_progress:
                self._append("Can't run a shell command while a turn is running (press ESC to cancel it first).", msg_type="error")
                return
            self._start_worker(self._run_shell_command(command), name="shell")
            return
        if cmd == "/exit":
            self.run_worker(self.action_quit(), name="exit_worker", exit_on_error=False)
            return
        if cmd in ("/clear", "/compact") and self._turn_in_progress:
            self._append(f"Can't run {cmd} while a turn is running (press ESC to cancel it first).", msg_type="error")
            return
        if cmd == "/clear":
            self._start_worker(self._clear_session(), name="clear")
            return
        if cmd == "/compact":
            self._start_worker(self._compact_session(), name="compact")
            return
        if cmd == "/memories":
            self._start_worker(self._show_memories(), name="memories")
            return
        name, _, arg = text.strip().partition(" ")
        name, arg = name.lower(), arg.strip()
        if name in ("/resume", "/model") and self._turn_in_progress:
            self._append(f"Can't run {name} while a turn is running (press ESC to cancel it first).", msg_type="error")
            return
        if name == "/resume":
            self._start_worker(self._resume_command(arg), name="resume")
            return
        if name == "/model":
            self._model_command(arg)
            return
        if name == "/permissions":
            self._permissions_command(arg)
            return
        if name == "/status" and not arg:
            self._start_worker(self._show_status(), name="status")
            return
        if name == "/rename":
            if not arg:
                self._append("Usage: /rename <title>", msg_type="error")
            else:
                self._start_worker(self._rename_session(arg), name="rename")
            return
        if name == "/export":
            self._start_worker(self._export_session(arg), name="export")
            return
        if cmd.split()[:1] == ["/undo"]:
            args = cmd.split()[1:]
            if args not in ([], ["force"]):
                self._append("Usage: /undo [force]", msg_type="error")
            elif self._turn_in_progress:
                self._append("Can't run /undo while a turn is running (press ESC to cancel it first).", msg_type="error")
            else:
                self._start_worker(self._undo_changes(force=bool(args)), name="undo")
            return
        if cmd == "/help":
            width = max(len(c) for c in COMMAND_DESCRIPTIONS) + 4
            commands = "\n".join(f"  {c.ljust(width)}{desc}" for c, desc in COMMAND_DESCRIPTIONS.items())
            self._append(
                f"Commands:\n{commands}\n\n"
                "  !<command>   Run a shell command directly; its output is added to the conversation\n\n"
                "Shortcuts: ESC cancel turn · Ctrl+Q quit · Ctrl+J newline · / command menu",
                msg_type="system",
            )
            return
        # A lone "/word" is a mistyped command; paths like /etc/hosts still go to the model
        if cmd.startswith("/") and " " not in cmd and cmd.count("/") == 1 and cmd not in COMMAND_DESCRIPTIONS:
            self._append(f"Unknown command {cmd}. Type / to see commands.", msg_type="error")
            return

        # User message — only append if starting a new turn; queued messages append when they run
        if not self._turn_in_progress:
            self._append(text, msg_type="user")
            self._begin_turn(text)
        else:
            self._prompt_queue.append(text)
            self._update_status(loader="Queued")
            self._append(f" Queued — will run after current turn (position {len(self._prompt_queue)})", msg_type="system")

    # ── Approval prompt ────────────────────────────────────────────────────────

    def _show_permission_prompt(self, question: str, options: list[tuple[str, str]]) -> None:
        """Swap the input for the approval selector until the prompt is answered."""
        self._hide_permission_prompt(restore_input=False)
        menu = self.query_one("#command-menu", CommandMenu)
        menu.display = False
        self.query_one("#input", EnterSubmits).display = False
        prompt = PermissionPrompt(question, options, id="permission-prompt")
        self.mount(prompt, before=menu)
        self.query_one("#messages", VerticalScroll).anchor()  # show the details being approved
        self.call_after_refresh(prompt.focus)

    def _ask_denial_feedback(self) -> None:
        """After "No": the input (yellow border) takes an optional note for the model."""
        self._hide_permission_prompt(restore_input=False)
        inp = self.query_one("#input", EnterSubmits)
        self._draft_before_feedback = inp.text
        inp.text = ""
        if hasattr(inp, "placeholder"):
            inp.placeholder = "Tell the model what to do instead (Enter to send, empty Enter to just deny)"
        inp.add_class("feedback-mode")
        inp.display = True
        inp.focus()

    def _hide_permission_prompt(self, restore_input: bool = True) -> None:
        for widget in self.query("#permission-prompt"):
            widget.remove()
        if not restore_input:
            return
        try:
            inp = self.query_one("#input", EnterSubmits)
        except Exception:
            return  # shutting down
        if inp.has_class("feedback-mode"):
            inp.remove_class("feedback-mode")
            if hasattr(inp, "placeholder"):
                inp.placeholder = ""
            inp.text = self._draft_before_feedback
            self._draft_before_feedback = ""
        inp.display = True
        inp.focus()

    def on_permission_prompt_chosen(self, event: PermissionPrompt.Chosen) -> None:
        if event.source_id == "picker":
            callback = self._picker_callback
            self._close_picker()
            if callback is not None:
                self._start_worker(callback(event.value), name="picker")
            return
        self.adapter.choose(event.value)

    def _show_picker(self, question: str, options: list[tuple[str, str]], on_choose: Callable[[str], Awaitable[None]],
                     cursor: int = 0, filterable: bool = False) -> None:
        """A selector in place of the input (/resume, /model, /permissions); ESC closes it."""
        self._close_picker()
        self._picker_callback = on_choose
        menu = self.query_one("#command-menu", CommandMenu)
        menu.display = False
        self.query_one("#input", EnterSubmits).display = False
        hint = ("Type to filter · " if filterable else "") + "↑/↓ select · Enter confirm · Esc close"
        picker = PermissionPrompt(question, options, hint=hint, cursor=cursor, filterable=filterable, id="picker")
        self.mount(picker, before=menu)
        self.call_after_refresh(picker.focus)

    def _picker_open(self) -> bool:
        return bool(self.query("#picker"))

    def _close_picker(self) -> None:
        self._picker_callback = None
        if not self._picker_open():
            return
        for widget in self.query("#picker"):
            widget.remove()
        if self.query("#permission-prompt"):
            return  # an approval prompt owns the input area
        try:
            inp = self.query_one("#input", EnterSubmits)
        except Exception:
            return
        inp.display = True
        inp.focus()

    def _begin_turn(self, text: str) -> None:
        self._turn_in_progress = True
        self._start_worker(self._run_turn(text), name=f"turn_{uuid.uuid4().hex[:6]}")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Open the "/" menu while typing a command; tint the input border in "!" shell mode."""
        text = event.text_area.text
        menu = self.query_one("#command-menu", CommandMenu)
        if text.startswith("/") and " " not in text and "\n" not in text:
            menu.show_matches(text)
        else:
            menu.display = False
        event.text_area.set_class(text.lstrip().startswith("!"), "shell-mode")

    # ── `!command`: user-run shell command ─────────────────────────────────────

    async def _run_shell_command(self, command: str) -> None:
        """Run a shell command typed as `!command` and record it in the conversation.

        It counts as a turn: prompts sent meanwhile are queued, and ESC kills the process.
        """
        self._turn_in_progress = True
        self._append(f"! {command}", msg_type="shell")
        loader_widget = self._show_loader("Running... (ESC to cancel)")
        self._turn_start_time = time.time()
        self._start_timer()
        self.adapter._cancelled = False
        self.adapter._cancel_event.clear()
        try:
            task = asyncio.ensure_future(run_shell(command, timeout=SHELL_TIMEOUT_S))
            while not task.done():
                if self.adapter._cancel_event.is_set():
                    task.cancel()
                    break
                await asyncio.wait({task}, timeout=0.1)
            if task.cancelled():
                self.adapter._cancelled = False
                self.adapter._cancel_event.clear()
                self._append("  ⎿  Cancelled (process killed)", msg_type="shell-output")
                return
            try:
                code, stdout, stderr, note = task.result()
            except Exception as e:
                self._append(f"Shell error: {e}", msg_type="error")
                return
            output = "\n".join(part.rstrip("\n") for part in (stdout, stderr) if part.strip())
            if code not in (0, None):
                output += f"\n(exit {code})"
            if note:
                output += f"\n({note})"
            output = output.strip("\n") or "(no output)"
            self._append(_tool_result_line(output, SHELL_PREVIEW_LINES), msg_type="shell-output")

            content = _format_shell_step(command, truncate_tool_output(output, self.config.max_tool_output_chars))
            working_history = await self.store.load_working_history(self.session_id)
            await append_step(
                {"role": "user", "content": content},
                self.steps_history,
                working_history,
                self.current_session_history,
                self.session_id,
                self.store,
                "interactive_loop",
            )
        except Exception as e:
            self._append(f"Shell error: {e}", msg_type="error")
        finally:
            self._stop_timer()
            elapsed = time.time() - self._turn_start_time if self._turn_start_time is not None else 0.0
            self._turn_start_time = None
            self._turn_in_progress = False
            self._stop_loader(loader_widget)
            self._update_status(loader=f" {elapsed:.1f}s" if elapsed > 0 else "")
            self._process_prompt_queue()

    # ── One agent turn ─────────────────────────────────────────────────────────

    async def _build_instructions(self, user_text: str) -> str:
        return await build_turn_instructions(self._system_instructions, user_text, self.memory_store, self.episodic_store, self.user_id)

    async def _run_turn(self, user_text: str) -> None:
        loader_widget: Static | None = None
        try:
            turn_id = uuid.uuid4().hex
            turn_id_var.set(turn_id)
            session_id_var.set(self.session_id)  # /clear may have switched sessions since on_mount

            meta = await self.store.get_session_meta(self.session_id) or {}
            working_history = await self.store.load_working_history(self.session_id)

            await append_step(
                {"role": "user", "content": user_text},
                self.steps_history,
                working_history,
                self.current_session_history,
                self.session_id,
                self.store,
                "interactive_loop",
            )

            dynamic_instructions = await self._build_instructions(user_text)

            loader_widget = self._show_loader("Working... (ESC to cancel)")
            self._turn_start_time = time.time()
            self._start_timer()
            # Reset cancel state for new turn
            self.adapter._cancelled = False
            self.adapter._cancel_event.clear()
            self._last_streamed_answer = ""

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
                compaction_notes=meta.get("compaction_notes") or "",
                session_prompt_tokens=meta.get("prompt_tokens") or 0,
                session_completion_tokens=meta.get("completion_tokens") or 0,
                session_total_tokens=meta.get("total_tokens") or 0,
                last_input_tokens=meta.get("last_input_tokens") or 0,
                stream=True,
                max_iterations=self.max_iterations,
                permissions=self.permissions,
            )

            agent_text = result or "(no response)"
            # A streamed final answer is already on screen
            if agent_text.strip() != self._last_streamed_answer.strip():
                self._append(agent_text, msg_type="agent")

        except Exception as e:
            self._append(f"Turn error: {e}", msg_type="error")
        finally:
            self._stop_timer()
            final_elapsed = time.time() - self._turn_start_time if self._turn_start_time is not None else 0.0
            self._turn_start_time = None
            self._turn_in_progress = False
            self._finish_stream_widgets()
            self._stop_loader(loader_widget)
            self._update_status(loader=f" {final_elapsed:.1f}s" if final_elapsed > 0 else "")
            self._process_prompt_queue()

    def _process_prompt_queue(self) -> None:
        """Start the next queued prompt, if any."""
        if not self._prompt_queue or self._quitting:
            return
        next_text = self._prompt_queue.pop(0)
        self._append(next_text, msg_type="user")
        if self._prompt_queue:
            self._append(f" {len(self._prompt_queue)} queued", msg_type="system")
        self._begin_turn(next_text)

    # ── Loader / timer / streaming helpers ─────────────────────────────────────

    def _show_loader(self, label: str = "Working...") -> Static:
        """Show (or update) loader message in conversation. Returns the Static widget."""
        existing = self._loader_widget
        if existing is not None and existing.parent is not None:
            existing.styles.display = "block"
            existing.update(label)
            return existing
        loader_widget = Static(label, classes="msg msg-system", markup=False)
        self.query_one("#msg-container", Vertical).mount(loader_widget)
        self._loader_widget = loader_widget
        return loader_widget

    def _stop_loader(self, loader_widget: Static | None) -> None:
        """Remove the loader widget from the conversation stream."""
        if loader_widget is None:
            return
        try:
            loader_widget.remove()
        except Exception:
            pass
        if self._loader_widget is loader_widget:
            self._loader_widget = None

    def _start_timer(self) -> None:
        self._stop_timer()
        self._timer = self.set_interval(0.1, self._tick)

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        if self._turn_start_time is not None and not self.adapter._cancelled:
            self._update_status(loader=f" {time.time() - self._turn_start_time:.1f}s")

    def _update_stream_widget(self, kind: str, text: str) -> None:
        """Update (or create) the live widget for this model call's answer or reasoning text."""
        self._stream_texts[kind] = text
        widget = self._stream_widgets.get(kind)
        if widget is None or widget.parent is None:
            css = "msg msg-thought" if kind == "thought" else "msg msg-agent"
            widget = Static("" if kind == "thought" else text, classes=css, markup=False)
            self._stream_widgets[kind] = widget
            if kind == "thought":
                self._thought_shown = 0
            self._mount_message(widget)
        elif kind == "answer":
            widget.update(text)
        if kind == "thought":
            self._thought_step = max(1, (len(text) - self._thought_shown) // THOUGHT_REVEAL_TICKS)
            if self._thought_timer is None:
                self._thought_timer = self.set_interval(THOUGHT_REVEAL_INTERVAL_S, self._reveal_thought)

    def _reveal_thought(self) -> None:
        """Timer tick: show the next slice of the live thought text."""
        target = self._stream_texts.get("thought", "")
        widget = self._stream_widgets.get("thought")
        if widget is None or widget.parent is None or self._thought_shown >= len(target):
            self._stop_thought_reveal()
            return
        self._thought_shown = min(len(target), self._thought_shown + self._thought_step)
        widget.update(_thought_text(target[:self._thought_shown]))
        if self._thought_shown >= len(target):
            self._stop_thought_reveal()

    def _stop_thought_reveal(self) -> None:
        if self._thought_timer is not None:
            self._thought_timer.stop()
            self._thought_timer = None

    def _finish_stream_widgets(self, interrupted: bool = False) -> None:
        """End of one model call: render the answer as markdown and detach the live widgets."""
        self._stop_thought_reveal()
        thought_widget = self._stream_widgets.get("thought")
        if thought_widget is not None and thought_widget.parent is not None:
            thought_widget.update(_thought_text(self._stream_texts.get("thought", "")))
        answer_widget = self._stream_widgets.get("answer")
        answer_text = self._stream_texts.get("answer", "")
        if answer_widget is not None and answer_widget.parent is not None:
            if interrupted:
                answer_widget.update(answer_text + "\n[interrupted — retrying]")
            elif answer_text:
                answer_widget.update(RichMarkdown(answer_text))
                self._last_streamed_answer = answer_text
        self._stream_widgets.clear()
        self._stream_texts.clear()

    async def action_cancel(self) -> None:
        """Cancel current agent turn (ESC); a pending approval prompt resolves as "deny".

        If the "/" menu is open, ESC only closes it.
        """
        menu = self.query_one("#command-menu", CommandMenu)
        if menu.display:
            menu.display = False
            return
        if self._picker_open():
            self._close_picker()
            return
        if not self._turn_in_progress and not self.adapter.has_pending_confirm():
            return  # nothing to cancel; a stale flag would cancel the next turn's first step
        self.adapter._cancelled = True
        self.adapter._cancel_event.set()
        self.adapter.deny_pending_confirm()
        self._update_status(loader=" Cancelled")

    # ── Exit ───────────────────────────────────────────────────────────────────

    async def action_quit(self) -> None:
        """Ctrl+Q / /exit: clean up and quit; repeated presses are ignored."""
        if self._quitting:
            return
        self._quitting = True
        self.run_worker(self._run_cleanup_and_exit(), name="cleanup_worker", exit_on_error=False)

    async def _run_cleanup_and_exit(self) -> None:
        self._append("Saving session...")
        try:
            await self._cleanup_and_save_session()
        except Exception as e:
            self._append(f"Cleanup error: {e}", msg_type="error")
        finally:
            self.exit()

    async def _cleanup_and_save_session(self) -> None:
        """Cancel pending work, close MCP sessions and the DB, with bounded timeouts."""
        # 1. Cancel in-flight workers (turns, MCP connect, ...); they run on this loop
        self.adapter._cancelled = True
        self.adapter._cancel_event.set()
        self.adapter.deny_pending_confirm()
        workers = [w for w in self._pending_workers if not w.is_finished]
        for w in workers:
            w.cancel()
        self._pending_workers.clear()
        if workers:
            await asyncio.sleep(0.2)
        # 2. MCP sessions (same loop they were opened on)
        try:
            await asyncio.wait_for(self.mcp_client.cleanup(), timeout=5.0)
        except asyncio.TimeoutError:
            self._append("MCP shutdown timed out.", msg_type="error")
        # 3. Session DB
        await self.store.close()
        # 4. Force-stop any lingering litellm logging workers (background threads)
        try:
            import litellm
            worker = getattr(litellm, "logging_worker", None)
            if worker is not None and hasattr(worker, "stop"):
                worker.stop()
        except Exception:
            pass
        self._append(f"Session saved. Goodbye! (cancelled {len(workers)} pending task(s))")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _mount_message(self, widget: Static) -> None:
        """Mount above the loader (so the loader stays at the bottom)."""
        msg_container = self.query_one("#msg-container", Vertical)
        loader = self._loader_widget
        if loader is not None and loader.is_mounted and loader.parent is msg_container:
            msg_container.mount(widget, before=loader)
        else:
            msg_container.mount(widget)

    def _append(self, line: str, msg_type: str = "agent") -> None:
        """Add a message widget; msg_type picks its CSS class (text color, spacing)."""
        known = ("agent", "user", "confirm", "system", "error", "tool", "tool-result", "diff", "shell", "shell-output")
        css_class = f"msg msg-{msg_type if msg_type in known else 'agent'}"
        if msg_type in ("user", "shell"):
            # Sending a message jumps back to the bottom even if the user had scrolled up
            self.query_one("#messages", VerticalScroll).anchor()
        if msg_type == "user":
            msg = Static(Text("> " + line.replace("\n", "\n  ")), classes=css_class, markup=False)
        elif msg_type == "agent" and line:
            msg = Static(RichMarkdown(line), classes=css_class, markup=False)
        elif "\033[" in line:
            msg = Static(Text.from_ansi(line), classes=css_class, markup=False)
        else:
            msg = Static(line, classes=css_class, markup=False)
        self._mount_message(msg)

    async def _connect_one_mcp_server(self, name: str, cfg: dict) -> None:
        timeout = float(cfg.get("connect_timeout") or MCP_CONNECT_TIMEOUT_S)
        try:
            await asyncio.wait_for(
                self.mcp_client.connect_to_server(
                    server_name=name,
                    transport=cfg.get("transport", "stdio"),
                    command=cfg.get("command"),
                    args=cfg.get("args", []),
                    env=cfg.get("env"),
                    url=cfg.get("url"),
                    headers=cfg.get("headers"),
                ),
                timeout=timeout,
            )
            self._append(f"{name}", msg_type="system")
        except asyncio.TimeoutError:
            self._append(f"{name}: connection timed out after {timeout:.0f}s", msg_type="error")
        except Exception as e:
            self._append(f"{name}: {e}", msg_type="error")

    async def _connect_mcp_servers(self, mcp_servers: dict) -> None:
        """Connect every configured server concurrently, each with its own timeout."""
        await asyncio.gather(*(self._connect_one_mcp_server(name, cfg) for name, cfg in mcp_servers.items()))
        self._append(f"MCP ready ({len(self.mcp_client.servers)} server(s)).", msg_type="system")

    def _update_status(self, loader: str = "", context_window_percent: str = "") -> None:
        """Status bar: cwd, session, model, permission mode, session tokens, context %, timer/queue."""
        queue_text = f" {len(self._prompt_queue)} queued" if self._prompt_queue else ""
        loader = f"{loader}  {queue_text}" if loader and queue_text else (loader or queue_text)
        cwd_display = os.getcwd().replace(str(Path.home()), "~")
        cwd_display = cwd_display if len(cwd_display) < 36 else "..." + cwd_display[-33:]
        token_str = self.adapter._token_display or " ↑0 ↓0"
        if context_window_percent:
            self.adapter._last_context_window_percent = context_window_percent
        display_percent = self.adapter._last_context_window_percent
        mode = self.permissions.mode or self.config.permission_mode
        status = Text(f"/help {cwd_display}  {self.session_id}  {get_model()}  ")
        status.append(f"{mode} mode", style="bold #e06c75" if mode == "auto" else "")
        status.append(f"  {token_str}  {display_percent}  {loader or '0.0s'} ")
        try:
            self.query_one("#status", Static).update(status)
        except Exception:
            pass  # not mounted yet / shutting down

    def _fmt_tokens(self, n: int) -> str:
        if n < 1000:
            return str(n)
        if n < 1_000_000:
            return f"{n/1000:.1f}k"
        return f"{n/1_000_000:.1f}M"

    def _clear_input(self) -> None:
        try:
            ta = self.query_one("#input", EnterSubmits)
            ta.text = ""
        except Exception:
            pass
