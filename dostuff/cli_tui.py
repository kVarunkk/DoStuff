"""Real TUI for dostuff — full agent experience via textual, not skeleton.

Every turn, /compact, /clear and exit cleanup runs as an *async* worker on Textual's event
loop, so widgets are only touched from the UI thread and MCP sessions live on one loop.
Tools that run in threads (via asyncio.to_thread) reach the UI through TuiAdapter.emit,
which hops back to the UI thread when needed.
"""
import asyncio
import contextlib
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Iterable, Literal
from textual.app import App, ComposeResult
from textual.geometry import Region
from textual.message import Message
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.actions import SkipAction
from textual.binding import Binding
from textual.worker import Worker, WorkerState
from rich.console import Group
from rich.style import Style
from rich.text import Text
from rich.markdown import Markdown as RichMarkdown
from textual.events import Click, Key, Paste, TextSelected
from dostuff import __version__
from dostuff.config import NOTIFY_KINDS, STATUS_BAR_ITEMS, get_config, parse_compact_threshold
from dostuff.agent.permissions import PERMISSION_MODES, ApprovalReply, SessionPermissions, next_mode, parse_answer
from dostuff.agent.instructions import build_base_instructions, build_turn_instructions
from dostuff.helpers.agent.constants import COMMAND_DESCRIPTIONS, COMMAND_FLAGS
from dostuff.helpers.memory.export_session import session_to_markdown
from dostuff.helpers.memory.shell_step import format_shell_step as _format_shell_step, parse_shell_step as _parse_shell_step
from dostuff.tools.bash.bash_command import detached_job_id, run_shell
from dostuff.tools.bash.jobs import (
    KILL_WAIT_S, Job, JobError, JobEvent, find_orphans, fmt_runtime, get_registry, notice_line,
    prune_logs,
    session_of,
)
from dostuff.lib.memory.session_store import SQLiteSessionStore
from dostuff.lib.checkpoints import UndoResult, get_checkpoint_store
from dostuff.lib.worktree import Worktree, WorktreeError, create_worktree, detect_worktree, has_changes, remove_worktree, worktree_changes
from dostuff.lib.model import get_model
from dostuff.lib.tracing import session_id_var, turn_id_var
from dostuff.helpers.agent.append_step import append_step
from dostuff.helpers.agent.loop_check_cancelled import CANCELLED_TURN
from dostuff.helpers.agent.file_mentions import (PARTIAL_MENTION_RE, attached_paths, expand_file_mentions,
                                                 format_mention, mention_candidates, strip_attachments)
from dostuff.helpers.agent.model_cost import format_cost
from dostuff.helpers.agent.notices import WAKE_LIMIT, agent_notice_line, fmt_agent_runtime, format_notices, get_notice_queue, parse_notices
from dostuff.helpers.agent.pastes import PasteStore, sweep_old_pastes
from dostuff.helpers.agent.steering import SteerMessage, SteeringQueue, unframe_steered
from dostuff.helpers.agent.side_question import (
    BTW_FORK_DESCRIPTION, BTW_FORK_PROMPT, add_exchange, answer_side_question, discard_snapshot, kept_prompt,
    plan_mode_on, snapshot as live_side_snapshot,
)
from dostuff.agent.subagents.fork import ForkContext
from dostuff.helpers.agent.quick_memory import append_memory, memory_file, quick_memory_text
from dostuff.helpers.agent.context_usage import context_parts, count_tokens, render_context
from dostuff.helpers.ui.doctor import doctor_checks, hooks_listing
from dostuff.helpers.ui.tools_listing import tools_listing
from dostuff.helpers.ui.ask_answers import one_line as _one_line_answer, parse_answers
from dostuff.helpers.agent.slash_suggestion import slash_arg_hint, slash_suggestion
from dostuff.helpers.agent.tool_output import EMPTY_RESPONSE_NUDGE, format_call_label, truncate_tool_output
from dostuff.helpers.hooks.dispatch import hook_line
from dostuff.helpers.hooks.events import blocked_prompt_message, parse_stop_note, session_start, user_prompt_submit, with_prompt_context
from dostuff.helpers.mcp.load_mcp_config import load_mcp_config, mcp_config_sources, read_mcp_file
from dostuff.helpers.skills.custom_commands import discover_commands
from dostuff.helpers.skills.skill_command import parse_skill_command, skill_commands, skill_prompt
from dostuff.setup_wizard import PROVIDERS, missing_api_keys, read_template
from dostuff.helpers.models.add_model_entry import ModelConfigError, add_model_entry
from dostuff.helpers.models.list_models import FETCHED_PROVIDERS, ModelListError, provider_models
from dostuff.helpers.ui.emit import set_adapter
from dostuff.helpers.ui.prompt_history import append_history, load_history
from dostuff.helpers.ui.terminal_signals import (
    PUSH_TITLE, TitleState, format_title, notification_sequence, restore_title_sequence, should_notify, title_sequence,
)
from dostuff.tools.settings import (Scope, SettingsKind, mcp_path, new_settings_text, save_settings_file,
                                    settings_file)

if TYPE_CHECKING:
    from dostuff.agent.subagents.registry import AgentEvent, AgentRun
    from dostuff.lib.mcp.mcp_client import MCPClient
    from dostuff.lib.memory.episodic_memory_store import EpisodicMemoryStore
    from dostuff.lib.memory.semantic_memory_store import SemanticMemoryStore

MCP_CONNECT_TIMEOUT_S = 150.0  # covers the 120 s OAuth browser callback
SHELL_TIMEOUT_S = 110.0  # `!command` typed by the user
WORKING_LABEL = "Working... (ESC to cancel)"
LOADING_LIBRARIES_LABEL = "Loading model libraries..."
WORKING_LABEL_JOBS = "Working... (ESC to cancel · Ctrl+B to background)"  # a model shell call or subagent is running
SETTINGS_WATCH_INTERVAL_S = 1.0  # config.yaml / mcp_config.json edited by hand: picked up within this
JOB_WAKE_DELAY_S = 1.0  # batch jobs that end together into one wake-up turn; also the retry interval while busy
PROMPT_RETRY_S = 0.2  # how often a background agent's held-back approval prompt checks again (subagents D8)
JOB_WAKE_IDLE_WORKERS = frozenset({"jobs_housekeeping", "load_backend", "mcp_connect", "mcp_sync"})  # don't block a wake
TOOL_RESULT_PREVIEW_LINES = 8  # tool output shown in the conversation; the model gets all of it
SHELL_PREVIEW_LINES = 40
JOB_DETAIL_LINES = 8  # output rows in a job's picker (from the jobs list)
QUEUE_BAR_MAX_ROWS = 4  # queued prompts listed above the input; /queue lists them all
QUEUE_DETAIL_LINES = 8  # rows of a queued prompt shown in its picker
PREVIEW_CHARS_PER_LINE = 200  # preview char budget = lines * this, so one giant line can't flood the screen
PICKER_MAX_SESSIONS = 9  # /resume picker: number keys reach every option
PICKER_MAX_VISIBLE = 10  # longer option lists scroll with the cursor
EDIT_PROMPT_ESC_SECONDS = 1.5  # Esc, then Esc again within this, takes back the last prompt
QUIT_HINT = "ctrl+q or /exit to quit"  # flashed when Esc / Ctrl+C has nothing to do
HISTORY_SEARCH_MAX = 500  # Ctrl+R: newest prompts offered
SKILL_MENU_DESCRIPTION_CHARS = 100  # skill descriptions can be paragraphs; the "/" menu shows the start
TODO_RESULT_PREFIX = "todo_write: "  # loop() emits results as "<tool>: <result>"
ASK_RESULT_PREFIX = "ask_user: "  # its result is a few short "question → answer" lines, shown in full
ASK_SUBMIT_LABEL = "Submit answers"
ASK_HINT = "↑/↓ select · Enter confirm · ←/→ change question · Esc cancel"
ASK_INERT_HINT = "answer every question first"
ASK_OTHER_LABEL = "Other (type your own)"
ASK_EMPTY_INPUT_HINT = "type an answer (Esc cancels the questions)"
# The current question's tab: bold orange, plus the solid bar `_tabs` puts in front of it. No background:
# under the ansi-dark theme a hex `on` colour snaps to black, which is invisible on a dark terminal.
ASK_CURRENT_TAB = "bold #e5c07b"
# Gemini sends reasoning as whole summary paragraphs ~1 s apart; each new chunk is typed out
# over THOUGHT_REVEAL_TICKS ticks so it reads like streaming instead of popping in as a block
THOUGHT_REVEAL_INTERVAL_S = 0.03
THOUGHT_REVEAL_TICKS = 25


def _mcp_files_signature() -> tuple[tuple[int, int] | None, ...]:
    """(mtime, size) of each MCP config file; None for a missing one. A change means reconnect."""
    signature: list[tuple[int, int] | None] = []
    for path in mcp_config_sources():
        try:
            st = path.stat()
            signature.append((st.st_mtime_ns, st.st_size))
        except OSError:
            signature.append(None)
    return tuple(signature)


# litellm, chromadb and mcp take several seconds to import, so the TUI draws its first frame
# without them and loads them in a thread (DostuffTUI._load_backend). These wrappers import on
# first use; tests patch them by name.

def _import_backend() -> None:
    import dostuff.agent.loop  # noqa: F401  (litellm)
    import dostuff.helpers.agent.manage_context  # noqa: F401
    import dostuff.lib.mcp.mcp_client  # noqa: F401  (mcp)
    from dostuff.lib.memory.chroma import chroma_available
    if chroma_available():
        import chromadb  # noqa: F401  (optional: the `memory` extra)


async def agent_loop(*args: Any, **kwargs: Any) -> Any:
    from dostuff.agent.loop import loop
    return await loop(*args, **kwargs)


def get_model_token_limit() -> int:
    from dostuff.helpers.agent.get_model_token_limit import get_model_token_limit as limit
    return limit()


async def compact_context(*args: Any, **kwargs: Any) -> Any:
    from dostuff.helpers.agent.manage_context import compact_context as compact
    return await compact(*args, **kwargs)


def _preview_clipped(text: str, max_lines: int) -> bool:
    """Whether _preview hides anything: too many lines, or too many chars (one huge JSON line)."""
    lines = text.rstrip("\n").splitlines()
    return len(lines) > max_lines or sum(map(len, lines)) > max_lines * PREVIEW_CHARS_PER_LINE


def _preview(text: str, max_lines: int) -> str:
    """First max_lines lines and at most max_lines * PREVIEW_CHARS_PER_LINE chars, with a marker for the rest."""
    lines = text.rstrip("\n").splitlines() or [""]
    budget = max_lines * PREVIEW_CHARS_PER_LINE
    shown: list[str] = []
    for line in lines[:max_lines]:
        if len(line) > budget:  # a long line (minified JSON, base64…) would wrap over the whole screen
            shown.append(line[:budget])
            hidden = len(line) - budget + sum(map(len, lines[len(shown):]))
            return "\n".join(shown + [f"… +{hidden} chars (ctrl+o to expand)"])
        shown.append(line)
        budget -= len(line)
    extra = len(lines) - max_lines
    if extra > 0:
        shown.append(f"… +{extra} lines (ctrl+o to expand)")
    return "\n".join(shown)


def _tool_result_line(text: str, max_lines: int = TOOL_RESULT_PREVIEW_LINES) -> str:
    """Tool output under its call, Claude Code style: '  ⎿  first line' + indented rest."""
    return "  ⎿  " + _preview(text, max_lines).replace("\n", "\n     ")


class SelectableStatic(Static):
    """Static whose Rich renderables (Markdown) are flattened to Text at the current width.

    Textual can only select Text / Content, so a plain Static holding RichMarkdown can't be
    selected with the mouse; this keeps the Markdown look and makes it selectable.
    """

    _flat_cache: tuple[object, int, Text] | None = None

    def _flat(self, width: int) -> Text | None:
        content = self.content
        if not isinstance(content, RichMarkdown) or width <= 0:
            return None
        cached = self._flat_cache
        if cached is not None and cached[0] is content and cached[1] == width:
            return cached[2]
        console = self.app.console
        rows = []
        for line in console.render_lines(content, console.options.update_width(width), pad=False):
            row = Text()
            for seg in line:
                if not seg.control:
                    row.append(seg.text, seg.style)
            row.rstrip()  # trailing padding: Textual would re-wrap lines that already fit
            rows.append(row)
        text = Text("\n").join(rows)
        self._flat_cache = (content, width, text)
        return text

    def render(self) -> Any:
        flat = self._flat(self.content_size.width)
        return flat if flat is not None else super().render()

    def get_content_height(self, container: Any, viewport: Any, width: int) -> int:
        flat = self._flat(width)
        if flat is not None:
            return len(flat.plain.split("\n"))
        return super().get_content_height(container, viewport, width)


def _copy_to_clipboard(app: App, text: str) -> None:
    """OSC 52 (Textual) plus clip.exe on Windows, for consoles that ignore OSC 52."""
    app.copy_to_clipboard(text)
    if sys.platform != "win32":
        return

    def run_clip() -> None:
        try:
            subprocess.run(["clip"], input=text.encode("utf-16"), timeout=5,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
        except (OSError, subprocess.SubprocessError):
            pass
    threading.Thread(target=run_clip, daemon=True).start()


class ToolResult(Static):
    """A tool or shell result: its first max_lines lines, or all of it once expanded.

    A click toggles this one; Ctrl+O toggles every result (DostuffTUI.action_toggle_tool_results).
    """

    def __init__(self, text: str, max_lines: int, expanded: bool = False, **kwargs: Any) -> None:
        self.full_text = text
        self.max_lines = max_lines
        self.expanded = expanded and self.collapsible
        super().__init__(self._content(), markup=False, **kwargs)

    @property
    def collapsible(self) -> bool:
        return _preview_clipped(self.full_text, self.max_lines)

    def _content(self) -> Text | str:
        if self.expanded:
            line = _tool_result_line(self.full_text, max_lines=len(self.full_text)) + "\n     (ctrl+o to collapse)"
        else:
            line = _tool_result_line(self.full_text, self.max_lines)
        return Text.from_ansi(line) if "\033[" in line else line

    def set_expanded(self, expanded: bool) -> None:
        expanded = expanded and self.collapsible
        if expanded != self.expanded:
            self.expanded = expanded
            self.update(self._content())

    def on_click(self) -> None:
        if self.screen.get_selected_text():
            return  # the end of a mouse selection, not a click
        self.set_expanded(not self.expanded)


class Transcript(VerticalScroll):
    """The conversation viewport.

    Textual pins an anchored scrollbox in the same layout pass it arranges it (`_compositor`):
    `total_region.bottom - (container_size.height - scrollbar_size_horizontal)`. It reads
    `container_size`, which the compositor only writes back *after* the pass, so the first frame
    after the viewport changes height is pinned for the *previous* height. Swapping the input for
    the picker (or a taller approval prompt, or giving the rows back when it closes) changes
    `#messages`' height by itself, so that frame kept the old scroll offset and the transcript was
    drawn a few rows off — the bounce you see as the selectors come in and go out.

    `_get_scrollable_region` runs just before that arithmetic inside the pass, so setting the
    container size there makes the anchor use the size being laid out. Textual stores the same
    value afterwards, so nothing else changes.
    """

    def _get_scrollable_region(self, region: Region) -> Region:
        self._container_size = region.size
        return super()._get_scrollable_region(region)


class _JumpToBottomLabel(Static):
    """The pill inside `JumpToBottom`; a click goes back to the latest message."""

    ALLOW_SELECT = False  # a click with a little mouse movement would start a selection instead

    def on_click(self) -> None:
        messages = self.app.query_one("#messages", VerticalScroll)
        messages.anchor()  # scroll_end inside also re-attaches a released anchor
        if isinstance(self.parent, JumpToBottom):
            self.parent.display = False


class JumpToBottom(Horizontal):
    """Shown while the conversation is scrolled up.

    The widget itself is a full-width row so it can center a small pill (`#jump-bottom-label`)
    over the conversation instead of banding the whole width; only the pill is clickable."""

    def __init__(self, label: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._label = label

    def compose(self) -> ComposeResult:
        yield _JumpToBottomLabel(self._label, id="jump-bottom-label")


class JobsBar(Static):
    """Beside the status bar while background shells run; a click (or ↓ in an empty input) lists them,
    a second click closes the list."""

    ALLOW_SELECT = False  # a click with a little mouse movement would start a selection instead

    def on_click(self) -> None:
        self.app.action_toggle_jobs()  # type: ignore[attr-defined]


class AgentsBar(Static):
    """Beside the shells item while this session's subagents run: `⚙ 2 agents`, or `⚠ 1 agent waiting` while
    one waits for approval; a click opens /agents (subagents spec D14)."""

    ALLOW_SELECT = False

    def on_click(self) -> None:
        self.app.action_show_agents()  # type: ignore[attr-defined]


class QueueBar(Static):
    """Above the input while prompts are queued, one row each; a click (or /queue) lists them to
    edit, run next or remove, a second click closes the list."""

    ALLOW_SELECT = False

    def on_click(self) -> None:
        self.app.action_toggle_queue()  # type: ignore[attr-defined]


def _one_line(text: str, width: int) -> str:
    """`text` on one row: whitespace runs collapsed, cut to `width` with an ellipsis."""
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 1] + "…"


def _fmt_elapsed(seconds: float) -> str:
    """Whole seconds, Claude Code style: 45s, 1m 5s, 1h 2m."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {s % 3600 // 60}m"


def _agent_status(run: "AgentRun") -> str:
    return "waiting for approval" if run.status == "waiting_approval" else run.status


def _agent_label(run: "AgentRun") -> str:
    """One /agents row: id, status, runtime, description · type, tokens, cost, background or foreground."""
    from dostuff.helpers.agent.notices import fmt_agent_runtime
    runtime = fmt_agent_runtime((run.ended or time.time()) - run.started) if run.started else "—"
    return (f"{run.id}  {_agent_status(run)}  {runtime}  {run.description} · {run.agent_type}  "
            f"{run.tokens:,} tokens  ${run.cost_usd:.2f}  {'background' if run.background else 'foreground'}")


def _job_status(job: Job) -> str:
    """`running` / `exited (1)` / `killed (by user)` for /jobs."""
    if job.status == "running":
        return "running"
    return f"exited ({job.exit_code})" if job.status == "exited" else f"killed (by {job.killed_by})"


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
    """Suggestions shown above the input: slash commands while it starts with '/', project paths
    while an `@mention` is typed at the cursor (mode "file")."""

    can_focus = False
    mode = "command"
    commands: dict[str, str] = COMMAND_DESCRIPTIONS  # the app adds the skill commands

    def show_matches(self, prefix: str) -> bool:
        """List commands starting with prefix; returns False (and hides) if none match."""
        self.mode = "command"
        matches = [c for c in self.commands if c.startswith(prefix.lower())]
        self.clear_options()
        if not matches:
            self.display = False
            return False
        width = max(len(c) for c in self.commands) + 4
        self.add_options(
            Option(Text.assemble((c.ljust(width), "bold"), (self.commands[c], "dim")), id=c)
            for c in matches
        )
        self.highlighted = 0
        self.display = True
        return True

    def show_paths(self, paths: list[str]) -> bool:
        """List project paths for an @mention; returns False (and hides) if there are none."""
        self.mode = "file"
        self.clear_options()
        if not paths:
            self.display = False
            return False
        self.add_options(Option(Text(path, style="bold" if path.endswith("/") else ""), id=path) for path in paths)
        self.highlighted = 0
        self.display = True
        return True

    def selected_command(self) -> str | None:
        if not self.display or self.highlighted is None:
            return None
        return self.get_option_at_index(self.highlighted).id


def _option_window(count: int, cursor: int, visible: int = PICKER_MAX_VISIBLE) -> tuple[int, int]:
    """The first and last row shown for `count` rows with the cursor at `cursor`: it scrolls by half a
    screen, so the markers (`↑ N more` / `↓ N more`) appear the same way in every selector."""
    start = max(0, min(cursor - visible // 2, count - visible))
    return start, min(start + visible, count)


def _append_option_rows(text: Text, rows: list[str], cursor: int, start: int, end: int,
                        meta_key: str = "option", indent: str = "  ") -> None:
    """The visible rows of a selector, numbered from `start` (the label already carries its number).

    The rows share the scroll markers and the `❯` cursor mark, so the approval selector, the /resume
    picker and the ask-user panel can't drift apart; each row carries `meta={meta_key: i}` for clicks.
    """
    if start:
        text.append(f"\n{indent}↑ {start} more", style="dim")
    for i in range(start, end):
        row = Style(meta={meta_key: i})  # read back by on_click, wrapped lines included
        text.append("\n")
        if i == cursor:
            text.append(f"❯ {rows[i]}", style=Style(bold=True, color="#8ab4f8") + row)
        else:
            text.append(f"  {rows[i]}", style=row)
    if end < len(rows):
        text.append(f"\n{indent}↓ {len(rows) - end} more", style="dim")


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
    ALLOW_SELECT = False  # clicking an option picks it

    class Chosen(Message):
        def __init__(self, value: str, source_id: str = "") -> None:
            super().__init__()
            self.value = value
            self.source_id = source_id  # "permission-prompt" or "picker"

    def __init__(self, question: str, options: list[tuple[str, str]], hint: str = APPROVAL_HINT, cursor: int = 0,
                 filterable: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.configure(question, options, hint=hint, cursor=cursor, filterable=filterable)

    def configure(self, question: str, options: list[tuple[str, str]], hint: str = APPROVAL_HINT, cursor: int = 0,
                  filterable: bool = False, detail: list[tuple[str, str]] | None = None) -> None:
        """Show a new question; the app reuses one #picker rather than mounting another with the same id."""
        self._question = question
        self._detail = detail or []  # (line, style) between the question and the options, cut to one row each
        self._all_options = options  # (value, label)
        self._options = options  # the ones matching the filter
        self._hint = hint
        self._cursor = cursor
        self._chosen = False
        self._filterable = filterable
        self._filter = ""
        if self.is_mounted:
            self._render_options()

    def on_mount(self) -> None:
        self._render_options()

    def _render_options(self) -> None:
        text = Text(self._question, style="bold")
        width = max((self.content_size.width or self.app.size.width - 4) - 2, 20)
        for line, style in self._detail:
            text.append("\n  " + (line if len(line) <= width else line[:width - 1] + "…"), style=style)
        if self._filterable:
            text.append(f"\n  Filter: {self._filter}▏", style="#8ab4f8")
            if not self._options:
                typed = self._filter.strip()
                text.append("\n  No match" + (f"; Enter uses '{typed}' as typed" if typed else ""), style="dim")
        count = len(self._options)
        rows = [f"{i + 1}. {self._options[i][1]}" for i in range(count)]
        start, end = _option_window(count, self._cursor)
        _append_option_rows(text, rows, self._cursor, start, end)
        text.append("\n" + self._hint, style="dim")
        self.update(text)

    def _choose(self, index: int) -> None:
        if self._chosen:
            return
        self._chosen = True
        self.post_message(self.Chosen(self._options[index][0], source_id=self.id or ""))

    def on_click(self, event: Click) -> None:
        index = event.style.meta.get("option")
        if isinstance(index, int) and index < len(self._options):
            self._cursor = index
            self._render_options()
            self._choose(index)

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


class AskUserPanel(Static, can_focus=True):
    """The `ask_user` panel (spec D6): the questions as tabs, the current one's options, and one submit row.

    One instance is composed hidden and reused (mounting a second `#ask-panel` raised DuplicateIds before,
    NOTES.md 2026-10-31), and `configure` resets it for the next call. The panel owns what it must to render
    itself — the answers recorded so far, the cursors and the `✓` tab strip — and posts one `Submit(answers)`;
    it never talks to the adapter or to the tool (tasks.md open point 2).

    Nothing submits on a single keystroke: a pick only records the answer and moves on, and the call is
    submitted from the `Submit answers` row, which is inert until every question has an answer.
    """

    class Submit(Message):
        def __init__(self, answers: list[str]) -> None:
            super().__init__()
            self.answers = answers

    class NeedsInput(Message):
        """The `Other` row was picked: the app hands the input over for the typed answer (D6/REQ-4.3)."""

        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    class Inert(Message):
        """Enter (or a click) on the submit row while a question is still unanswered."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(markup=False, **kwargs)
        self.questions: list[dict] = []
        self._single: list[str | None] = []  # the single-select pick, per question
        self._toggles: list[set[int]] = []  # the multi-select picks (option indexes), per question
        self._other: list[str] = []  # the typed answer to "Other", per question
        self.cursor_question = 0
        self.cursor_row = 0  # an option row, or len(options) for the Other row
        self.submit_focused = False  # the cursor is on the submit row

    # ── State ──────────────────────────────────────────────────────────────────

    def configure(self, questions: list[dict]) -> None:
        """Show a new call's questions, forgetting the previous call's answers."""
        self.questions = questions
        self._single = [None] * len(questions)
        self._toggles = [set() for _ in questions]
        self._other = [""] * len(questions)
        self.cursor_question = 0
        self.cursor_row = 0
        self.submit_focused = False
        if self.is_mounted:
            self._render_panel()

    def _options(self, index: int) -> list[dict]:
        return self.questions[index].get("options") or []

    def _multi(self, index: int) -> bool:
        return bool(self.questions[index].get("multi_select"))

    def _row_count(self, index: int) -> int:
        """Option rows plus the `Other` row; 1 for a free-text question (D6)."""
        return len(self._options(index)) + 1

    def answer(self, index: int) -> str | None:
        """One question's answer, or None while it is unanswered. Multi-select joins in **option order**
        with the typed answer last (D5)."""
        if not 0 <= index < len(self.questions):
            return None
        if not self._multi(index):
            return self._single[index] or None
        picks = [_one_line_answer(option.get("label", "")) for i, option in enumerate(self._options(index))
                 if i in self._toggles[index]]
        picks += [self._other[index]] if self._other[index] else []
        return ", ".join(pick for pick in picks if pick) or None

    def answers(self) -> list[str | None]:
        return [self.answer(i) for i in range(len(self.questions))]

    def answered(self, index: int) -> bool:
        return self.answer(index) is not None

    def all_answered(self) -> bool:
        return bool(self.questions) and all(self.answered(i) for i in range(len(self.questions)))

    def record_other(self, index: int, text: str) -> None:
        """The typed answer to `Other` (or to a free-text question) came back from the input."""
        answer = _one_line_answer(text)
        if not answer:
            return
        self._other[index] = answer  # the row shows what was typed, single- or multi-select
        if not self._multi(index):
            self._single[index] = answer
        self.cursor_question, self.submit_focused = index, False
        self.cursor_row = self._other_row(index) if self._multi(index) else 0
        self._advance(index)
        self._render_quietly()

    def _other_row(self, index: int) -> int:
        return len(self._options(index))

    def _advance(self, from_index: int) -> None:
        """Auto-advance to the next unanswered question (wrapping), else the submit row (D6); when every
        question is answered it stops on the submit row."""
        count = len(self.questions)
        for step in range(1, count + 1):
            index = (from_index + step) % count
            if not self.answered(index):
                self._land_on(index)
                return
        self.submit_focused = True

    def _land_on(self, index: int) -> None:
        """The cursor arrives at a question. A question with no options shows its single `Other` row rather
        than taking the input: the panel came up with all the answers on screen, so moving onto one (or
        landing on it after a pick) must never swallow them. Enter on that row asks for the input (D6)."""
        self.cursor_question, self.cursor_row, self.submit_focused = index, 0, False

    # ── Rendering ──────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self._render_panel()

    def _tabs(self) -> Text:
        """The tab strip (D6): one tab per question, the current one unmistakable.

        The strip is the only thing that says which question you are on, so the current tab is bold orange
        and carries a solid bar; an answered one carries `✓` in the same slot, so every tab is the same
        width and the strip does not re-flow as answers land. Plain glyphs, never emoji: this line shares
        its marks with the option rows' own `❯`/`✓`."""
        text = Text()
        for i, question in enumerate(self.questions):
            if i:
                text.append(" · ", style="dim")
            here = i == self.cursor_question and not self.submit_focused
            if here:
                mark, mark_style = "▌ ", ASK_CURRENT_TAB
            elif self.answered(i):
                mark, mark_style = "✓ ", "bold #98c379"
            else:
                mark, mark_style = "  ", ""
            text.append(mark, style=mark_style)
            label = _one_line_answer(question.get("header") or f"Q{i + 1}")
            text.append(label, style=ASK_CURRENT_TAB if here else "")
        return text

    def _submit_line(self) -> tuple[str, str]:
        """(text, style) of the submit row: bright when it can submit, dim with the hint until then (D6)."""
        if self.all_answered():
            label, style = ASK_SUBMIT_LABEL, "bold #8ab4f8"
        else:
            label, style = f"{ASK_SUBMIT_LABEL}  ({ASK_INERT_HINT})", "dim"
        return ((" ❯ " if self.submit_focused else "   ") + label), style

    def _row_texts(self, index: int, width: int) -> list[str]:
        """`✓ 1. label   description` per row, the description cut to one row (D6: the `detail` idea)."""
        rows: list[str] = []
        for i, option in enumerate(self._options(index)):
            picked = i in self._toggles[index] if self._multi(index) else self._single[index] == _one_line_answer(option.get("label", ""))
            label = _one_line_answer(option.get("label", ""))
            line = f"{'✓' if picked else ' '} {i + 1}. {label}"
            description = _one_line_answer(option.get("description", ""))
            if description:
                room = max(width - len(line) - 3, 8)
                line += "   " + (description if len(description) <= room else description[:room - 1] + "…")
            rows.append(line)
        typed = self._other[index]
        number = len(self._options(index)) + 1
        rows.append(f"{'✓' if typed else ' '} {number}. {ASK_OTHER_LABEL}" + (f": {typed}" if typed else ""))
        return rows

    def _render_panel(self) -> None:
        if not self.questions:
            self.update(Text(""))
            return
        width = max((self.content_size.width or self.app.size.width) - 4, 20)
        index = min(self.cursor_question, len(self.questions) - 1)
        text = self._tabs()
        text.append("\n   " + _one_line_answer(self.questions[index].get("question", "")), style="bold")
        rows = self._row_texts(index, width)
        start, end = _option_window(len(rows), min(self.cursor_row, len(rows) - 1))
        _append_option_rows(text, rows, self.cursor_row if not self.submit_focused else -1, start, end)
        text.append("\n   " + "─" * max(min(width - 3, 56), 10), style="dim")
        line, style = self._submit_line()
        text.append("\n" + line, style=Style(color="#8ab4f8", bold=True) + Style(meta={"ask-submit": True})
                    if style.startswith("bold") else Style(dim=True, meta={"ask-submit": True}))
        text.append("\n " + self._hint(), style="dim")
        self.update(text)

    def _render_quietly(self) -> None:
        if self.is_mounted:
            self._render_panel()

    def _hint(self) -> str:
        """The keys line. A free-text question's only row is `Other`, so say how to take the input from it;
        the generic hint would leave Enter unexplained on that row."""
        index = self.cursor_question
        if not self.submit_focused and not self._options(index):
            return "Enter or o types the answer · ←/→ change question · Esc cancel"
        return ASK_HINT

    # ── Keys and clicks (D6 table) ─────────────────────────────────────────────

    def _record_row(self, row: int) -> None:
        """A pick or toggle of one row of the current question; `Other` hands the input over."""
        index = self.cursor_question
        if row >= self._other_row(index):
            self.cursor_row = self._other_row(index)
            if self._multi(index) and self._other[index]:
                self._other[index] = ""  # picking a typed "Other" again toggles it off (D6)
                self._render_quietly()
                return
            self.post_message(self.NeedsInput(index))
            return
        if self._multi(index):
            if row in self._toggles[index]:
                self._toggles[index].discard(row)
            else:
                self._toggles[index].add(row)
            self.cursor_row = row
            self._render_quietly()
            return  # a toggle stays put: several picks are the point
        self._single[index] = _one_line_answer(self._options(index)[row].get("label", ""))
        self._other[index] = ""
        self._advance(index)
        self._render_quietly()

    def _submit(self) -> None:
        if self.all_answered():
            self.post_message(self.Submit([self.answer(i) or "" for i in range(len(self.questions))]))
        else:
            self.post_message(self.Inert())

    def _go_to_question(self, index: int) -> None:
        self._land_on(index % len(self.questions))
        self._render_quietly()

    def previous_question(self) -> None:
        """Shift+Tab (the app's binding reaches here while the panel is up, D6)."""
        if not self.questions:
            return
        if self.submit_focused:
            self._go_to_question(len(self.questions) - 1)
        else:
            self._go_to_question(self.cursor_question - 1)

    def on_click(self, event: Click) -> None:
        meta = event.style.meta
        if meta.get("ask-submit"):
            self.submit_focused = True
            self._render_quietly()
            self._submit()
            return
        row = meta.get("option")  # the same meta key every selector's rows carry
        if isinstance(row, int) and 0 <= row < self._row_count(self.cursor_question):
            self.cursor_row, self.submit_focused = row, False
            self._record_row(row)

    def on_key(self, event: Key) -> None:
        key = event.key
        if key in ("tab", "right"):
            if self.submit_focused:
                return
            if self.cursor_question + 1 < len(self.questions):
                self._go_to_question(self.cursor_question + 1)
            else:
                self.submit_focused = True
                self._render_quietly()
        elif key in ("shift+tab", "left", "backtab"):
            self.previous_question()
        elif key in ("up", "k", "down", "j"):
            if self.submit_focused:
                self.submit_focused = False
                self.cursor_row = 0
            else:
                step = -1 if key in ("up", "k") else 1
                self.cursor_row = (self.cursor_row + step) % self._row_count(self.cursor_question)
            self._render_quietly()
        elif key == "enter":
            if self.submit_focused:
                self._submit()
            else:
                self._record_row(self.cursor_row)
        elif key == "space" and self._multi(self.cursor_question):
            self.submit_focused = False  # a toggle is a row pick, from the submit row too
            self._record_row(self.cursor_row)
        elif key == "o" and not self.submit_focused and self.cursor_row >= self._other_row(self.cursor_question):
            # `o` is the deliberate way into a typed answer, for a free-text question whose panel is just
            # its Other row. It always asks for the input — never the Enter row's toggle-off, which would
            # throw a typed answer away when the user meant to edit it.
            self.cursor_row = self._other_row(self.cursor_question)
            self.post_message(self.NeedsInput(self.cursor_question))
        elif key.isdigit():
            number = int(key)
            if 1 <= number <= len(self._options(self.cursor_question)):
                self.submit_focused, self.cursor_row = False, number - 1
                self._record_row(number - 1)
            elif number == self._other_row(self.cursor_question) + 1:
                self.submit_focused, self.cursor_row = False, self._other_row(self.cursor_question)
                self._record_row(self._other_row(self.cursor_question))
            else:
                return
        else:
            return
        event.stop()
        event.prevent_default()


class BtwPanel(Vertical, can_focus=True):
    """The `/btw` side answer (btw spec D6): a title row, a scrollable body and a footer of keys.

    Read-only, and outside `#msg-container`, so nothing in it can reach the transcript or `/export`. One instance is
    composed hidden and reused, never mounted per question (the `DuplicateIds` lesson of `#picker` and
    `#ask-panel`). It sits above the input and leaves the conversation visible: the turn keeps running underneath.

    The panel owns its keys while focused (Enter/Space close, `f` fork out, `c` copy, ↑/↓ scroll); Esc is the app's
    priority binding, which closes the panel unless an approval or a question is waiting (`action_cancel`).
    """

    BINDINGS = [
        Binding("enter", "close", "Close", show=False),
        Binding("space", "close", "Close", show=False),
        Binding("f", "fork_out", "Fork out", show=False),
        Binding("c", "copy", "Copy", show=False),
        Binding("up", "scroll_body(-1)", show=False),
        Binding("down", "scroll_body(1)", show=False),
        Binding("pageup", "scroll_body(-10)", show=False),
        Binding("pagedown", "scroll_body(10)", show=False),
    ]

    class Close(Message):
        """Enter or Space."""

    class ForkOut(Message):
        """`f` on an answer: hand the question to a background fork (D8)."""

    class Copy(Message):
        """`c` on an answer."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.question = ""
        self.snapshot: Any = None  # the ForkContext the side call used; a fork-out gets this same object (D8)
        self.answer: Any = None  # the SideAnswer once it arrived
        self.error = ""
        self.body_text = ""  # what the body shows, as plain text

    def compose(self) -> ComposeResult:
        yield Static(id="btw-title", markup=False)
        with VerticalScroll(id="btw-scroll", can_focus=False):
            yield Static(id="btw-body", markup=False)
        yield Static(id="btw-footer", markup=False)

    def configure(self, question: str) -> None:
        """A new side question: forget the last answer, show `Asking…`."""
        self.question, self.snapshot, self.answer, self.error = question, None, None, ""
        self.body_text = "Asking…"
        self.query_one("#btw-title", Static).update(Text(f"/btw — {_one_line(question, 500)}"))
        self.query_one("#btw-body", Static).update(Text("Asking…", style="#8a8a8a"))
        self.query_one("#btw-footer", Static).update(BTW_ASKING_FOOTER)
        self.query_one("#btw-scroll", VerticalScroll).scroll_home(animate=False)

    def show_answer(self, answer: Any, usage_text: str) -> None:
        self.answer = answer
        self.body_text = answer.text
        self.query_one("#btw-body", Static).update(RichMarkdown(answer.text))
        self.query_one("#btw-footer", Static).update(
            BTW_ANSWER_FOOTER + (f" · {usage_text}" if usage_text else ""))

    def show_fork_error(self, message: str) -> None:
        """A fork-out that could not start: said under the answer, which stays (REQ-4.1)."""
        self.error = message
        self.body_text = f"{self.answer.text}\n\n{message}" if self.answer is not None else message
        self.query_one("#btw-body", Static).update(
            Group(RichMarkdown(self.answer.text), Text(""), Text(message, style="#e06c75"))
            if self.answer is not None else Text(message, style="#e06c75"))

    def show_error(self, message: str) -> None:
        """A failed call or fork-out: said here, never as a turn error in the transcript (D6)."""
        self.error = self.body_text = message
        self.query_one("#btw-body", Static).update(Text(message, style="#e06c75"))
        self.query_one("#btw-footer", Static).update(BTW_ANSWER_FOOTER if self.answer is not None else BTW_CLOSE_FOOTER)

    def action_close(self) -> None:
        self.post_message(self.Close())

    def action_fork_out(self) -> None:
        if self.answer is not None:  # inert while asking: the fork is given the answer to check (§7.3)
            self.post_message(self.ForkOut())

    def action_copy(self) -> None:
        if self.answer is not None:
            self.post_message(self.Copy())

    def action_scroll_body(self, rows: int) -> None:
        self.query_one("#btw-scroll", VerticalScroll).scroll_relative(y=rows, animate=False)


BTW_ASKING_FOOTER = "Esc close"
BTW_CLOSE_FOOTER = "Esc/Enter/Space close"
BTW_ANSWER_FOOTER = "Esc/Enter/Space close · f fork out · c copy"
BTW_BUSY = "a side question is already running"
BTW_USAGE = "Usage: /btw <question>"
BTW_HANDED_OFF = "handed to a background agent · /agents to watch it"
BTW_FORK_STRANDED = "⚙ a side-question agent is still running; its answer stays with '{old}'"


# Shift+Enter needs a terminal that reports it (kitty keyboard protocol, e.g. kitty, WezTerm,
# Ghostty); elsewhere it arrives as plain Enter, so the others stay as fallbacks.
NEWLINE_KEYS = {"shift+enter", "ctrl+enter", "alt+enter", "ctrl+j"}
# Windows (VT input) sends Backspace as DEL and Ctrl+Backspace as ^H; Textual names both "backspace".
# Elsewhere some terminals send ^H for plain Backspace, so it's only remapped on Windows.
CTRL_H_DELETES_WORD = sys.platform == "win32"


class EnterSubmits(TextArea):
    """TextArea where Enter submits and Shift+Enter inserts a newline.

    Ctrl+Enter, Alt+Enter, Ctrl+J and a trailing "\\" before Enter also insert a newline, for
    terminals that send Shift+Enter as a plain Enter.

    While the "/" menu is open, Up/Down move through it, Tab completes and Enter runs the
    highlighted command. Otherwise Up on the first line recalls the previous prompt and Down on
    the last line goes forward again, back to the draft that was being typed.

    A slash command being typed shows the rest of it (or its flag, e.g. /model --active) as
    dim ghost text after the cursor; Tab or Right accepts it. Then a dim hint for its arguments
    (e.g. /rename <title>), which is only shown, never inserted.
    """

    # Ctrl+W / Ctrl+Backspace / Alt+Backspace delete the previous word (TextArea's own bindings)
    BINDINGS = [Binding("ctrl+delete", "delete_word_right", "Delete word right", show=False)]

    class Submitted(Message):
        """Sent when user presses Enter."""
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prompt_history: list[str] = []  # oldest first
        self._history_index: int | None = None  # None = editing the draft
        self._draft = ""
        self.recalled_text: str | None = None  # last text put in by recall (keeps the "/" menu shut)
        self.pastes = PasteStore()
        self.paste_collapse_lines = 0  # set by the app from ui.paste_collapse_lines
        self.commands: Iterable[str] = COMMAND_DESCRIPTIONS  # for ghost text; the app adds the skill commands
        self.completion = ""  # the part of `suggestion` Tab/Right accept; the rest is the argument hint

    def add_history(self, text: str) -> None:
        if not self.prompt_history or self.prompt_history[-1] != text:
            self.prompt_history.append(text)
        self._history_index = None
        self._draft = ""

    def _recall(self, step: int) -> bool:
        """Move through history (-1 older, +1 newer); False if there's nowhere to go."""
        if self._history_index is None:
            if step > 0 or not self.prompt_history:
                return False
            self._draft = self.text
            index = len(self.prompt_history) - 1
        else:
            index = self._history_index + step
            if index < 0:
                return False
        if index >= len(self.prompt_history):
            self._history_index = None
            text = self._draft
        else:
            self._history_index = index
            text = self.prompt_history[index]
        self.recalled_text = text
        self.text = text
        self.move_cursor(self.document.end)
        return True

    def partial_mention(self) -> str | None:
        """The `@...` being typed right before the cursor (with the "@"), or None."""
        row, col = self.cursor_location
        match = PARTIAL_MENTION_RE.search(self.document.get_line(row)[:col])
        return match.group(1) if match else None

    def update_suggestion(self, highlighted: str | None = None) -> None:
        """Ghost text for a slash command, only with the cursor at the end (not for recalled prompts)."""
        at_end = self.cursor_location == self.document.end and not self.selected_text
        recalled = self.text == self.recalled_text
        if not at_end or recalled:
            self.completion = self.suggestion = ""
            return
        self.completion = slash_suggestion(self.text, highlighted, self.commands)
        self.suggestion = self.completion + slash_arg_hint(self.text + self.completion, self.commands)

    def action_cursor_right(self, select: bool = False) -> None:
        """Right accepts only the completion, never the argument hint."""
        if self.completion:
            self.insert(self.completion)
            return
        self.suggestion = ""  # just the hint (or nothing): TextArea would insert it, so move as usual
        super().action_cursor_right(select)
        self.update_suggestion()

    def on_text_area_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        self.update_suggestion()  # TextArea only refreshes it on edits; the cursor may have moved to/off the end

    def _complete_mention(self, path: str) -> None:
        """Replace the @mention at the cursor with `path`; a folder stays open for the next part."""
        partial = self.partial_mention()
        if partial is None:
            return
        row, col = self.cursor_location
        mention = format_mention(path)
        if path.endswith("/") and mention.endswith('"'):
            mention = mention[:-1]  # @"my dir/ : keep typing inside the quotes
        self.replace(mention + ("" if path.endswith("/") else " "), (row, col - len(partial)), (row, col))

    async def _on_paste(self, event: Paste) -> None:
        """A long paste goes in as `[Pasted text #N +L lines]`; the app swaps the text back on submit."""
        text = event.text.replace("\r\n", "\n").replace("\r", "\n")
        if self.read_only or not self.pastes.should_collapse(text, self.paste_collapse_lines):
            return  # Textual runs TextArea._on_paste next (handlers go down the MRO), which inserts it
        event.stop()
        event.prevent_default()
        if result := self._replace_via_keyboard(self.pastes.add(text), *self.selection):
            self.move_cursor(result.end_location)
            self.focus()

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
            if event.key in ("up", "down"):
                if event.key == "up":
                    menu.action_cursor_up()
                else:
                    menu.action_cursor_down()
                if menu.mode == "command":
                    self.update_suggestion(menu.selected_command())
            elif menu.mode == "file":
                path = menu.selected_command()
                menu.display = False
                if path:
                    self._complete_mention(path)
            else:
                command = menu.selected_command()
                menu.display = False
                if command and event.key == "tab":
                    # A space after a command taking arguments, so its flag (e.g. --active) or hint shows next
                    takes_args = command in COMMAND_FLAGS or slash_arg_hint(command, self.commands)
                    self.text = command + (" " if takes_args else "")
                    self.move_cursor(self.document.end)
                elif command:
                    self.post_message(self.Submitted(command))
            return
        if event.key == "tab" and self.completion:
            event.stop()
            event.prevent_default()
            self.insert(self.completion)
            return
        if event.key in NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if event.key == "backspace" and event.character == "\x08" and CTRL_H_DELETES_WORD:
            event.stop()
            event.prevent_default()
            if not self.read_only:
                self.action_delete_word_left()
            return
        row = self.cursor_location[0]
        if (event.key == "up" and row == 0 and self._recall(-1)) or (
                event.key == "down" and row == self.document.line_count - 1 and self._recall(1)):
            event.stop()
            event.prevent_default()
            return
        if event.key == "down" and not self.text and self.app.query_one("#jobs-bar", JobsBar).display:
            event.stop()
            event.prevent_default()
            self.app.action_show_jobs()  # type: ignore[attr-defined]
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            row, col = self.cursor_location
            if col and self.document.get_line(row)[col - 1] == "\\":
                # "\" + Enter: swap the backslash for a newline instead of submitting
                self.replace("\n", (row, col - 1), (row, col))
                return
            self.post_message(self.Submitted(self.text))
            return
        await super()._on_key(event)


EditorKind = Literal["config", "mcp", "memory"]  # memory: an AGENTS.md (scope "user" = the global one)


class SettingsEditor(TextArea):
    """/config's (and /memory's) editor for one whole file, shown in place of the conversation and input.

    Ctrl+S saves (an invalid file stays open with the error); ESC is the app's cancel binding,
    which closes it, asking for a second ESC when there are unsaved changes.
    """

    BINDINGS = [Binding("ctrl+s", "save", "Save", priority=True)]

    class Save(Message):
        pass

    def __init__(self, text: str, kind: EditorKind, scope: Scope, **kwargs: Any) -> None:
        language = {"config": "yaml", "mcp": "json", "memory": "markdown"}[kind]
        super().__init__(text, language=language, soft_wrap=kind == "memory",
                         tab_behavior="indent", show_line_numbers=True, **kwargs)
        self.kind: EditorKind = kind
        self.scope: Scope = scope
        self.saved_text = text
        self.discard_armed = False  # ESC once with unsaved changes; the next ESC discards them

    @property
    def modified(self) -> bool:
        return self.text != self.saved_text

    def action_save(self) -> None:
        self.post_message(self.Save())


def _agent_registry() -> Any:
    """The subagent registry, imported late: it pulls in the MCP client, which loads after the first frame."""
    from dostuff.agent.subagents.registry import get_agent_registry
    return get_agent_registry()


class TuiAdapter:
    """
    Translates loop() output to TUI widget updates, and TUI input back to loop().

    loop() calls:
      adapter.emit(event_type, data)    → TUI shows message/tool result (any thread)
      adapter.ask(message)              → TUI shows a selector; returns an ApprovalReply
                                          ("yes" | "always" | "no", plus feedback for "no")
      adapter.ask_batch(messages)       → one selector for several calls; adds "each"
      adapter.ask_user(questions)       → the ask_user tool: every answer in order, or None when cancelled

    TUI calls:
      adapter.choose(value)             → an option was picked in the selector
      adapter.got_input(text)           → typed reply (feedback after "No", or the answer to a question)
    """

    def __init__(self, app: "DostuffTUI"):
        self._app = app
        self._waiting_for_confirm = False
        self._confirm_event: asyncio.Event | None = None
        self._confirm_reply: ApprovalReply | None = None
        self._feedback_mode = False  # "No" was picked; the input now takes feedback for the model
        self._feedback_on_no = True  # whether "No" asks for that feedback (not for the hook trust prompt)
        # The question channel (ask-user spec D7), beside the approval one and sharing its lock
        self._waiting_for_ask = False
        self._ask_event: asyncio.Event | None = None
        self._ask_answers: list[str] | None = None  # one resolution's answers (None = cancelled)
        self._ask_question: dict | None = None  # the question the input is answering right now
        self._other_mode = False  # the input takes an answer right now (a free-text call, or "Other")
        self._ask_panel_owns_input = False  # ...and it belongs to a question of the open panel
        self._other_index = 0  # which question of the panel the input is answering
        self._ask_lock = asyncio.Lock()  # one prompt at a time
        self._last_turn_usage: dict = {}
        self._session_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._cancelled: bool = False
        self._cancel_event: asyncio.Event = asyncio.Event()
        self._last_context_window_percent = ""
        self._token_display = ""
        self._session_cost = 0.0  # USD of calls with a known price (0 = none, or unknown model)
        self._turn_cost = 0.0

    def reset_usage(self) -> None:
        self._last_turn_usage = {}
        self._session_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._session_cost = 0.0
        self._turn_cost = 0.0
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
            if data.get("cost"):
                self._session_cost += data["cost"]
                self._turn_cost += data["cost"]
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
                app._append_tool_result(text.removeprefix(TODO_RESULT_PREFIX), max_lines=100)
                return
            if text.startswith(ASK_RESULT_PREFIX):  # the answers are the whole point: show every pair (D9)
                pairs = parse_answers(text.removeprefix(ASK_RESULT_PREFIX))
                if pairs is not None:
                    app._append_answers_block(pairs)
                    return  # a cancelled or refused result has no pairs and falls through to the usual line
            app._append_call_result(text)
            return
        if event_type in ("confirm", "system", "error"):
            app._append(str(data), msg_type=event_type)
            return
        if event_type == "hook" and isinstance(data, dict):
            app._append(hook_line(data), msg_type="system")
            return
        if event_type == "settings_changed":  # the settings tools changed config.yaml or the MCP servers
            app._apply_settings(mcp=data == "mcp")
            return
        if event_type in ("steered", "steer_blocked", "steer_failed") and isinstance(data, dict):
            # The loop's verdict on one steering message (steering spec D6, D9)
            app._on_steer_event(event_type, data)
            return
        label = event_type.upper().replace("_", " ")
        app._append(f"[{label}] {data}", msg_type="agent")

    async def ask(self, message: str, always_label: str = "", background: bool = False) -> ApprovalReply:
        """`background`: from a background agent, so it waits until the user isn't typing (subagents D8)."""
        options = [("yes", "Yes")]
        if always_label:
            options.append(("always", f"Yes, and don't ask again this session for {always_label}"))
        options.append(("no", "No, and tell the model what to do differently"))
        return await self._prompt(f"⚠ {message}", "Do you want to proceed?", options, background=background)

    async def ask_batch(self, messages: list[str], always_label: str = "", background: bool = False) -> ApprovalReply:
        count = len(messages)
        details = f"⚠ The model wants to run {count} actions that need approval:"
        for i, message in enumerate(messages, 1):
            details += f"\n\n[{i}/{count}] {message}"
        options = [("yes", f"Yes, allow all {count}")]
        if always_label:
            options.append(("always", f"Yes to all, and don't ask again this session for {always_label}"))
        options += [("each", "Let me review each one"), ("no", "No to all, and tell the model what to do differently")]
        return await self._prompt(details, f"Allow these {count} actions?", options, background=background)

    async def ask_trust(self, message: str) -> ApprovalReply:
        """The project-hooks trust question (helpers/hooks/trust.py): no feedback, nothing goes to the model."""
        options = [("yes", "Yes, run them (asked again if they change)"), ("no", "No, skip them this session")]
        return await self._prompt(f"⚠ {message}", "Run this project's hooks?", options, feedback=False)

    async def confirm(self, details: str, question: str, yes: str, no: str) -> bool:
        """A yes/no question from the app itself (stopping background jobs); ESC means no."""
        reply = await self._prompt(details, question, [("yes", yes), ("no", no)], feedback=False)
        return reply.answer == "yes"

    async def _acquire_prompt(self, background: bool) -> None:
        """Take `_ask_lock`; a background agent's prompt also waits until `_prompt_blocked()` is false, so it
        never cuts into typing or another picker (subagents D8), unless the user picked Answer approval for
        it in /agents. It doesn't hold the lock while it waits."""
        from dostuff.agent.subagents import current_agent
        run = current_agent.get()

        def held() -> bool:
            return background and not (run is not None and run.answer_now) and self._app._prompt_blocked()
        while True:
            while held():
                await asyncio.sleep(PROMPT_RETRY_S)
            await self._ask_lock.acquire()
            if not held():
                if run is not None:
                    run.answer_now = False  # its next prompt waits again
                return
            self._ask_lock.release()  # the user started typing while it waited for the lock

    @contextlib.asynccontextmanager
    async def _hold_prompt(self, background: bool = False) -> AsyncIterator[None]:
        """Hold the one-prompt lock for the duration of a prompt and put the title back when it resolves,
        so an approval and a question restore `running`/`done` in exactly one place (ask-user D7)."""
        await self._acquire_prompt(background)
        try:
            yield
        finally:
            # a background agent's prompt can come while idle
            self._app._set_title("running" if self._app._turn_in_progress else "done")
            self._ask_lock.release()

    async def _prompt(self, details: str, question: str, options: list[tuple[str, str]],
                      feedback: bool = True, background: bool = False) -> ApprovalReply:
        # Runs on the UI loop (turns are async workers), so widgets can be touched directly
        async with self._hold_prompt(background):
            self._waiting_for_confirm = True
            self._feedback_mode = False
            self._feedback_on_no = feedback
            self._confirm_reply = None
            evt = asyncio.Event()
            self._confirm_event = evt
            self._app._mount_message(Static(_confirm_text(details), classes="msg msg-confirm", markup=False))
            self._app._show_permission_prompt(question, options)
            self._app._approval_waiting(question)
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

    # ── The question channel (ask-user spec D7) ────────────────────────────────

    async def ask_user(self, questions: list[dict]) -> list[tuple[str, str]] | None:
        """Ask the questions; every answer in question order, or None when the user cancelled.

        The panel renders all of them and posts one `Submit` (tasks.md open point 2); a call whose questions
        are all free text never shows it, and the input takes each answer (D6). Emotionally this is the
        sibling of `_prompt`: it shares the lock, the title and the notification.
        """
        if self._cancelled or self._cancel_event.is_set():
            return None  # the turn is being cancelled: don't open a prompt on the way out
        async with self._hold_prompt():
            if self._uses_panel(questions):
                answers = await self._ask_with_panel(questions)
            else:
                answers = await self._ask_free_text(questions)
            if answers is None:
                return None  # cancelled: the tool records the D5 cancel text
            return [(_one_line_answer(q.get("question", "")), answer) for q, answer in zip(questions, answers)]

    @staticmethod
    def _uses_panel(questions: list[dict]) -> bool:
        """A call with no option questions never shows the panel: each Enter is the submission (D6)."""
        return any(question.get("options") for question in questions)

    async def _ask_with_panel(self, questions: list[dict]) -> list[str] | None:
        """The panel's one answer list, or None on Esc. A free-text question in a mixed call hands the
        input over when its turn comes; the panel keeps the rest."""
        event = self._start_ask(questions[0] if questions else {})
        try:
            self._app._show_ask_panel(questions)
            await event.wait()
            return self._ask_answers
        finally:
            self._finish_ask()

    async def _ask_free_text(self, questions: list[dict]) -> list[str] | None:
        """Every question is free text: the input takes each answer with the question as its placeholder
        (D6), and the last Enter is the submission; there is no submit row."""
        answers: list[str] = []
        for question in questions:
            event = self._start_ask(question)
            try:
                self._other_mode = True  # the input is the answer to this question
                self._ask_question = question
                self._app._hand_input_over(_one_line_answer(question.get("question", "")))
                await event.wait()
            finally:
                self._finish_ask()
            if self._ask_answers is None:
                return None
            answers.extend(self._ask_answers)
        return answers

    def _start_ask(self, question: dict) -> asyncio.Event:
        """Open the question channel: one event per resolution, and the title/notification of REQ-4.6."""
        self._waiting_for_ask = True
        self._other_mode = False
        self._ask_panel_owns_input = False
        self._ask_answers = None
        self._ask_question = question
        event = asyncio.Event()
        self._ask_event = event
        self._app._approval_waiting(_one_line_answer(question.get("question", "")))
        return event

    def _finish_ask(self) -> None:
        self._waiting_for_ask = False
        self._other_mode = False
        self._ask_panel_owns_input = False
        self._other_index = 0
        self._ask_event = None
        self._ask_question = None
        self._app._hide_ask_panel()
        self._app._give_input_back()

    def _resolve_ask(self, answers: list[str] | None) -> None:
        """One resolution of the question channel: the answers (None = cancelled) wake the waiting turn."""
        if self._waiting_for_ask and self._ask_event is not None:
            self._ask_answers = answers
            self._ask_event.set()

    def cancel_pending_ask(self) -> None:
        """Esc, Ctrl+Q or /exit while a question is open: resolve it as cancelled. A no-op when idle (D7)."""
        self._resolve_ask(None)

    def has_pending_ask(self) -> bool:
        return self._waiting_for_ask

    def has_pending_prompt(self) -> bool:
        """The app's one gate (D7): a prompt of either kind owns the input. `has_pending_confirm` stays
        approval-only, so the approval paths are untouched (CON-005)."""
        return self._waiting_for_confirm or self._waiting_for_ask

    def _resolve(self, reply: ApprovalReply) -> None:
        if self._waiting_for_confirm and self._confirm_event is not None:
            self._confirm_reply = reply
            self._confirm_event.set()

    # ── Called by the TUI ───────────────────────────────────────────────────────

    def choose(self, value: str) -> None:
        """An option was picked; "no" first asks for (optional) feedback in the input."""
        if not self._waiting_for_confirm:
            return
        if value == "no" and self._feedback_on_no:
            self._feedback_mode = True
            self._app._ask_denial_feedback()
            return
        self._resolve(ApprovalReply(value))  # type: ignore[arg-type]

    def got_input(self, text: str) -> None:
        if self._other_mode:
            # The input is answering a question (ask-user D7): in a free-text-only call the text is the whole
            # answer to this question, so a number is text too; otherwise it answers the "Other" row.
            answer = _one_line_answer(text)
            if not answer:  # open point 4: an empty Enter records nothing; Esc is the way out
                self._app._flash_input_hint(ASK_EMPTY_INPUT_HINT)
                return
            if self._ask_panel_owns_input:
                self._app._ask_panel_other(self._other_index, answer)
            else:
                self._resolve_ask([answer])
        elif self._feedback_mode:
            self._resolve(ApprovalReply("no", feedback=text))
        else:
            self._resolve(ApprovalReply(parse_answer(text)))

    def got_ask_submit(self, answers: list[str]) -> None:
        """The panel's `Submit answers` row: every question is answered, so the call resolves (D6)."""
        self._resolve_ask(list(answers))

    def begin_other(self, index: int) -> None:
        """The panel's `Other` row (or a free-text question in a mixed call): the input takes the text."""
        if not self._waiting_for_ask:
            return
        self._other_mode = True
        self._ask_panel_owns_input = True
        self._other_index = index

    def end_other(self) -> None:
        """The typed answer was recorded: the input is a prompt input again, and the panel takes over."""
        self._other_mode = False
        self._ask_panel_owns_input = False

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
    #ask-panel {
        height: auto; margin-top: 1; padding: 0 1; background: transparent; border: round #8ab4f8;
    }
    .msg-error { color: #e06c75; }
    .msg-system { color: #8a8a8a; }
    .msg-loader { color: #8ab4f8; }
    .msg-thought { color: #8a8a8a; text-style: italic; }
    .msg-tool { }
    .msg-tool-result { color: #8a8a8a; margin-top: 0; }
    .msg-diff { margin-top: 0; }
    .msg-shell { color: #d7769f; }
    .msg-shell-output { color: #8a8a8a; margin-top: 0; }
    /* The "/" and @mention menu floats over the bottom of the conversation on its own layer, so
       opening it doesn't resize #messages (that relayout repainted the whole screen: a flash). */
    #conversation { height: 1fr; layers: base menu; }
    #command-menu {
        layer: menu; dock: bottom;
        height: auto; max-height: 8; display: none; background: ansi_default;
        border: none; padding: 0 1; scrollbar-size-vertical: 1;
    }
    #command-menu > .option-list--option-highlighted { background: transparent; color: #8ab4f8; }
    #command-menu > .option-list--option-hover { background: transparent; }
    #input {
        height: auto; min-height: 3; max-height: 12; margin-top: 1;
        background: transparent; border: round #4a4a4a; padding: 0 1;
    }
    #input:focus { border: round #7a7a7a; }
    /* "Copied N characters" hint, shown in the input's top border instead of a toast */
    #input { border-title-align: right; border-title-color: #8a8a8a; }
    #input.shell-mode { border: round #d7769f; }
    #input.feedback-mode { border: round #e5c07b; }
    #input > .text-area--cursor-line { background: transparent; }
    /* Ghost text: the theme's $text-muted is the terminal's plain foreground under ansi-dark */
    #input > .text-area--suggestion { color: #6a6a6a; }
    /* The placeholder is where a handed-over input puts the question (ask-user D7) and where the denial
       note asks for text. Textual's default is `$text 40%`, which an ANSI theme cannot blend, so it paints
       plain white and reads as a real prompt. Dim it to the ghost text's grey, one step quieter than text. */
    #input > .text-area--placeholder { color: #6a6a6a; }
    #status-row { height: 1; }
    #status { width: auto; height: 1; color: #8a8a8a; padding: 0 1; }
    #jobs-bar { width: auto; height: 1; display: none; padding: 0 1; }
    #jobs-bar:hover { background: #2c2c2c; }
    #agents-bar { width: auto; height: 1; display: none; padding: 0 1; }
    #agents-bar:hover { background: #2c2c2c; }
    #queue-bar { height: auto; display: none; padding: 0 1; margin-top: 1; }
    #queue-bar:hover { background: #2c2c2c; }
    #jump-bottom {
        dock: bottom; display: none; width: 100%; height: 1; align-horizontal: center;
    }
    #jump-bottom-label {
        width: auto; height: 1; padding: 0 2;
        text-align: center; color: #8ab4f8; text-style: bold; background: #2c2c2c;
    }
    #jump-bottom-label:hover { background: #3a3a3a; }
    #agent-view { height: 1fr; }
    #agent-view-title { height: auto; padding: 0 1; color: #8ab4f8; text-style: bold; }
    #agent-view-scroll { height: 1fr; border: round #4a4a4a; }
    #agent-view-msgs { height: auto; }
    #agent-view-status { height: auto; padding: 0 1; color: #8a8a8a; }
    /* /btw: #agent-view's look (title colour, round border), but auto height so the conversation stays visible */
    #btw-panel {
        height: auto; max-height: 50vh; margin-top: 1; padding: 0 1; background: transparent;
        border: round #4a4a4a;
    }
    #btw-panel:focus { border: round #7a7a7a; }
    #btw-title { height: 1; color: #8ab4f8; text-style: bold; text-wrap: nowrap; text-overflow: ellipsis; }
    #btw-scroll { height: auto; max-height: 40vh; scrollbar-size-vertical: 1; }
    #btw-body { height: auto; }
    #btw-footer { height: auto; color: #8a8a8a; }
    #settings-panel { height: 1fr; }
    #settings-title { height: auto; padding: 0 1; color: #8ab4f8; }
    #settings-editor { height: 1fr; background: transparent; border: round #8ab4f8; }
    #settings-editor > .text-area--cursor-line { background: transparent; }
    #settings-status { height: auto; padding: 0 1; color: #8a8a8a; }
    #settings-status.error { color: #e06c75; }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+o", "toggle_tool_results", "Expand tool output", priority=True),
        Binding("shift+tab", "cycle_permission_mode", "Cycle permission mode", priority=True),
        Binding("ctrl+r", "history_search", "Search prompt history", priority=True),
        Binding("ctrl+b", "background_jobs", "Move running commands to the background", priority=True),
    ]

    # ── Setup ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        session_id: str,
        **kwargs,
    ):
        # ansi_color + an ANSI theme make "transparent" mean the terminal's own background;
        # without them Textual paints the theme's RGB background (black) behind everything.
        kwargs.setdefault("ansi_color", True)
        super().__init__(**kwargs)
        self.theme = "ansi-dark"
        self.session_id = session_id
        self.adapter = TuiAdapter(self)
        set_adapter(self.adapter)

        # Cwd already switched in cli.py before TUI launched; Config now reads from correct cwd
        self.config = get_config()
        self.store = SQLiteSessionStore(
            db_path=str(self.config.get_data_dir() / "sessions.db")
        )
        # Memory stores and the MCP client need litellm / chromadb / mcp; _load_backend sets them
        # after the first frame, then sets _backend_ready
        self.memory_store: "SemanticMemoryStore | None" = None
        self.episodic_store: "EpisodicMemoryStore | None" = None
        self.mcp_client: "MCPClient | None" = None
        self._backend_ready = asyncio.Event()
        self._backend_error = ""
        self._context_tokens = 0  # last_input_tokens; the context % needs litellm's model table

        # Session state
        self.user_id = ""
        self.steps_history: list = []
        self.current_session_history: list = []
        # SessionStart hook context for this session (hooks spec REQ-2.4), and the source to fire on its next turn
        self.session_hook_context: dict[str, str] = {}
        self._pending_session_start: str | None = None
        self._setup_done = False
        self._system_instructions: str = ""
        self._loader_widget: Static | None = None  # loader widget at bottom of stream
        self._loader_label = ""  # loader text without the elapsed time / queue count
        self._stream_widgets: dict[str, Static] = {}  # "answer" / "thought" → live widget
        self._stream_texts: dict[str, str] = {}
        self._thought_shown = 0  # chars of the live thought text revealed so far
        self._thought_step = 1
        self._thought_timer: Timer | None = None
        self._last_streamed_answer: str = ""
        self._prompt_queue: list = []
        # Mid-turn steering (plan/2026-11-04-steering): one queue per app, filled at submit and drained by
        # loop() at the next tool boundary. In memory on purpose (D4) — an undelivered message is exactly
        # as durable as a queued prompt.
        self._steering = SteeringQueue()
        # id → raw text of every message this app put on the queue. The loop's `take_ready` removes an
        # entry before it reports back, so the raw lives here until its steered / blocked / failed event
        # lands (D9: show and re-queue what the user typed, never the expansion).
        self._steer_raws: dict[str, str] = {}
        self._pending_workers: list[Worker] = []
        self._copy_hint_timer: Timer | None = None
        self._turn_in_progress: bool = False
        self._wake_timer: Timer | None = None
        self._wake_turns = 0  # wake-up turns since the user's last message
        self._turn_start_time: float | None = None
        self._timer: Timer | None = None
        self._quitting = False
        self._quit_confirming = False
        self.max_iterations = self.config.max_iterations
        # Session-scoped approvals ("always") — reset when the app restarts
        self.permissions = SessionPermissions()
        self._draft_before_feedback = ""  # input text set aside while it takes denial feedback
        # The conversation's height before an approval prompt took the input's place (_hold_transcript)
        self._viewport_before_prompt: int | None = None
        self._picker_callback: Callable[[str], Awaitable[None]] | None = None
        self._message_target = "#msg-container"  # where _append mounts; the agent view swaps it while rendering
        self._agent_view_run: "AgentRun | None" = None  # the agent whose transcript the agent view shows
        self._agent_view_rendered = 0  # steps of it already rendered there
        self._agent_view_timer: Timer | None = None
        self._agent_view_refreshing = False
        # "/name" → skill or custom command (helpers/skills/skill_command.py)
        self._skill_commands: dict[str, dict[str, Any]] = {}
        self._mcp_configured: list[str] = []
        self._mcp_server_configs: dict[str, dict] = {}  # as connected; diffed when the settings change
        self._mcp_files_signature: tuple[tuple[int, int] | None, ...] = ()  # set when the backend connects MCP
        self._settings_check_running = False
        self._reported_load_error = ""  # a broken config.yaml is reported once, not every check
        self._tool_results_expanded = False  # Ctrl+O; new results follow it
        self._history_path = self.config.get_data_dir() / "prompt_history.jsonl"
        self._pastes = PasteStore()  # collapsed pastes in the input (helpers/agent/pastes.py)
        self.worktree: Worktree | None = None  # set while the cwd is a dostuff git worktree
        self.exit_note = ""  # printed by cli.py after the TUI closes (e.g. a kept worktree)
        self._notify_kind = self.config.notifications  # /notifications changes it for the session
        self._title_state: TitleState = "idle"
        self._title_pushed = False
        self._last_escape = 0.0  # time.monotonic() of an Esc with nothing to cancel (Esc Esc edits the last prompt)
        # (session id, steps_history index) → the prompt as typed, before @mentions / skills expanded it
        self._sent_prompts: dict[tuple[str, int], str] = {}
        # /btw (btw spec D5): this session's side thread as sent (question step, answer), and the call in flight
        self._side_exchanges: list[tuple[str, str]] = []
        self._btw_worker: Worker | None = None
        self._btw_forking = False  # an `f` is starting its fork; a second press does nothing (D8)

    # ── Compose UI ─────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        # Not focusable: clicking the conversation keeps focus in the input (see _keep_input_focus)
        with Vertical(id="conversation"):
            with Transcript(id="messages", can_focus=False):
                yield Vertical(id="msg-container")
                yield JumpToBottom("↓ Jump to bottom", id="jump-bottom")
            yield CommandMenu(id="command-menu")
        yield QueueBar(id="queue-bar")
        btw = BtwPanel(id="btw-panel")
        btw.display = False  # a /btw side answer, above the input (btw spec D6)
        yield btw
        picker = PermissionPrompt("", [], id="picker")
        picker.display = False  # shown in place of the input by _show_picker
        yield picker
        panel = AskUserPanel(id="ask-panel")
        panel.display = False  # the ask_user questions, in place of the input (D6)
        yield panel
        yield EnterSubmits(id="input", language="")
        with Horizontal(id="status-row"):
            yield Static(id="status")
            yield JobsBar(id="jobs-bar")
            yield AgentsBar(id="agents-bar")

    def ui(self, fn: Callable[[], Any]) -> None:
        """Run fn on the UI thread (directly if already there)."""
        if threading.get_ident() == getattr(self, "_thread_id", None):
            fn()
            return
        # An app that has exited keeps its closed loop, and call_from_thread would build its coroutine
        # before failing on it ("coroutine ... was never awaited"); the emit shim still points here
        loop = getattr(self, "_loop", None)
        if loop is None or loop.is_closed():
            return
        try:
            self.call_from_thread(fn)
        except RuntimeError:
            pass  # app is not running (shutting down)

    def _keep_input_focus(self, focused: Widget | None) -> None:
        """Focus always goes back to the input (or the selector shown in its place)."""
        if not isinstance(focused, (EnterSubmits, PermissionPrompt, AskUserPanel, BtwPanel, SettingsEditor)):
            # call_later, not call_after_refresh: waiting on a repaint is slow on Windows
            self.call_later(self._refocus_input)

    def _refocus_input(self) -> None:
        if isinstance(self.screen.focused, (EnterSubmits, PermissionPrompt, AskUserPanel, BtwPanel, SettingsEditor)):
            return
        try:
            # A prompt wins over the side answer: it is what the turn waits on (btw spec D6)
            for widget in (*self.query(PermissionPrompt), *self.query(AskUserPanel), *self.query(BtwPanel),
                           *self.query(SettingsEditor)):
                if widget.display:  # the #picker stays mounted while hidden
                    self.screen.set_focus(widget)
                    return
            inp = self.query_one("#input", EnterSubmits)
        except Exception:
            return  # shutting down
        if inp.display:
            # Set directly; widget.focus() would defer one more message hop
            self.screen.set_focus(inp)

    def _update_jump_to_bottom(self) -> None:
        """Show the jump-to-bottom bar only while the user has scrolled away from the end."""
        messages = self.query_one("#messages", VerticalScroll)
        self.query_one("#jump-bottom", JumpToBottom).display = (
            messages._anchor_released and not messages.is_vertical_scroll_end
        )

    def on_text_selected(self, event: TextSelected) -> None:
        """Select to copy, like Claude Code: releasing the mouse copies the selection."""
        text = self.screen.get_selected_text()
        if text:
            _copy_to_clipboard(self, text)
            self._flash_input_hint(f"copied {len(text)} characters")

    def _flash_input_hint(self, text: str, seconds: float = 2) -> None:
        """A dim note on the right of the input's top border, cleared after `seconds`."""
        inp = self.query_one("#input", EnterSubmits)
        inp.border_title = text
        if self._copy_hint_timer is not None:
            self._copy_hint_timer.stop()
        self._copy_hint_timer = self.set_timer(seconds, lambda: setattr(inp, "border_title", None))

    # ── Terminal title and notifications ───────────────────────────────────────

    def _write_terminal(self, sequence: str) -> None:
        if sequence and not self.is_headless and self._driver is not None:
            self._driver.write(sequence)

    def _set_title(self, state: TitleState) -> None:
        """Run status in the terminal tab title (ui.terminal_title); the old title is saved first."""
        self._title_state = state
        if not self.config.terminal_title:
            return
        if not self._title_pushed:
            self._title_pushed = True
            self._write_terminal(PUSH_TITLE)
        self._write_terminal(title_sequence(format_title(state, Path.cwd().name)))

    def _restore_title(self) -> None:
        if self._title_pushed:
            self._title_pushed = False
            self._write_terminal(restore_title_sequence())

    def _notify(self, message: str, elapsed: float) -> None:
        """Bell / OSC 9 (ui.notifications, /notifications) if the terminal is unfocused or the wait was long."""
        if should_notify(self._notify_kind, elapsed, self.config.notify_after_seconds, self.app_focus):
            self._write_terminal(notification_sequence(self._notify_kind, message))

    def _approval_waiting(self, question: str) -> None:
        self._set_title("waiting")
        started = self._turn_start_time
        self._notify(f"dostuff: approval needed: {question}", time.time() - started if started else 0.0)

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
        messages = self.query_one("#messages", VerticalScroll)
        messages.anchor()
        self.watch(messages, "scroll_y", self._update_jump_to_bottom, init=False)
        inp = self.query_one("#input", EnterSubmits)
        inp.focus()
        self.watch(self.screen, "focused", self._keep_input_focus, init=False)
        get_registry().add_listener(self._on_job_event)
        _agent_registry().add_listener(self._on_agent_event)
        _agent_registry().attach_store(self.store)  # agent rows and transcripts (subagents D14)
        inp.prompt_history = load_history(self._history_path, os.getcwd())
        inp.pastes = self._pastes
        inp.paste_collapse_lines = self.config.paste_collapse_lines
        try:
            await asyncio.to_thread(sweep_old_pastes, self.config.paste_retention_days)
        except OSError:
            pass

        self._system_instructions, skills = build_base_instructions(interactive=True)
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
        self.worktree = await asyncio.to_thread(detect_worktree)
        if self.worktree:
            self._append(
                f" Working in worktree '{self.worktree.name}' (branch {self.worktree.branch}). /worktree exit leaves it.",
                msg_type="system",
            )

        if existing_meta:
            self._append(f" Resumed session '{self.session_id}' ({len(self.steps_history)} step(s)).")
            self._render_history()
        else:
            self._append(f" New session '{self.session_id}'.")

        custom = await asyncio.to_thread(discover_commands)
        self._skill_commands = skill_commands(skills, custom)
        menu_commands = dict(COMMAND_DESCRIPTIONS)
        for command, skill in self._skill_commands.items():
            description = " ".join(str(skill.get("description", "")).split())
            if len(description) > SKILL_MENU_DESCRIPTION_CHARS:
                description = description[:SKILL_MENU_DESCRIPTION_CHARS - 1] + "…"
            kind = f"{skill['scope']} command" if skill.get("kind") == "command" else "skill"
            menu_commands[command] = f"({kind}) {description}".rstrip()
        self.query_one("#command-menu", CommandMenu).commands = menu_commands
        inp.commands = menu_commands
        if skills:
            skill_names = ", ".join(s["name"] for s in skills)
            self._append(f"Skills loaded: {len(skills)} — {skill_names}", msg_type="system")
        else:
            self._append("Skills loaded: 0 (add to .dostuff/skills/ or ~/.agents/skills/)", msg_type="system")
        if custom:
            self._append(f"Custom commands: {', '.join('/' + c['name'] for c in custom)}", msg_type="system")

        if self.config.permission_mode == "auto":
            self._append(
                "⚠ permissions.mode is 'auto': every tool call (shell, writes, deletes, MCP, web fetches) runs WITHOUT asking.",
                msg_type="error",
            )

        if self.config.status_bar_unknown:
            self._append(
                f"ui.status_bar: unknown item(s) {', '.join(self.config.status_bar_unknown)} ignored. "
                f"Known: {', '.join(STATUS_BAR_ITEMS)}.",
                msg_type="error",
            )
        for warning in self.config.hooks_warnings:
            self._append(warning, msg_type="error")
        self._start_worker(self._jobs_housekeeping(), name="jobs_housekeeping")
        self.set_interval(SETTINGS_WATCH_INTERVAL_S, self._check_settings_files)

        self._update_status()
        self._append("Type a message or /help for commands.")
        self._start_worker(self._load_backend(), name="load_backend")

    async def _load_backend(self) -> None:
        """After the first frame: import litellm / chromadb / mcp in a thread, create the memory
        stores and MCP client, then connect MCP servers in the background."""
        try:
            await asyncio.to_thread(_import_backend)
            from dostuff.lib.mcp.mcp_client import MCPClient
            from dostuff.lib.mcp.mcp_client_registration_store import MCPClientRegistrationStore
            from dostuff.lib.memory.stores import get_episodic_store, get_semantic_store, make_tool_registry_store

            data_dir = self.config.get_data_dir()
            # Same instances the save_memory / save_episode tools write to (None without chromadb)
            self.memory_store, self.episodic_store, tool_store = await asyncio.to_thread(
                lambda: (get_semantic_store(), get_episodic_store(), make_tool_registry_store()))
            self.mcp_client = MCPClient(
                registration_store=MCPClientRegistrationStore(path=str(data_dir / "mcp_client_registrations.json")),
                tool_registry_store=tool_store,
            )
            if self.memory_store is None:
                self._note_memory_off()
            else:
                await self._ensure_embedding_model()
        except Exception as e:
            self._backend_error = str(e)
            self._append(f"Startup failed: {e}", msg_type="error")
            return
        finally:
            self._backend_ready.set()
        self._refresh_context_status(self._context_tokens)
        self._warn_if_model_unconfigured()

        # Before reading, so an edit that lands mid-read is seen by the next _check_settings_files()
        self._mcp_files_signature = await asyncio.to_thread(_mcp_files_signature)
        mcp_servers = load_mcp_config()
        self._mcp_configured = list(mcp_servers)
        self._mcp_server_configs = dict(mcp_servers)
        if mcp_servers:
            self._append(f"Connecting to {len(mcp_servers)} MCP server(s)...", msg_type="system")
            self._start_worker(self._connect_mcp_servers(mcp_servers), name="mcp_connect")
        else:
            self._append("Running with local tools only.", msg_type="system")

    def _note_memory_off(self) -> None:
        """Without chromadb, say so once if memories were saved before (e.g. an upgrade from 0.1)."""
        if (self.config.get_data_dir() / "chroma").exists():
            from dostuff.lib.memory.chroma import MEMORY_INSTALL_HINT
            self._append(f"Long-term memory is off: chromadb isn't installed. Your saved memories are kept; "
                         f"run `{MEMORY_INSTALL_HINT}` to turn it back on.", msg_type="system")

    async def _ensure_embedding_model(self) -> None:
        """First run: download Chroma's embedding model with a message, before the first turn
        (otherwise the first memory lookup downloads it silently and the turn seems to hang)."""
        from dostuff.lib.memory.chroma import download_embedding_model, embedding_model_missing
        if not await asyncio.to_thread(embedding_model_missing):
            return
        self._append("Downloading the memory embedding model (~80 MB, first run only)...", msg_type="system")
        try:
            await asyncio.to_thread(download_embedding_model)
        except Exception as e:
            self._append(f"Embedding model download failed: {e}\n"
                         "Memory recall is off until it succeeds (it retries on the next start).", msg_type="error")
            return
        self._append("Embedding model ready.", msg_type="system")

    async def _wait_for_backend(self, keep_loader: bool = False) -> bool:
        """Wait for _load_backend (with a loader if it's still running); False if it failed.
        `keep_loader`: leave that loader up for the caller to relabel (a turn's, so it never blinks out)."""
        if not self._backend_ready.is_set():
            loader_widget = self._show_loader(LOADING_LIBRARIES_LABEL)
            try:
                await self._backend_ready.wait()
            finally:
                if not keep_loader:
                    self._stop_loader(loader_widget)
        if self._backend_error:
            self._append(f"Not available: startup failed ({self._backend_error}). Restart dostuff.", msg_type="error")
            return False
        return True

    async def _load_session(self, session_id: str) -> dict | None:
        """Make session_id current: load its steps and token counters; returns its meta (None if new).

        A stored session gets this cwd and a fresh last_used; a new one gets its DB row with its
        first step, so launches that never send anything don't leave empty sessions behind.
        """
        self._clear_side_thread(self.session_id)  # /resume and /fork; a no-op at startup (btw spec D10)
        self.session_id = session_id
        session_id_var.set(session_id)
        self.steps_history = await self.store.load(session_id)
        self.current_session_history = []
        existing_meta = await self.store.get_session_meta(session_id)
        self.session_hook_context = {}
        self._pending_session_start = "resume" if existing_meta else "startup"
        meta = existing_meta or {}
        if existing_meta:
            try:  # its agents, as records send_message can continue (subagents D14)
                await _agent_registry().load(session_id, self.store)
            except Exception as e:
                self._append(f"Could not load this session's agents: {e}", msg_type="error")
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
        self.adapter._session_cost = float(meta.get("cost_usd") or 0.0)
        self._refresh_context_status(meta.get("last_input_tokens") or 0)
        return existing_meta

    def _refresh_context_status(self, last_input_tokens: int) -> None:
        self._context_tokens = last_input_tokens
        if not self._backend_ready.is_set() or self._backend_error:
            return  # _load_backend calls this again once litellm is loaded
        token_limit = get_model_token_limit()
        percent = (last_input_tokens / token_limit * 100) if token_limit > 0 else 0
        limit_str = self._fmt_tokens(token_limit) if token_limit else ""
        self._update_status(context_window_percent=f"{percent:.1f}%/{limit_str}" if limit_str else f"{percent:.1f}%")

    def _render_history(self) -> None:
        for step in self.steps_history:
            self._render_history_step(step)

    def _warn_if_model_unconfigured(self) -> None:
        if not self._backend_ready.is_set():
            return  # needs litellm; _load_backend calls this once it's loaded
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

    def _render_history_step(self, step: dict, prefix: str = "") -> None:
        """Render a stored step like the live adapter does (● call / ⎿ result). `prefix` (`[agent: …] `) marks
        a subagent's transcript in /agents."""
        role = step.get("role", "?")
        content = step.get("content") or ""
        if role == "tool":
            if step.get("name") == "ask_user" and not prefix:
                pairs = parse_answers(content)
                if pairs is not None:  # the same block the live view showed (T5.2, CON-007)
                    self._append_answers_block(pairs)
                    return
            self._append_call_result(f"{prefix}{step.get('name', '')}: {content}")
            return
        if role == "user":
            shell = _parse_shell_step(content)
            stop_note = parse_stop_note(content)
            notices = parse_notices(content)
            if stop_note is not None:  # live, the loop showed this as a ⚓ line, not a prompt
                self._append(prefix + hook_line({"event": "Stop", "decision": "block", "reason": stop_note[1]}),
                             msg_type="system")
            elif notices is not None:  # live, these showed when the jobs ended
                for line in notices:
                    self._append(prefix + line, msg_type="system")
            elif shell is not None:
                self._append(f"! {shell[0]}", msg_type="shell")
                self._append_tool_result(shell[1], SHELL_PREVIEW_LINES, msg_type="shell-output")
            elif content and content != EMPTY_RESPONSE_NUDGE:
                self._append(prefix + strip_attachments(content), msg_type="user")
                attached = attached_paths(content)
                if attached:
                    self._append("Attached " + ", ".join(attached), msg_type="system")
            return
        if content:
            self._append(prefix + content, msg_type="agent")
        for tc in step.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            self._append(f"{prefix}● {format_call_label(fn.get('name', '?'), fn.get('arguments', ''))}",
                         msg_type="tool")

    # ── Commands ──────────────────────────────────────────────────────────────

    async def _compact_session(self) -> None:
        if not await self._wait_for_backend():
            return
        loader_widget = self._show_loader("Compacting...")
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

    async def _clear_session(self) -> None:
        """Start a new session (like Claude Code); the old one stays in the DB, resumable with /resume.

        Todos and undo checkpoints are keyed by session id, so they start fresh too.
        Session-scoped approvals ("always") are kept. The new session's DB row is created with
        its first step. Running background jobs keep running and move to the new session.
        """
        old_session_id = self.session_id
        new_session_id = uuid.uuid4().hex[:16]
        self._clear_side_thread(old_session_id)
        self.session_id = new_session_id
        carried = self._carry_background(old_session_id)
        self.steps_history = []
        self.current_session_history = []
        self.session_hook_context = {}
        self._pending_session_start = "clear"
        self.adapter.reset_usage()
        await self.query_one("#msg-container", Vertical).remove_children()
        self._refresh_context_status(0)
        self._append(
            f" New session '{new_session_id}'. Previous session saved; /resume or `dostuff resume {old_session_id}` brings it back.",
            msg_type="system",
        )
        self._report_carried(carried)

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
        finally:
            carried = self._carry_background(old_session_id)  # _load_session may switch before it fails
        await self.query_one("#msg-container", Vertical).remove_children()
        self._append(f" Resumed session '{session_id}' ({len(self.steps_history)} step(s)).", msg_type="system")
        self._render_history()
        if had_steps:
            self._append(f" Previous session '{old_session_id}' saved; /resume brings it back.", msg_type="system")
        self._report_carried(carried)

    async def _fork_session(self, arg: str) -> None:
        """/fork [--worktree] [title]: copy this session into a new one and switch to it; the original stays as it was.

        Files on disk are shared unless --worktree also gives the fork its own git worktree.
        The fork starts with no undo checkpoints or todos.
        """
        words = arg.split()
        in_worktree = "--worktree" in words
        title = " ".join(w for w in words if w != "--worktree")
        if in_worktree and self.worktree:
            self._append(f"Already in worktree '{self.worktree.name}'; /worktree exit first.", msg_type="error")
            return
        if not self.steps_history:
            self._append("Nothing to fork yet: this session has no messages.", msg_type="system")
            return
        old_session_id = self.session_id
        new_session_id = uuid.uuid4().hex[:16]
        # A fork-out is not carried into the fork: its run keeps the original as owner (btw spec D10, OUT-008)
        stranded = any(r.agent_type == "fork" and r.description == BTW_FORK_DESCRIPTION and r.ended is None
                       for r in _agent_registry().runs(old_session_id))
        try:
            await self.store.fork_session(old_session_id, new_session_id, title or None)
            await self._load_session(new_session_id)
        except Exception as e:
            self._append(f"Fork failed: {e}", msg_type="error")
            return
        await self.query_one("#msg-container", Vertical).remove_children()
        self._render_history()
        self._append(
            f" Forked '{old_session_id}' into '{new_session_id}'. You're in the fork now; "
            f"/resume {old_session_id} goes back to the original.",
            msg_type="system",
        )
        if stranded:
            self._append(BTW_FORK_STRANDED.format(old=old_session_id), msg_type="system")
        if in_worktree:
            await self._enter_worktree(f"fork-{new_session_id[:8]}")

    # ── Worktrees ──────────────────────────────────────────────────────────────

    async def _worktree_command(self, arg: str) -> None:
        """/worktree [name]: move this session into a new git worktree; /worktree exit leaves it."""
        if arg.lower() == "exit":
            await self._exit_worktree()
        elif self.worktree:
            self._append(
                f"Already in worktree '{self.worktree.name}' ({self.worktree.path}); /worktree exit first.",
                msg_type="error",
            )
        else:
            await self._enter_worktree(arg or None)

    async def _enter_worktree(self, name: str | None) -> None:
        try:
            wt = await asyncio.to_thread(create_worktree, name)
        except WorktreeError as e:
            self._append(f"Worktree failed: {e}", msg_type="error")
            return
        await self._move_to(wt.path)
        self.worktree = wt
        self._append(
            f" Created worktree '{wt.name}' at {wt.path} on branch {wt.branch}, from HEAD (uncommitted changes in "
            f"{wt.repo_root} are not carried over). This session works there now; /worktree exit leaves it.",
            msg_type="system",
        )

    async def _exit_worktree(self) -> None:
        """Back to the main checkout; an unchanged worktree is removed, a changed one asks keep/remove."""
        wt = self.worktree
        if wt is None:
            self._append("Not in a worktree. /worktree [name] creates one.", msg_type="error")
            return
        try:
            dirty, commits = await asyncio.to_thread(worktree_changes, wt)
        except WorktreeError as e:
            self._append(f"Could not check the worktree for changes: {e}", msg_type="error")
            return
        if not dirty and not commits:
            await self._leave_worktree(wt, remove=True)
            return
        summary = " and ".join(part for part in (
            f"{len(dirty)} uncommitted file(s)" if dirty else "",
            f"{commits} commit(s)" if commits else "",
        ) if part)

        async def choose(value: str) -> None:
            await self._leave_worktree(wt, remove=value == "remove")

        self._show_picker(
            f"Worktree '{wt.name}' has {summary}. Keep it?",
            [("keep", f"Keep it (branch {wt.branch} stays at {wt.path})"),
             ("remove", "Remove it and its branch, discarding the changes")],
            choose,
        )

    async def _leave_worktree(self, wt: Worktree, remove: bool) -> None:
        await self._move_to(wt.repo_root)  # first: Windows can't delete the cwd
        self.worktree = None
        if not remove:
            self._append(f" Left worktree '{wt.name}'; kept at {wt.path} on branch {wt.branch}.", msg_type="system")
            return
        try:
            await asyncio.to_thread(remove_worktree, wt)
        except WorktreeError as e:
            self._append(f"Back in {wt.repo_root}, but removing the worktree failed: {e}", msg_type="error")
            return
        self._append(f" Left and removed worktree '{wt.name}'. Back in {wt.repo_root}.", msg_type="system")

    async def _move_to(self, path: Path) -> None:
        """chdir the session to path; tools, the status bar and the system prompt follow the cwd."""
        os.chdir(path)
        self._system_instructions, _ = await asyncio.to_thread(build_base_instructions, True)
        meta = await self.store.get_session_meta(self.session_id)
        if meta:
            await self.store.save_session_meta(
                self.session_id,
                working_dir=str(path),
                prompt_tokens=meta.get("prompt_tokens") or 0,
                completion_tokens=meta.get("completion_tokens") or 0,
                total_tokens=meta.get("total_tokens") or 0,
                compaction_notes=meta.get("compaction_notes") or "",
                last_input_tokens=meta.get("last_input_tokens") or 0,
            )
        self._render_status()

    async def _close_worktree_on_exit(self) -> None:
        """On quit: remove an unchanged worktree; keep a changed one and say where it is."""
        wt = self.worktree
        if wt is None:
            return
        try:
            changed = await asyncio.to_thread(has_changes, wt)
        except WorktreeError as e:
            self.exit_note = f"Kept worktree {wt.path} (branch {wt.branch}); could not check it for changes: {e}"
            return
        if changed:
            self.exit_note = (
                f"Kept worktree {wt.path} (branch {wt.branch}): it has changes. "
                f"`dostuff resume {self.session_id}` reopens it; `git worktree remove` deletes it."
            )
            return
        try:
            await self._move_to(wt.repo_root)
            await asyncio.to_thread(remove_worktree, wt)
            self.exit_note = f"Removed worktree '{wt.name}' (no changes)."
        except (OSError, WorktreeError) as e:
            self.exit_note = f"Could not remove worktree {wt.path}: {e}"

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
        scope = ("Set the default model (new sessions start with it)?" if active else
                 "Switch to? This session only; /model --active sets the default for new sessions.")
        self._show_picker(
            f"Model: {current}. {scope} (or type any provider/model)",
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
        if provider in FETCHED_PROVIDERS:
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
            "." if is_default else f" for this session only. To start new sessions with it too, run /model {name} --active")
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
            f"Model: {get_model()} for this session only. To start new sessions with it too, run "
            f"/model {entry['name']} --active",
            msg_type="system",
        )
        self._warn_if_model_unconfigured()
        meta = await self.store.get_session_meta(self.session_id) or {}
        self._refresh_context_status(meta.get("last_input_tokens") or 0)

    _PERMISSION_MODE_LABELS = {
        "ask": "ask        every write / delete / shell / MCP / web fetch call asks",
        "allowlist": "allowlist  as ask, plus config's permissions.allow patterns run without asking",
        "auto": "auto       nothing asks (not recommended)",
        "plan": "plan       read-only: the agent investigates and answers with a plan, no changes",
    }
    _MODE_HINTS = {
        "ask": "ask mode", "allowlist": "allowlist mode", "auto": "auto mode: nothing asks",
        "plan": "plan mode: read-only",
    }

    def action_cycle_permission_mode(self) -> None:
        """Shift+Tab: next permission mode for this session (ask → auto → plan → allowlist)."""
        if self._ask_panel_open():
            self._ask_panel().previous_question()  # Shift+Tab in the questions panel: back a question (D6)
            return
        if self._settings_open() or self._picker_open():
            return
        mode = next_mode(self.permissions.mode or self.config.permission_mode)
        self.permissions.mode = mode  # type: ignore[assignment]
        self._update_status()
        self._flash_input_hint(self._MODE_HINTS[mode] + " · shift+tab to change", seconds=3)

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
            self._append("⚠ Every tool call (shell, writes, deletes, MCP, web fetches) now runs WITHOUT asking.", msg_type="error")
        elif mode == "plan":
            self._append("Plan mode: file changes and non-read-only shell commands are denied; the agent answers with a plan.", msg_type="system")
        elif mode == "allowlist" and not self.config.permission_allow:
            self._append("permissions.allow in config.yaml is empty, so allowlist behaves like ask.", msg_type="system")

    def _compact_auto_command(self, arg: str) -> None:
        """/compact auto <N|off>: the auto-compaction threshold for this session."""
        if not arg:
            current = self.config.compact_threshold
            self._append(f"Auto-compact: {f'at {current}% of the context window' if current else 'off'}. "
                         "Usage: /compact auto <1-100|off>", msg_type="system")
            return
        try:
            percent = parse_compact_threshold(arg)
        except ValueError:
            self._append("Usage: /compact auto <1-100|off>", msg_type="error")
            return
        self.config.session_compact_threshold = (percent,)
        self._append(
            (f"Auto-compact at {percent}% of the context window" if percent else
             "Auto-compact off (the context is still compacted if the window overflows; /compact runs it by hand)")
            + " for this session. Set loop.auto_compact_threshold in config.yaml to change the default.",
            msg_type="system",
        )

    def _notifications_command(self, arg: str) -> None:
        """/notifications [on|off|bell|osc9]: notify when a turn finishes or an approval waits (session only)."""
        if not arg:
            self._append(f"Notifications: {self._notify_kind}. Usage: /notifications on|off|bell|osc9 "
                         "(ui.notifications in config.yaml sets the default).", msg_type="system")
            return
        kind = {"on": self.config.notifications if self.config.notifications != "off" else "bell"}.get(arg, arg)
        if kind not in NOTIFY_KINDS:
            self._append("Usage: /notifications on|off|bell|osc9", msg_type="error")
            return
        self._notify_kind = kind
        if kind == "off":
            self._append("Notifications off for this session.", msg_type="system")
            return
        how = "terminal bell" if kind == "bell" else "OSC 9 desktop notification"
        self._append(f"Notifications on ({how}) for this session: when a turn takes over "
                     f"{self.config.notify_after_seconds:g}s or the terminal is in the background, and when an "
                     "approval is waiting.", msg_type="system")

    def _copy_last_answer(self) -> None:
        """/copy: the model's last answer to the clipboard."""
        answer = next((s.get("content") for s in reversed(self.steps_history)
                       if isinstance(s, dict) and s.get("role") == "assistant" and not s.get("tool_calls")
                       and isinstance(s.get("content"), str) and s["content"].strip()), None)
        if not answer:
            self._append("No answer to copy yet.", msg_type="error")
            return
        _copy_to_clipboard(self, answer)
        self._flash_input_hint(f"copied last answer ({len(answer)} characters)")

    # ── /config: edit a settings file in place ─────────────────────────────────

    def _config_command(self, arg: str) -> None:
        """/config: pick config.yaml or the MCP servers, user or project; /config [mcp] [project] opens one."""
        words = arg.lower().split()
        if words:
            unknown = [w for w in words if w not in ("mcp", "config", "user", "project")]
            if unknown:
                self._append("Usage: /config [mcp] [project]", msg_type="error")
                return
            kind = "mcp" if "mcp" in words else "config"
            scope = "project" if "project" in words else "user"
            self._start_worker(self._open_settings(f"{kind}:{scope}"), name="config")
            return
        targets: list[tuple[SettingsKind, Scope, str]] = [
            ("config", "user", "config.yaml: user, all projects"),
            ("config", "project", "config.yaml: this project (overrides the user file key by key)"),
            ("mcp", "user", "MCP servers: user, all projects"),
            ("mcp", "project", "MCP servers: this project"),
        ]
        if mcp_path("user") == mcp_path("project"):  # mcp.config_path names one file for both
            targets = targets[:2] + [("mcp", "user", "MCP servers (mcp.config_path)")]
        options = []
        for kind, scope, label in targets:
            path = settings_file(kind, scope)
            missing = "" if path.is_file() else ", not created yet"
            options.append((f"{kind}:{scope}", f"{label}  ({str(path).replace(str(Path.home()), '~')}{missing})"))
        self._show_picker("Edit which settings file? (API keys stay in .env files)", options, self._open_settings)

    async def _open_settings(self, target: str) -> None:
        """target: "config:user", "mcp:project", "memory:project", … (the /config and /memory pickers' values)."""
        name = target.split(":", 1)[0]
        kind: EditorKind = name if name in ("mcp", "memory") else "config"  # type: ignore[assignment]
        scope: Scope = "project" if target.endswith(":project") else "user"
        path = self._editor_path(kind, scope)
        try:
            if path.is_file():
                text = await asyncio.to_thread(path.read_text, encoding="utf-8")
            else:
                text = "" if kind == "memory" else new_settings_text(kind, scope)
        except OSError as e:
            self._append(f"Could not read {path}: {e}", msg_type="error")
            return
        self._close_settings(force=True)
        self.query_one("#command-menu", CommandMenu).display = False
        self.query_one("#conversation").display = False
        self.query_one("#input", EnterSubmits).display = False
        shown = str(path).replace(str(Path.home()), "~")
        editor = SettingsEditor(text.replace("\r\n", "\n"), kind, scope, id="settings-editor")
        panel = Vertical(
            Static(Text.assemble((shown, "bold"), ("" if path.is_file() else "  (new file)", "dim")), id="settings-title"),
            editor,
            Static("Ctrl+S save and apply · Esc close", id="settings-status"),
            id="settings-panel",
        )
        await self.mount(panel, before="#input")
        editor.focus()

    @staticmethod
    def _editor_path(kind: EditorKind, scope: Scope) -> Path:
        if kind == "memory":
            return memory_file("global" if scope == "user" else "project")
        return settings_file(kind, scope)

    def _reload_instructions(self) -> None:
        """Rebuild the system prompt, e.g. after an AGENTS.md change, so the next message sees it."""
        self._system_instructions, _ = build_base_instructions(interactive=True)

    # ── /memory and `# note`: the AGENTS.md instruction files ──────────────────

    def _memory_targets(self) -> list[tuple[str, str]]:
        options = []
        for scope, label in (("project", "Project: this directory"), ("global", "Global: all projects")):
            path = memory_file(scope)  # type: ignore[arg-type]
            missing = "" if path.is_file() else ", not created yet"
            options.append((scope, f"{label}  ({str(path).replace(str(Path.home()), '~')}{missing})"))
        return options

    def _memory_command(self, arg: str) -> None:
        """/memory: pick the project or global AGENTS.md; /memory project|global opens one in the editor."""
        scope = {"project": "project", "global": "user", "user": "user"}.get(arg.lower())
        if arg and scope is None:
            self._append("Usage: /memory [project|global]", msg_type="error")
            return
        if scope:
            self._start_worker(self._open_settings(f"memory:{scope}"), name="memory")
            return

        async def choose(value: str) -> None:
            await self._open_settings(f"memory:{'user' if value == 'global' else 'project'}")
        self._show_picker("Edit which instructions file?", self._memory_targets(), choose)

    def _quick_memory(self, note: str) -> None:
        """`# note`: ask project or global, then append the note to that AGENTS.md."""
        async def save(scope: str) -> None:
            path = memory_file(scope)  # type: ignore[arg-type]
            try:
                await asyncio.to_thread(append_memory, path, note)
            except OSError as e:
                self._append(f"Could not save to {path}: {e}", msg_type="error")
                return
            self._reload_instructions()
            self._append(f"Saved to {str(path).replace(str(Path.home()), '~')}: - {note}  (/memory edits it)",
                         msg_type="system")
        self._show_picker(f"Save \"{note[:60]}\" to which instructions file?", self._memory_targets(), save)

    def _settings_open(self) -> bool:
        return bool(self.query("#settings-panel"))

    def _settings_status(self, text: str, error: bool = False) -> None:
        status = self.query_one("#settings-status", Static)
        status.update(text)
        status.set_class(error, "error")

    def on_settings_editor_save(self, event: SettingsEditor.Save) -> None:
        editor = self.query_one("#settings-editor", SettingsEditor)
        if not editor.modified:
            self._close_settings()
            return
        try:
            if editor.kind == "memory":
                path = self._editor_path("memory", editor.scope)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(editor.text, encoding="utf-8")
                self._reload_instructions()
                summary = f"Saved {str(path).replace(str(Path.home()), '~')}; the agent follows it from the next message."
            else:
                summary = save_settings_file(editor.kind, editor.scope, editor.text, adapter=self.adapter)
        except (ValueError, OSError) as e:
            self._settings_status(f"Not saved: {e}", error=True)
            return
        editor.saved_text = editor.text
        self._close_settings()
        note = "  (tracing changes apply after a restart)" if editor.kind == "config" else ""
        self._append(summary + note, msg_type="system")

    def _close_settings(self, force: bool = False) -> None:
        """Close the /config editor; with unsaved changes the first call only warns (ESC twice discards)."""
        editors = list(self.query(SettingsEditor))
        if not editors:
            return
        editor = editors[0]
        if editor.modified and not force and not editor.discard_armed:
            editor.discard_armed = True
            self._settings_status("Unsaved changes: Esc again to discard them, Ctrl+S to save", error=True)
            return
        for widget in self.query("#settings-panel"):
            widget.remove()
        self.query_one("#conversation").display = True
        inp = self.query_one("#input", EnterSubmits)
        inp.display = True
        inp.focus()

    async def _show_status(self) -> None:
        if not await self._wait_for_backend() or self.mcp_client is None:
            return
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
            *([("Worktree", f"{self.worktree.name}  (branch {self.worktree.branch}, from {self.worktree.repo_root})")]
              if self.worktree else []),
            ("Model", get_model() + (f"  ⚠ missing {', '.join(missing)}" if missing else "")),
            ("Permissions", mode + (f"  ({len(self.permissions.approved)} 'always' approval(s) this session)" if self.permissions.approved else "")),
            ("Tokens", f"↑{self._fmt_tokens(usage.get('prompt_tokens', 0))} ↓{self._fmt_tokens(usage.get('completion_tokens', 0))}"
                       + (f"  ({self._fmt_tokens(usage['cached_tokens'])} cached)" if usage.get("cached_tokens") else "")),
            ("Cost", format_cost(self.adapter._session_cost) if self.adapter._session_cost
                     else "n/a (nothing priced yet, or litellm has no price for this model)"),
            ("Context", f"{self._fmt_tokens(last_input)} / {self._fmt_tokens(token_limit)}{percent}"),
            ("Auto-compact", (f"at {t}% of the window" if (t := self.config.compact_threshold) else "off")
             + ("  (this session; /compact auto)" if self.config.session_compact_threshold else "")),
            ("Notifications", (f"{self._notify_kind} after {self.config.notify_after_seconds:g}s or in the background"
                               if self._notify_kind != "off" else "off") + "  (/notifications)"),
            ("Steps", str(len(self.steps_history))),
            ("MCP", f"{len(connected)}/{len(self._mcp_configured)} connected" + (f": {', '.join(connected)}" if connected else "")),
            ("Status bar", (", ".join(self.config.status_bar_items) or "(empty)") + "  (ui.status_bar in config.yaml)"),
            ("Version", __version__),
        ]
        width = max(len(k) for k, _ in rows) + 2
        self._append("\n".join(f"  {k.ljust(width)}{v}" for k, v in rows), msg_type="system")

    async def _show_context(self) -> None:
        """/context: estimated tokens per part of the next request (system prompt, tools, messages...)."""
        if not await self._wait_for_backend():
            return
        from dostuff.helpers.agent.environment_context import build_environment_block
        from dostuff.helpers.agent.load_project_instructions import load_project_instructions
        from dostuff.lib.model import get_litellm_model
        from dostuff.tools.definitions import TOOL_SCHEMAS, default_tool_names
        try:
            meta = await self.store.get_session_meta(self.session_id) or {}
            messages = await self.store.load_working_history(self.session_id)
            model = get_litellm_model()

            def measure() -> list[tuple[str, int]]:
                return context_parts(
                    self._system_instructions + "\n\n" + build_environment_block(),
                    load_project_instructions(),
                    {name: TOOL_SCHEMAS[name] for name in default_tool_names()},
                    messages,
                    meta.get("compaction_notes") or "",
                    lambda text: count_tokens(text, model),
                )
            parts = await asyncio.to_thread(measure)
            text = render_context(parts, get_model_token_limit(), get_model(), meta.get("last_input_tokens") or 0)
        except Exception as e:
            self._append(f"Could not measure the context: {e}", msg_type="error")
            return
        self._mount_message(Static(text, classes="msg msg-system", markup=False))

    async def _show_doctor(self) -> None:
        """/doctor: the `dostuff doctor` checks, ✓ / ✗ with a fix for each problem."""
        try:
            checks = await asyncio.to_thread(doctor_checks)
        except Exception as e:
            self._append(f"Doctor failed: {e}", msg_type="error")
            return
        text = Text(f"dostuff {__version__}\n")
        for check in checks:
            if check.ok is None:
                text.append(f"  - {check.label}\n", style="#8a8a8a")
                continue
            text.append(f"  {'✓' if check.ok else '✗'} {check.label}\n", style="#98c379" if check.ok else "#e06c75")
            if not check.ok and check.fix:
                text.append(f"      → {check.fix}\n", style="#e5c07b")
        problems = sum(1 for c in checks if c.ok is False)
        text.append(f"{problems} problem(s) found." if problems else "All good.",
                    style="#e06c75" if problems else "#98c379")
        self._mount_message(Static(text, classes="msg msg-system", markup=False))

    async def _show_hooks(self, reload: bool = False) -> None:
        """/hooks: what loaded from which file and whether it will run; `/hooks r` re-reads config first."""
        try:
            if reload:
                self.config = get_config()  # the instance the hook engine reads
                await asyncio.to_thread(self.config.reload)
                self._apply_settings(mcp=False)
            text = await asyncio.to_thread(hooks_listing)
        except Exception as e:
            self._append(f"Could not list hooks: {e}", msg_type="error")
            return
        if reload:
            self._append("Reloaded config.yaml (user and project).", msg_type="system")
        self._append(text, msg_type="system")

    async def _show_tools(self) -> None:
        """/tools: the agent's tools, how each asks, whether subagents get it, and the MCP servers' tools."""
        if not await self._wait_for_backend():
            return
        try:
            text = tools_listing(self.mcp_client.mcp_tools if self.mcp_client is not None else None)
        except Exception as e:
            self._append(f"Could not list tools: {e}", msg_type="error")
            return
        self._mount_message(Static(text, classes="msg msg-system", markup=False))

    def _session_jobs(self) -> list[Job]:
        """This session's background jobs, main agent and subagents (foreground calls and `!command`s aren't)."""
        return [j for j in get_registry().jobs() if not j.foreground and session_of(j.owner) == self.session_id]

    def _find_job(self, job_id: str) -> Job:
        """A job by id; the main agent's wins when a subagent used the same id."""
        matches = [j for j in self._session_jobs() if j.id == job_id.casefold()]
        if not matches:
            raise JobError(f'No job "{job_id}". Type /jobs to list them.')
        return next((j for j in matches if j.owner == self.session_id), matches[-1])

    async def _jobs_command(self, arg: str) -> None:
        """/jobs: the session's jobs; `/jobs <id>`: its recent output and logs; `/jobs kill <id>`: stop it."""
        registry = get_registry()
        words = arg.split()
        if not words:
            jobs = self._session_jobs()
            if not jobs:
                self._append("No background jobs.", msg_type="system")
                return
            rows = [("ID", "OWNER", "STATUS", "RUNTIME", "COMMAND")] + [
                (j.id, registry.owner_label(j.owner), _job_status(j), fmt_runtime(j.runtime),
                 " ".join(j.command.split())) for j in jobs]
            widths = [max(len(r[i]) for r in rows) for i in range(4)]
            lines = ["  ".join(c.ljust(w) for c, w in zip(r[:4], widths)) + "  " + r[4] for r in rows]
            self._append("\n".join(lines) + "\n\n/jobs <id> shows its output · /jobs kill <id> stops it",
                         msg_type="system")
            return
        if len(words) > 2 or (words[0] == "kill") != (len(words) == 2):
            self._append("Usage: /jobs [<id> | kill <id>]", msg_type="error")
            return
        try:
            job = self._find_job(words[-1])
        except JobError as e:
            self._append(str(e), msg_type="error")
            return
        if words[0] == "kill":
            await self._kill_job(job)
        else:
            self._show_job_output(job)

    async def _agents_command(self, arg: str) -> None:
        """/agents: this session's agents, to read, stop or answer; /agents types: the agent types;
        /agents reload: re-read the agent files (subagents spec D2, D14)."""
        from dostuff.agent.subagents import definitions as agent_defs
        word, _, name = arg.strip().partition(" ")
        word, name = word.lower(), name.strip()
        if word not in ("", "types", "reload", "r") or (name and word != "types"):
            self._append("Usage: /agents [types [name] | reload]", msg_type="error")
            return
        if not word:
            self._show_agents_list()
            return
        if word in ("reload", "r"):
            loaded = await asyncio.to_thread(agent_defs.reload)
            note = f" · {len(loaded.warnings)} warning(s), see /agents types" if loaded.warnings else ""
            self._append(f"Reloaded: {len(loaded.types)} agent types{note}.", msg_type="system")
            return
        if name:
            await self._show_agent_type(name)
            return
        loaded = await agent_defs.refresh_definitions()
        self._append(agent_defs.types_listing(loaded, get_config()), msg_type="system")

    async def _show_agent_type(self, name: str) -> None:
        """One agent type's config and system prompt: `/agents types <name>`, or "Show its type" on an agent."""
        from dostuff.agent.subagents import definitions as agent_defs
        from dostuff.agent.subagents.fork import FORK_TYPE
        loaded = await agent_defs.refresh_definitions()
        detail = agent_defs.type_detail(name, loaded, get_config())
        if detail is None:
            known = ", ".join(sorted([*loaded.types, FORK_TYPE]))
            self._append(f"No agent type '{name}'. Types: {known}.", msg_type="error")
            return
        self._append(detail, msg_type="system")

    def _session_agents(self) -> list["AgentRun"]:
        """This session's agents: the ones that haven't ended first, then by start time."""
        from dostuff.agent.subagents.registry import ENDED_STATUSES
        return sorted(_agent_registry().runs(self.session_id),
                      key=lambda r: (r.status in ENDED_STATUSES, r.started or float("inf")))

    def _show_agents_list(self) -> None:
        """/agents (or a click on the agents item): pick one of this session's agents."""
        runs = self._session_agents()
        if not runs:
            self._append("No agents in this session.", msg_type="system")
            return
        self._show_picker("Agents", [(r.id, _agent_label(r)) for r in runs], self._agent_actions)

    async def _agent_actions(self, agent_id: str) -> None:
        """An agent picked from /agents: its transcript, then Answer approval / Stop it / back."""
        run = _agent_registry().get(agent_id)
        if run is None or run.owner != self.session_id:
            return
        self._close_picker()
        await self._open_agent_view(run)
        options = []
        if run.status == "waiting_approval":
            options.append(("answer", "Answer approval (show its prompt now)"))
        if run.task is not None and not run.task.done():
            options.append(("stop", "Stop it (the agent's partial work comes as a notice)"))
        options.append(("type", f"Show its type ({run.agent_type}): system prompt and config"))
        options.append(("back", "Back to the list"))

        async def act(choice: str) -> None:
            if choice == "answer":
                run.answer_now = True  # its prompt shows once this picker closes, even while you type
                self._close_picker()
                self._close_agent_view()
            elif choice == "stop":
                self._close_picker()
                self._close_agent_view()
                await _agent_registry().stop(run, "user")  # a notice without a wake-up (D7)
                self._append(f'Stopped agent "{run.id}" ({run.description}).', msg_type="system")
            elif choice == "type":
                self._close_picker()
                self._close_agent_view()
                await self._show_agent_type(run.agent_type)
            else:
                self._close_agent_view()
                self._show_agents_list()

        self._show_picker(f'Agent "{run.id}" · {run.description} · {_agent_status(run)}', options, act)

    # ── The agent view: one agent's transcript, in place of the conversation ──

    def _agent_view_open(self) -> bool:
        return bool(self.query("#agent-view"))

    async def _agent_steps(self, run: "AgentRun") -> list[dict]:
        """The agent's whole transcript from the store (compacted steps too), else what it has in memory."""
        steps: list[dict] = []
        try:
            await _agent_registry().flush()
            steps = await self.store.load_agent_steps(run.id)
        except Exception:
            pass
        return steps or list(run.history)

    def _agent_view_title(self, run: "AgentRun") -> str:
        return (f'agent "{run.id}" · {run.description} · {run.agent_type} · {_agent_status(run)} · '
                f"{self._agent_view_rendered} steps")

    async def _open_agent_view(self, run: "AgentRun") -> None:
        """Show the agent's transcript in its own panel, hiding the conversation (like /config's editor), so the
        agent's steps never land in the main conversation. A running agent's new steps appear as they're saved."""
        self._close_agent_view()
        self.query_one("#command-menu", CommandMenu).display = False
        self.query_one("#conversation").display = False
        panel = Vertical(
            Static("", id="agent-view-title", markup=False),
            VerticalScroll(Vertical(id="agent-view-msgs"), id="agent-view-scroll", can_focus=False),
            Static("Esc close · Ctrl+O expand results · /agents for the list", id="agent-view-status"),
            id="agent-view",
        )
        await self.mount(panel, before="#queue-bar")
        self._agent_view_run = run
        self._agent_view_rendered = 0
        await self._refresh_agent_view()
        self.query_one("#agent-view-scroll", VerticalScroll).anchor()
        self._agent_view_timer = self.set_interval(1.0, self._refresh_agent_view)

    async def _refresh_agent_view(self) -> None:
        """Render the steps saved since the last refresh; stop polling once the agent has ended."""
        run = self._agent_view_run
        if run is None or self._agent_view_refreshing or not self._agent_view_open():
            return
        from dostuff.agent.subagents.registry import ENDED_STATUSES
        self._agent_view_refreshing = True
        try:
            ended = run.status in ENDED_STATUSES
            steps = await self._agent_steps(run)
            if self._agent_view_run is not run or not self._agent_view_open():
                return
            new = steps[self._agent_view_rendered:]
            show_result = ended and run.result and not steps and self._agent_view_rendered == 0
            if new or show_result:
                self._message_target = "#agent-view-msgs"
                try:
                    for step in new:
                        self._render_history_step(step)
                    if show_result:
                        self._append(run.result or "", msg_type="agent")
                finally:
                    self._message_target = "#msg-container"
                self._agent_view_rendered = len(steps) or (1 if show_result else 0)
            self.query_one("#agent-view-title", Static).update(self._agent_view_title(run))
            if ended and self._agent_view_timer is not None:
                self._agent_view_timer.stop()
                self._agent_view_timer = None
        finally:
            self._agent_view_refreshing = False

    def _close_agent_view(self) -> None:
        if self._agent_view_timer is not None:
            self._agent_view_timer.stop()
            self._agent_view_timer = None
        self._agent_view_run = None
        if not self._agent_view_open():
            return
        for widget in self.query("#agent-view"):
            widget.remove()
        self.query_one("#conversation").display = True

    async def _kill_job(self, job: Job) -> None:
        if job.ended is not None:
            self._append(f'Job "{job.id}" already {_job_status(job)}.', msg_type="system")
            return
        await get_registry().kill(job, "user")  # the owner gets a notice on its next model call (D7)
        self._append(f'Killed job "{job.id}" (pid {job.pid}); the agent will be told.', msg_type="system")

    def _show_job_output(self, job: Job) -> None:
        """`/jobs <id>`: the job's last 40 lines of each stream and its log files."""
        registry = get_registry()
        parts = [f'⚙ job "{job.id}" · {registry.owner_label(job.owner)} · {_job_status(job)} · '
                 f"{fmt_runtime(job.runtime)} · pid {job.pid}", f"$ {job.command}"]
        for stream in ("stdout", "stderr"):
            lines = job.tail(stream, 40)  # type: ignore[arg-type]
            parts.append(f"{stream} (last {len(lines)} lines):" if lines else f"{stream}: (none)")
            parts += [f"  {line}" for line in lines]
        parts.append(f"Logs: {job.out_path}\n      {job.err_path}")
        self._append("\n".join(parts), msg_type="system")

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

    async def _memory_ready(self) -> bool:
        """Wait for the stores; False (after saying so) when long-term memory is off."""
        if not await self._wait_for_backend():
            return False
        if self.memory_store is None or self.episodic_store is None:
            from dostuff.lib.memory.chroma import MEMORY_INSTALL_HINT
            self._append(f"Long-term memory is off (chromadb isn't installed). Turn it on with `{MEMORY_INSTALL_HINT}`.", msg_type="system")
            return False
        return True

    async def _show_memories(self) -> None:
        if not await self._memory_ready():
            return
        assert self.memory_store is not None
        try:
            facts = await self.memory_store.list_all(self.user_id)
        except Exception as e:
            self._append(f"Could not load memories: {e}", msg_type="error")
            return
        if not facts:
            self._append("No saved memories.", msg_type="system")
            return
        lines = [f"- {f['key']}: {f['value']}  ({f['source']})" for f in facts]
        self._append(f"Saved memories ({len(facts)}):\n" + "\n".join(lines) + "\n\nForget one with /forget <key>.", msg_type="system")

    def _episodes_dir(self) -> str | None:
        """The directory episodes are recalled from (None = all), as `memory.episodic_scope` says."""
        return str(Path.cwd().resolve()) if self.config.episodic_scope == "project" else None

    async def _show_episodes(self) -> None:
        if not await self._memory_ready():
            return
        assert self.episodic_store is not None
        working_dir = self._episodes_dir()
        try:
            episodes = await self.episodic_store.list_all(self.user_id, working_dir=working_dir)
        except Exception as e:
            self._append(f"Could not load episodes: {e}", msg_type="error")
            return
        where = "for this directory" if working_dir else "in any directory"
        if not episodes:
            self._append(f"No saved episodes {where}.", msg_type="system")
            return
        # The id's random tail is enough for /forget
        lines = [f"- [{ep['event_type']}] {ep['anchor_event']}  ({ep['created_at_iso'][:10]}, {ep['id'].rsplit('_', 1)[-1]})"
                 for ep in episodes]
        self._append(f"Saved episodes {where} ({len(episodes)}), newest first:\n" + "\n".join(lines)
                     + "\n\nForget one with /forget <id>.", msg_type="system")

    async def _forget(self, target: str) -> None:
        """Delete the fact with this key, else the episode with this id (or its last part)."""
        if not await self._memory_ready():
            return
        assert self.memory_store is not None and self.episodic_store is not None
        try:
            if await self.memory_store.delete(self.user_id, target):
                self._append(f"Forgot memory '{target}'.", msg_type="system")
                return
            episodes = await self.episodic_store.list_all(self.user_id)
            matches = [ep for ep in episodes if ep["id"] == target or ep["id"].endswith(f"_{target}")]
            if len(matches) > 1:
                self._append(f"'{target}' matches {len(matches)} episodes; use the full id.", msg_type="error")
                return
            if matches and await self.episodic_store.delete(self.user_id, matches[0]["id"]):
                self._append(f"Forgot episode [{matches[0]['event_type']}] {matches[0]['anchor_event']}", msg_type="system")
                return
        except Exception as e:
            self._append(f"Could not forget '{target}': {e}", msg_type="error")
            return
        self._append(f"No memory key or episode id '{target}' (see /memories and /episodes).", msg_type="error")

    def on_enter_submits_submitted(self, event: EnterSubmits.Submitted) -> None:
        text = event.text
        # A pending approval or question consumes the input (empty Enter after "No" = deny without feedback)
        if self.adapter.has_pending_prompt():
            self.adapter.got_input(self._pastes.inline(text))
            self._clear_input()
            return
        if not text:
            return

        if self._agent_view_open():  # sending anything goes back to the conversation, where its output shows
            self._close_agent_view()
        self._clear_input()
        self._remember_prompt(text)
        cmd = text.strip().lower()

        if cmd.split(maxsplit=1)[:1] == ["/btw"]:  # works mid-turn: never queued (btw spec D1, CON-006)
            self._btw_command(self._pastes.inline(text.strip()[len("/btw"):]).strip())
            return
        if cmd.split(maxsplit=1)[:1] == ["/steer"]:  # a mid-turn correction, parsed ahead of the guards (steering D2)
            self._steer_command(text.strip()[len("/steer"):])
            return
        if text.lstrip().startswith("!"):
            command = self._pastes.inline(text.lstrip()[1:]).strip()
            if not command:
                return
            if self._turn_in_progress:
                self._append("Can't run a shell command while a turn is running (press ESC to cancel it first).", msg_type="error")
                return
            self._start_worker(self._run_shell_command(command), name="shell")
            return
        note = quick_memory_text(text)
        if note is not None:
            self._quick_memory(note)
            return
        if cmd == "/exit":
            self.run_worker(self.action_quit(), name="exit_worker", exit_on_error=False)
            return
        if (cmd in ("/clear", "/compact") or (cmd.startswith("/compact ") and cmd.split()[1:2] != ["auto"])) \
                and self._turn_in_progress:
            self._append(f"Can't run {cmd} while a turn is running (press ESC to cancel it first).", msg_type="error")
            return
        if cmd == "/clear":
            self._start_worker(self._clear_session(), name="clear")
            return
        if cmd == "/compact":
            self._start_worker(self._compact_session(), name="compact")
            return
        if cmd.split()[:2] == ["/compact", "auto"]:
            self._compact_auto_command(" ".join(cmd.split()[2:]))
            return
        if cmd == "/memories":
            self._start_worker(self._show_memories(), name="memories")
            return
        name, _, arg = text.strip().partition(" ")
        name, arg = name.lower(), arg.strip()
        if name in ("/resume", "/model", "/fork", "/worktree", "/config", "/memory") and self._turn_in_progress:
            self._append(f"Can't run {name} while a turn is running (press ESC to cancel it first).", msg_type="error")
            return
        if name == "/resume":
            self._start_worker(self._resume_command(arg), name="resume")
            return
        if name == "/fork":
            self._start_worker(self._fork_session(arg), name="fork")
            return
        if name == "/worktree":
            self._start_worker(self._worktree_command(arg), name="worktree")
            return
        if name == "/model":
            self._model_command(arg)
            return
        if name == "/permissions":
            self._permissions_command(arg)
            return
        if name == "/copy" and not arg:
            self._copy_last_answer()
            return
        if name == "/notifications":
            self._notifications_command(arg.lower())
            return
        if name == "/config":
            self._config_command(arg)
            return
        if name == "/memory":
            self._memory_command(arg)
            return
        if name == "/context" and not arg:
            self._start_worker(self._show_context(), name="context")
            return
        if name == "/doctor" and not arg:
            self._start_worker(self._show_doctor(), name="doctor")
            return
        if name == "/tools" and not arg:
            self._start_worker(self._show_tools(), name="tools")
            return
        if name == "/hooks":
            if arg.lower() not in ("", "r", "reload"):
                self._append("Usage: /hooks [r]  (r re-reads the config files)", msg_type="error")
            else:
                self._start_worker(self._show_hooks(reload=bool(arg)), name="hooks")
            return
        if name == "/jobs":
            self._start_worker(self._jobs_command(arg), name="jobs")
            return
        if name == "/agents":
            self._start_worker(self._agents_command(arg), name="agents")
            return
        if name == "/queue":
            if arg.lower() == "clear":
                self._clear_queue()
            elif arg:
                self._append("Usage: /queue [clear]", msg_type="error")
            else:
                self.action_show_queue()
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
        if name == "/episodes" and not arg:
            self._start_worker(self._show_episodes(), name="episodes")
            return
        if name == "/forget":
            if not arg:
                self._append("Usage: /forget <memory key or episode id>", msg_type="error")
            else:
                self._start_worker(self._forget(arg), name="forget")
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
            custom = "".join(f"\n  {c}  ({_rel_path(s['location'])})" for c, s in self._skill_commands.items()
                             if s.get("kind") == "command")
            if custom:
                custom = f"\n\nCustom commands (/<name> [arguments]; $ARGUMENTS in the file is replaced):{custom}"
            skills = "".join(f"\n  {c}" for c, s in self._skill_commands.items() if s.get("kind") != "command")
            if skills:
                skills = f"\n\nSkills (/<skill> [arguments] runs one):{skills}"
            self._append(
                f"Commands:\n{commands}{custom}{skills}\n\n"
                "  !<command>   Run a shell command directly; its output is added to the conversation\n"
                "  # <note>     Add a line to AGENTS.md (project or global), which the agent follows\n\n"
                "Shortcuts: ESC cancel turn · Ctrl+Q quit · Shift+Enter (or \\ then Enter) newline · "
                "Shift+Tab cycle permission mode (ask → auto → plan → allowlist) · "
                "Ctrl+Backspace / Ctrl+W delete word · Ctrl+Delete delete next word · "
                "Ctrl+O expand/collapse tool output (or click one) · Ctrl+R search prompt history · "
                "Esc Esc (empty input) edit the previous prompt · / command menu",
                msg_type="system",
            )
            return
        # A lone "/word" is a mistyped command; paths like /etc/hosts still go to the model
        if cmd.startswith("/") and " " not in cmd and cmd.count("/") == 1 and cmd not in COMMAND_DESCRIPTIONS \
                and cmd not in self._skill_commands:
            self._append(f"Unknown command {cmd}. Type / to see commands.", msg_type="error")
            return

        # User message — only append if starting a new turn; queued messages append when they run
        if not self._turn_in_progress:
            self._append(text, msg_type="user")
            self._begin_turn(text)
        elif self.config.mid_turn_messages == "steer" and parse_skill_command(text, self._skill_commands) is None:
            # `steer` mode (D3, REQ-3.2): a plain message corrects the turn at its next tool boundary.
            # A skill or custom command wraps the text in SKILL.md and announces itself — that is a turn's
            # prompt, not a correction, so it still waits in `_prompt_queue` (REQ-3.3) in both modes.
            self._steer_submit(text)
        else:
            self._prompt_queue.append(text)
            self._render_queue()

    # ── Mid-turn steering (/steer) ─────────────────────────────────────────────

    def _steer_command(self, arg: str) -> None:
        """`/steer <message>` (steering spec D2, REQ-3.1): a steering message mid-turn, an ordinary
        prompt when idle — the habit is never an error."""
        message = self._pastes.inline(arg).strip()
        if not message:
            self._append("Usage: /steer <message>", msg_type="error")
            return
        if self._turn_in_progress:
            self._steer_submit(message)
            return
        self._append(message, msg_type="user")
        self._begin_turn(message)

    def _steer_submit(self, message: str) -> None:
        """A message the running turn reads at the next tool boundary (D2/D3).

        The slot is reserved synchronously, so submit order holds no matter when preparation
        finishes (CON-006); nothing is written to the history here — the loop appends it once,
        at delivery (CON-004).
        """
        msg = self._steering.put(message)
        self._steer_raws[msg.id] = msg.raw
        self._render_queue()
        self._start_worker(self._prepare_steer(msg), name=f"steer_{msg.id[:6]}")

    def _on_steer_event(self, event_type: str, data: dict) -> None:
        """The loop's verdict on one steering message (steering spec D6, D9).

        `steered`: finished the stream, then the raw text as a user message — it lands after the output
        it was reacting to, where it was read — and is recorded at the delivered index, so Esc Esc puts
        back what the user typed rather than the expanded attachments. Blocked / failed: the reason or
        the error, never a message in the transcript (nothing was appended to the history either).
        """
        raw = self._steer_raws.pop(str(data.get("id")), None)
        if event_type == "steered":
            if raw is None:
                return
            self._finish_stream_widgets()
            self._append(raw, msg_type="user")
            self._sent_prompts[(self.session_id, int(data.get("index") or 0))] = raw
        elif event_type == "steer_blocked":
            self._append(blocked_prompt_message(str(data.get("reason") or "")), msg_type="system")
        else:  # steer_failed: preparation could not finish; nothing was delivered
            self._append(str(data.get("error") or "the message could not be prepared"), msg_type="error")
        self._render_queue()

    async def _prepare_steer(self, msg: SteerMessage) -> None:
        """D5: @mentions and pastes resolved off the UI thread, in submit order; `ready`/`fail` hands
        the entry to the loop. The "Attached …" lines show here, as they do for a prompt."""
        try:
            expansion = await asyncio.to_thread(expand_file_mentions, msg.raw, self.config.max_tool_output_chars)
            if expansion.attached:
                self._append("Attached " + ", ".join(expansion.attached), msg_type="system")
            if expansion.skipped:
                self._append("Not attached: " + "; ".join(expansion.skipped), msg_type="error")
            content = await asyncio.to_thread(self._pastes.resolve, expansion.content, self.session_id,
                                              self.config.paste_file_lines, self.config.max_tool_output_chars)
        except Exception as e:  # unreadable mention, broken paste file: reported, never delivered (REQ-2.4)
            self._steering.fail(msg.id, str(e))
            self._render_queue()
            return
        self._steering.ready(msg.id, content)
        self._render_queue()

    # ── Approval prompt ────────────────────────────────────────────────────────

    def _show_permission_prompt(self, question: str, options: list[tuple[str, str]]) -> None:
        """Swap the input for the approval selector until the prompt is answered."""
        self._hide_permission_prompt(restore_input=False)
        menu = self.query_one("#command-menu", CommandMenu)
        menu.display = False
        inp = self.query_one("#input", EnterSubmits)
        if inp.display:  # not when one prompt follows another
            self._viewport_before_prompt = self.query_one("#messages", VerticalScroll).size.height
        inp.display = False
        prompt = PermissionPrompt(question, options, id="permission-prompt")
        self.mount(prompt, before="#input")
        self.query_one("#messages", VerticalScroll).anchor()  # show the details being approved
        self.call_after_refresh(prompt.focus)

    # ── The question panel (ask-user D6) ───────────────────────────────────────

    def _ask_panel(self) -> AskUserPanel:
        return self.query_one("#ask-panel", AskUserPanel)

    def _show_ask_panel(self, questions: list[dict]) -> None:
        """Swap the input for the questions panel until the call is submitted or cancelled (D6)."""
        panel = self._ask_panel()
        self._close_picker()  # /resume-style pickers and the questions never share the input
        menu = self.query("#command-menu")
        if menu:
            menu.first().display = False
        inp = self.query_one("#input", EnterSubmits)
        if inp.display:
            self._viewport_before_prompt = self.query_one("#messages", VerticalScroll).size.height
        inp.display = False
        panel.configure(questions)
        panel.display = True
        # No anchor(): like _show_picker, the panel must not drag a transcript the user scrolled up back
        # to the bottom. `Transcript._get_scrollable_region` already keeps a bottom-pinned view flush (T4.6).
        self.call_after_refresh(panel.focus)

    def _hide_ask_panel(self) -> None:
        """Put the panel away; the input comes back through `_give_input_back` (idempotent when idle)."""
        panel = self.query("#ask-panel")
        if panel and panel.first().display:
            panel.first().display = False

    def _ask_panel_open(self) -> bool:
        panels = self.query("#ask-panel")
        return bool(panels) and panels.first().display

    def on_ask_user_panel_submit(self, event: AskUserPanel.Submit) -> None:
        self.adapter.got_ask_submit(event.answers)

    def on_ask_user_panel_needs_input(self, event: AskUserPanel.NeedsInput) -> None:
        """The `Other` row: the input takes the typed answer and the panel waits for it (D6/REQ-4.3)."""
        question = self._ask_panel().questions[event.index]
        self.adapter.begin_other(event.index)
        prompt = _one_line_answer(question.get("question", ""))
        if not (question.get("options") or []):
            # A free-text question's one row *is* "Other", so naming it in the placeholder would read
            # "Other (type your own): Name the app?" — the question alone is clearer.
            self._hand_input_over(prompt or "Type your answer (Enter sends, Esc cancels)")
            return
        self._hand_input_over(f"{ASK_OTHER_LABEL}: {prompt or 'Type your answer (Enter sends, Esc cancels)'}")

    def on_ask_user_panel_inert(self, event: AskUserPanel.Inert) -> None:
        panel = self._ask_panel()
        missing = [str(i + 1) for i in range(len(panel.questions)) if not panel.answered(i)]
        self._flash_input_hint(f"answer every question first ({', '.join(missing)})")

    def _ask_panel_other(self, index: int, text: str) -> None:
        """The typed `Other` answer came back: record it, give the input back and re-focus the panel."""
        panel = self._ask_panel()
        panel.record_other(index, text)
        self.adapter.end_other()
        self._give_input_back()  # the draft and the placeholder come back...
        self.query_one("#input", EnterSubmits).display = False  # ...but the panel still owns the screen
        panel.display = True
        panel.focus()

    def _ask_denial_feedback(self) -> None:
        """After "No": the input (yellow border) takes an optional note for the model."""
        self._hand_input_over("Tell the model what to do instead (Enter to send, empty Enter to just deny)")

    def _hand_input_over(self, placeholder: str) -> None:
        """Give the input to a prompt (ask-user D7): hide the prompt widget, yellow border, a new placeholder,
        and the draft set aside until `_give_input_back` restores it. Used by the denial flow and by a
        question, so the save/restore exists once."""
        if self.query("#permission-prompt") or self._ask_panel_open():
            self._hold_transcript()
        self._hide_permission_prompt(restore_input=False)
        self._hide_ask_panel()  # the "Other" row keeps its answers: only the widget leaves the screen
        menu = self.query("#command-menu")
        if menu:  # not every hand-over has the menu mounted (a bare adapter in a unit test)
            menu.first().display = False
        inp = self.query_one("#input", EnterSubmits)
        if not inp.has_class("feedback-mode"):
            self._draft_before_feedback = inp.text  # a second hand-over keeps the draft it was given
        inp.text = ""
        if hasattr(inp, "placeholder"):
            inp.placeholder = placeholder
        inp.add_class("feedback-mode")
        inp.display = True
        inp.focus()

    def _give_input_back(self) -> None:
        """The undo of `_hand_input_over` (and the approval prompt's restore): the draft comes back and the
        input is a prompt input again."""
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

    def _hold_transcript(self) -> None:
        """Called as an approval prompt gives its place back to the shorter input during a turn. The
        conversation grows by the difference, and pinned to the bottom, every message would drop by it and
        jump back up as the next lines arrive (the hook trust prompt at session start bounced so).
        Instead the container keeps its height (min-height) for the turn: the freed rows are blank space
        under the loader that the next messages fill, and _release_transcript drops it when the turn ends."""
        before, self._viewport_before_prompt = self._viewport_before_prompt, None
        if before is None or not self._turn_in_progress or self._agent_view_run is not None:
            return
        scroll = self.query_one("#messages", VerticalScroll)
        grow = before - scroll.size.height
        if grow <= 0 or scroll.max_scroll_y == 0 or not scroll.is_vertical_scroll_end:
            return  # nothing would move: it all fits, or the user scrolled up
        container = self.query_one("#msg-container", Vertical)
        container.styles.min_height = container.size.height + grow

    def _release_transcript(self) -> None:
        try:
            self.query_one("#msg-container", Vertical).styles.min_height = None  # back to the CSS min-height
        except Exception:
            pass  # shutting down

    def _hide_permission_prompt(self, restore_input: bool = True) -> None:
        prompts = list(self.query("#permission-prompt"))
        if prompts and restore_input:
            self._hold_transcript()
        for widget in prompts:
            widget.remove()
        if not restore_input:
            return
        self._give_input_back()

    def on_permission_prompt_chosen(self, event: PermissionPrompt.Chosen) -> None:
        if event.source_id == "picker":
            callback = self._picker_callback
            self._close_picker()
            if callback is not None:
                self._start_worker(callback(event.value), name="picker")
            return
        self.adapter.choose(event.value)

    def _show_picker(self, question: str, options: list[tuple[str, str]], on_choose: Callable[[str], Awaitable[None]],
                     cursor: int = 0, filterable: bool = False, detail: list[tuple[str, str]] | None = None) -> None:
        """A selector in place of the input (/resume, /model, /permissions); ESC closes it.

        One #picker is reused: mounting a new one while Textual is still removing the last
        (removal is asynchronous) raised DuplicateIds, and the leftover kept _picker_open() true."""
        self._picker_callback = on_choose
        menu = self.query_one("#command-menu", CommandMenu)
        menu.display = False
        self.query_one("#input", EnterSubmits).display = False
        hint = ("Type to filter · " if filterable else "") + "↑/↓ select · Enter confirm · Esc close"
        picker = self.query_one("#picker", PermissionPrompt)
        picker.configure(question, options, hint=hint, cursor=cursor, filterable=filterable, detail=detail)
        picker.display = True
        self.call_after_refresh(picker.focus)

    def _picker_open(self) -> bool:
        pickers = self.query("#picker")
        return bool(pickers) and pickers.first().display

    def _close_picker(self) -> None:
        self._picker_callback = None
        if not self._picker_open():
            return
        self.query_one("#picker", PermissionPrompt).display = False
        if self.query("#permission-prompt"):
            return  # an approval prompt owns the input area
        try:
            inp = self.query_one("#input", EnterSubmits)
        except Exception:
            return
        inp.display = True
        inp.focus()

    # ── /btw: a side question in an overlay (plan/2026-11-03-btw/spec.md) ──────────

    def _btw_panel(self) -> BtwPanel:
        return self.query_one("#btw-panel", BtwPanel)

    def _btw_open(self) -> bool:
        panels = self.query("#btw-panel")
        return bool(panels) and panels.first().display

    def _btw_in_flight(self) -> bool:
        return self._btw_worker is not None and not self._btw_worker.is_finished

    def _btw_command(self, question: str) -> None:
        if not question:
            self._append(BTW_USAGE, msg_type="error")
            return
        if self._btw_in_flight():
            self._flash_input_hint(BTW_BUSY)
            return
        self._show_btw(question)
        self._btw_worker = self._start_worker(self._ask_side_question(question), name="btw")

    def _show_btw(self, question: str) -> None:
        """Shown at once with `Asking…`, filled when the answer arrives (D6). Takes focus for its keys."""
        panel = self._btw_panel()
        panel.configure(question)
        panel.display = True
        # Not while a prompt is up: the prompt keeps the focus and the first Esc (D6)
        if not self.adapter.has_pending_prompt():
            self.call_after_refresh(panel.focus)

    def _close_btw(self) -> None:
        """Hide the side answer (an unanswered call is cancelled) and give the focus back."""
        if self._btw_in_flight() and self._btw_worker is not None:
            self._btw_worker.cancel()
        panels = self.query("#btw-panel")
        if not panels or not panels.first().display:
            return
        panel = panels.first()
        panel.display = False
        if self.screen.focused is panel:
            self.screen.set_focus(None)
        self._refocus_input()

    async def _side_snapshot(self, question: str) -> ForkContext:
        """What a side question sees (D3): the running turn's snapshot; idle, the last turn's prompt parts with the
        stored history and notes (an idle /compact may have changed them, tasks.md open point 2); with neither, a
        system prompt built now, which is cache-cold once."""
        live = live_side_snapshot(self.session_id)
        if live is not None:
            return live
        meta = await self.store.get_session_meta(self.session_id) or {}
        history = await self.store.load_working_history(self.session_id)
        notes = meta.get("compaction_notes") or ""
        kept = kept_prompt(self.session_id)
        if kept is not None:
            return ForkContext(system_instruction=kept.system_instruction, compaction_notes=notes, history=history,
                               tool_names=list(kept.tool_names), session_hook_context=kept.session_hook_context)
        from dostuff.tools.definitions import default_tool_names  # pulls in the tools: after the first frame
        return ForkContext(system_instruction=await self._build_instructions(question), compaction_notes=notes,
                           history=history, tool_names=default_tool_names(),
                           session_hook_context=dict(self.session_hook_context))

    async def _ask_side_question(self, question: str) -> None:
        """One no-tools call over the conversation; the answer goes to the panel and the side thread, nowhere
        else (CON-004). A failure is shown in the panel, never as a turn error."""
        panel = self._btw_panel()
        session_id = self.session_id
        try:
            await self._backend_ready.wait()
            if self._backend_error:
                raise RuntimeError(f"startup failed ({self._backend_error})")
            snap = await self._side_snapshot(question)
            panel.snapshot = snap
            answer = await answer_side_question(snap, list(self._side_exchanges), question,
                                                plan_mode_on(self.permissions, self.config))
        except asyncio.CancelledError:
            raise  # the panel was closed while asking
        except Exception as e:
            panel.show_error(f"Side question failed: {e}")
            return
        if self.session_id != session_id:
            return  # the session changed while asking: this answer belongs to the old thread
        add_exchange(self._side_exchanges, answer.sent_question, answer.answer)
        panel.show_answer(answer, self._format_side_usage(answer.usage))

    def _format_side_usage(self, usage: dict | None) -> str:
        """`↑1.2k (1.1k cached) ↓340 · $0.0020`, shown on the panel and stored nowhere (D9)."""
        if not usage:
            return ""
        text = f"↑{self._fmt_tokens(usage.get('prompt_tokens', 0))}"
        if usage.get("cached_tokens"):
            text += f" ({self._fmt_tokens(usage['cached_tokens'])} cached)"
        text += f" ↓{self._fmt_tokens(usage.get('completion_tokens', 0))}"
        if usage.get("cost"):
            text += f" · {format_cost(usage['cost'])}"
        return text

    def _clear_side_thread(self, old_session_id: str) -> None:
        """A session switch: the side thread and the panel belong to the old session, and so do its kept prompt
        parts (D10). A fork-out still running is a background run like any other (carried, or stranded by /fork)."""
        self._side_exchanges.clear()
        self._close_btw()
        discard_snapshot(old_session_id)

    def on_btw_panel_close(self, event: BtwPanel.Close) -> None:
        self._close_btw()

    def on_btw_panel_fork_out(self, event: BtwPanel.ForkOut) -> None:
        panel = self._btw_panel()
        if self._btw_forking or panel.answer is None:
            return
        self._btw_forking = True
        self._start_worker(self._btw_fork_out(panel.question, panel.answer.text, panel.snapshot), name="btw_fork")

    async def _btw_fork_out(self, question: str, answer: str, snap: ForkContext | None) -> None:
        """`f`: the same question, the answer to check and the *same* snapshot to a background `fork` agent. Its
        report comes back by the ordinary notice path, as a user step in the transcript (D8)."""
        panel = self._btw_panel()
        try:
            await _agent_registry().start(
                owner=self.session_id, agent_type="fork", background=True, description=BTW_FORK_DESCRIPTION,
                prompt=BTW_FORK_PROMPT.format(question=question, answer=answer), fork=snap,
                mcp_client=self.mcp_client, adapter=self.adapter, permissions=self.permissions,
            )
        except Exception as e:  # ValueError for a missing snapshot, anything else from start: the panel stays
            panel.show_fork_error(f"Could not hand it to a background agent: {e}")
            return
        finally:
            self._btw_forking = False
        self._close_btw()
        self._append(BTW_HANDED_OFF, msg_type="system")

    def on_btw_panel_copy(self, event: BtwPanel.Copy) -> None:
        answer = self._btw_panel().answer
        if answer is not None:
            _copy_to_clipboard(self, answer.text)
            self._flash_input_hint(f"copied side answer ({len(answer.text)} characters)")

    def _begin_turn(self, text: str) -> None:
        self._turn_in_progress = True
        self._wake_turns = 0
        self._start_worker(self._run_turn(text), name=f"turn_{uuid.uuid4().hex[:6]}")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Open the "/" menu while typing a command; tint the input border in "!" shell mode."""
        if isinstance(event.text_area, SettingsEditor):
            return
        text = event.text_area.text
        menu = self.query_one("#command-menu", CommandMenu)
        recalled = isinstance(event.text_area, EnterSubmits) and text == event.text_area.recalled_text
        partial = event.text_area.partial_mention() if isinstance(event.text_area, EnterSubmits) and not recalled else None
        if text.startswith("/") and " " not in text and "\n" not in text and not recalled:
            menu.show_matches(text)
        elif partial is not None and not text.lstrip().startswith("!"):
            menu.show_paths(mention_candidates(partial[1:]))
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
        self._set_title("running")
        self._turn_start_time = time.time()
        self._start_timer()
        self.adapter._cancelled = False
        self.adapter._cancel_event.clear()
        try:
            # Owner "user": not Ctrl+B-able, and nobody would receive its notices (spec OUT-002)
            task = asyncio.ensure_future(run_shell(command, timeout=SHELL_TIMEOUT_S, owner="user"))
            while not task.done():
                if self.adapter._cancel_event.is_set():
                    task.cancel()
                    await asyncio.wait({task})  # let it kill the process tree before checking
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
            self._append_tool_result(output, SHELL_PREVIEW_LINES, msg_type="shell-output")

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
            self._turn_start_time = None
            self._turn_in_progress = False
            self._stop_loader(loader_widget)
            self._set_title("done")
            self._process_prompt_queue()

    # ── One agent turn ─────────────────────────────────────────────────────────

    async def _build_instructions(self, user_text: str) -> str:
        return await build_turn_instructions(self._system_instructions, user_text, self.memory_store, self.episodic_store, self.user_id)

    async def _run_turn(self, user_text: str, wake: bool = False) -> None:
        """One agent turn. `wake`: started by a background job ending (D15) — `user_text` is the pending
        notice text, and no user step is stored: the loop drains the notice into history itself."""
        loader_widget: Static | None = None
        cancelled = False  # the loop's own verdict: `adapter._cancelled` is cleared before it returns
        try:
            # One loader for the whole turn, from the start: a "Loading model libraries..." loader removed and
            # a "Working..." one mounted later left a frame with neither (the session-start bounce)
            ready = self._backend_ready.is_set()
            loader_widget = self._show_loader(WORKING_LABEL if ready else LOADING_LIBRARIES_LABEL)
            if not await self._wait_for_backend(keep_loader=True):
                return
            turn_id = uuid.uuid4().hex
            turn_id_var.set(turn_id)
            session_id_var.set(self.session_id)  # /clear may have switched sessions since on_mount

            meta = await self.store.get_session_meta(self.session_id) or {}
            working_history = await self.store.load_working_history(self.session_id)

            content = user_text
            if not wake:
                # `/<skill> args`: only the args get @mentions and pastes; the SKILL.md text goes around them
                skill_run = parse_skill_command(user_text, self._skill_commands)
                expansion = await asyncio.to_thread(expand_file_mentions, skill_run[1] if skill_run else user_text,
                                                    self.config.max_tool_output_chars)
                if expansion.attached:
                    self._append("Attached " + ", ".join(expansion.attached), msg_type="system")
                if expansion.skipped:
                    self._append("Not attached: " + "; ".join(expansion.skipped), msg_type="error")
                # After the @mentions, so an "@" inside pasted text isn't taken as one
                content = await asyncio.to_thread(self._pastes.resolve, expansion.content, self.session_id,
                                                  self.config.paste_file_lines, self.config.max_tool_output_chars)
                if skill_run:
                    try:
                        content = await asyncio.to_thread(skill_prompt, skill_run[0], content)
                    except OSError as e:
                        self._append(f"Could not read /{skill_run[0]['name']}: {e}", msg_type="error")
                        return
                    what = "command" if skill_run[0].get("kind") == "command" else "skill"
                    self._append(f"Running {what} {skill_run[0]['name']} ({_rel_path(skill_run[0]['location'])})",
                                 msg_type="system")

            # Reset cancel state before the hooks below, so ESC can stop a slow one
            self.adapter._cancelled = False
            self.adapter._cancel_event.clear()
            # config.yaml edited outside edit_config since the last turn (edit_file, a shell, an editor): apply it
            # here so the status bar etc. follow; the hooks below would reload it anyway
            if await asyncio.to_thread(self.config.reload_if_changed):
                self._apply_settings(mcp=False)
            # The loader is up before the hooks: they can ask the project-hooks trust question, and a loader
            # mounted only after the answer shifts the transcript (tool approvals already have one on screen)
            loader_widget = self._show_loader(WORKING_LABEL)  # relabels the turn's loader
            # SessionStart, then UserPromptSubmit: both before the user step is appended (hooks spec D1)
            if self._pending_session_start:
                source, self._pending_session_start = self._pending_session_start, None
                await session_start(source, self.session_hook_context, session_id=self.session_id,
                                    turn_id=turn_id, adapter=self.adapter)
            prompt_context = ""
            if wake:
                # Nobody typed a prompt, so no UserPromptSubmit. A user turn may have taken the notices meanwhile.
                if not get_notice_queue().pending(self.session_id):
                    return
            else:
                blocked, prompt_context = await user_prompt_submit(content, session_id=self.session_id,
                                                                   turn_id=turn_id, adapter=self.adapter)
                if blocked is not None:
                    self._append(blocked_prompt_message(blocked), msg_type="system")
                    return

                self._sent_prompts[(self.session_id, len(self.steps_history))] = user_text
                await append_step(
                    {"role": "user", "content": content},
                    self.steps_history,
                    working_history,
                    self.current_session_history,
                    self.session_id,
                    self.store,
                    "interactive_loop",
                )

            # The hook context is for this turn only: it is not part of the stored user step
            dynamic_instructions = with_prompt_context(await self._build_instructions(user_text), prompt_context)

            self._set_title("running")
            self._turn_start_time = time.time()
            self._start_timer()
            self.adapter._turn_cost = 0.0
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
                session_hook_context=self.session_hook_context,
                # D1: the app's mid-turn queue, drained at the next tool boundary (main session only)
                steering=self._steering,
            )

            agent_text = result or "(no response)"
            cancelled = result == CANCELLED_TURN
            # A streamed final answer is already on screen
            if agent_text.strip() != self._last_streamed_answer.strip():
                self._append(agent_text, msg_type="agent")

        except Exception as e:
            self._append(f"Turn error: {e}", msg_type="error")
        finally:
            self._stop_timer()
            started, self._turn_start_time = self._turn_start_time, None
            self._turn_in_progress = False
            cancelled = cancelled or self.adapter._cancelled  # an Esc after the loop's last check
            # Leftovers (steering spec D8): whatever outlived the turn — sent after the loop's last D7
            # check, still being prepared, or the turn hit the cap — goes back to the front of the queue
            # now, so the cancel path below and the "turn finished" notification both see it.
            leftovers = self._steering.take_all()
            if leftovers:
                for msg in leftovers:
                    self._steer_raws.pop(msg.id, None)  # never delivered: its raw is only a prompt again
                # Front of `_prompt_queue`, in order — and ahead of the queued prompts on the way back in
                self._prompt_queue[0:0] = [msg.raw for msg in leftovers]
            self._finish_stream_widgets()
            self._stop_loader(loader_widget)
            if started is not None:
                self._append_turn_summary(time.time() - started, cancelled)
                self._set_title("done")
                if not cancelled and not self._prompt_queue:
                    self._notify("dostuff: turn finished", time.time() - started)
            self._release_transcript()  # after the summary, in the same refresh, so it fills the held rows
            self._render_status()  # session cost changed
            if leftovers and not cancelled:
                # The cancel path says it its own way ("Moved N queued prompts back to the input")
                count = len(leftovers)
                self._append(f"{count} message{'s' if count != 1 else ''} arrived after the turn ended; "
                             f"running {'it' if count == 1 else 'them'} next", msg_type="system")
            self._process_prompt_queue(cancelled)
            self._schedule_wake()  # a job that ended after the loop's last drain

    def _append_turn_summary(self, elapsed: float, cancelled: bool = False) -> None:
        """`✻ Worked for 1m 5s · $0.01 · done 12:59 PM` under a finished turn.

        `cancelled` is the turn's own outcome, not `adapter._cancelled`: `loop_check_cancelled` clears that
        flag before `loop()` returns, so at summary time it has always been False already.
        """
        verb = "Cancelled after" if cancelled else "Worked for"
        parts = [f"✻ {verb} {_fmt_elapsed(elapsed)}"]
        if self.adapter._turn_cost:
            parts.append(format_cost(self.adapter._turn_cost))
        parts.append("done " + time.strftime("%I:%M %p").lstrip("0"))
        self._append(" · ".join(parts), msg_type="system")

    def _process_prompt_queue(self, cancelled: bool = False) -> None:
        """Start the next queued prompt, if any. After Esc the queue goes back in the input instead:
        cancelling stops the work, not just the turn that happened to be running.

        `cancelled` is passed in for the same reason as in `_append_turn_summary`: the adapter flag has
        already been cleared by the loop by the time the queue is looked at.
        """
        if not self._prompt_queue or self._quitting:
            return
        if cancelled or self.adapter._cancelled:
            prompts, self._prompt_queue = self._prompt_queue, []
            self._render_queue()
            self._put_back_in_input(prompts)
            self._append(f"Moved {len(prompts)} queued prompt{'s' if len(prompts) != 1 else ''} back to the input: "
                         "edit and send, or clear it.", msg_type="system")
            return
        next_text = self._prompt_queue.pop(0)
        self._render_queue()
        self._append(next_text, msg_type="user")
        self._begin_turn(next_text)

    # ── Queued prompts: listed above the input; /queue (or a click) edits, reorders or removes them ──

    def _queue_bar_text(self) -> Text:
        width = max(self.size.width - 8, 20)
        text = Text(style="#8a8a8a")
        pending = self._steering.pending()
        if pending:  # steering first: these are read before the queued group runs (steering spec D9)
            text.append(f"→ {len(pending)} for the agent · read at the next tool boundary")
            for i, msg in enumerate(pending[:QUEUE_BAR_MAX_ROWS], 1):
                text.append(f"\n  → {i}. {_one_line(msg.raw, width)}")
            if (more := len(pending) - QUEUE_BAR_MAX_ROWS) > 0:
                text.append(f"\n  … {more} more", style="dim")
            if self._foreground_jobs():  # D10: why it is waiting, and the way out
                text.append("\n  waiting for a running command · Ctrl+B to background it",
                            style="#e5c07b")
        if self._prompt_queue:
            if pending:
                text.append("\n")
            text.append(f"{len(self._prompt_queue)} queued · click or /queue to edit, reorder or remove")
            for i, prompt in enumerate(self._prompt_queue[:QUEUE_BAR_MAX_ROWS], 1):
                text.append(f"\n  {i}. {_one_line(prompt, width)}")
            if (more := len(self._prompt_queue) - QUEUE_BAR_MAX_ROWS) > 0:
                text.append(f"\n  … {more} more", style="dim")
        return text

    def _render_queue(self) -> None:
        """The queue changed: the bar above the input and the loader's count."""
        self._render_loader()
        try:
            bar = self.query_one("#queue-bar", QueueBar)
        except Exception:
            return  # not mounted yet / shutting down
        bar.display = bool(self._prompt_queue) or bool(self._steering.pending())
        if bar.display:
            bar.update(self._queue_bar_text())
        elif self._queue_picker_open():
            self._close_picker()

    def _put_back_in_input(self, prompts: list[str]) -> None:
        """Queued prompts into the input, ahead of anything already typed there."""
        inp = self.query_one("#input", EnterSubmits)
        inp.text = "\n\n".join(prompts + ([inp.text] if inp.text.strip() else []))
        inp.recalled_text = inp.text  # keep the "/" menu shut for a queued skill command
        inp.move_cursor(inp.document.end)

    def _clear_queue(self) -> None:
        count = len(self._prompt_queue)
        self._prompt_queue.clear()
        # The rows the picker listed are gone; with steering messages pending the bar stays up, so
        # _render_queue alone would leave the picker open over dead options (choosing one says
        # "That prompt already started."). Close it here — clicking the bar reopens the list.
        self._close_picker()
        self._render_queue()
        self._append(f"Cleared {count} queued prompt{'s' if count != 1 else ''}." if count else "No queued prompts.",
                     msg_type="system")

    def _queue_picker_open(self) -> bool:
        return self._picker_open() and self.query_one("#picker", PermissionPrompt)._question.startswith(
            ("Queued prompt", "Steering message"))

    def action_toggle_queue(self) -> None:
        """The queue bar was clicked: open the list, or close it when it's already showing."""
        if self._queue_picker_open():
            self._close_picker()
        else:
            self.action_show_queue()

    def action_show_queue(self) -> None:
        """/queue or a click on the queue bar: pick a steering message or a queued prompt, or clear the
        prompts. Re-opens over our own picker (Edit / Remove / Back show it again), never over another's.
        """
        if ((self._picker_open() and not self._queue_picker_open()) or self._settings_open()
                or self.query("#permission-prompt")):
            return
        pending = self._steering.pending()
        if not pending and not self._prompt_queue:
            self._append("No queued prompts or steering messages. Prompts sent while a turn runs wait here.",
                         msg_type="system")
            return
        prompts = list(self._prompt_queue)
        options = [(f"s{i}", f"→ {i + 1}. {_one_line(m.raw, 70)}") for i, m in enumerate(pending)]
        options += [(f"q{i}", f"{len(pending) + i + 1}. {_one_line(p, 70)}") for i, p in enumerate(prompts)]
        options.append(("clear", "Clear the queue"))

        async def choose(value: str) -> None:
            if value == "clear":
                self._clear_queue()
            elif value.startswith("s"):
                self._steer_actions(pending[int(value[1:])].id)
            else:
                self._queued_prompt_actions(prompts[int(value[1:])])

        self._show_picker("Queued prompts and steering messages (→ read at the next tool boundary)",
                          options, choose)

    def _steer_actions(self, msg_id: str) -> None:
        """One steering message: its text, then edit it (back to the input), remove it, or go back (D9)."""
        pending = self._steering.pending()
        msg = next((m for m in pending if m.id == msg_id), None)
        if msg is None:  # read by the loop between the listing and this choice (REQ-1.3)
            self._append("That message was already read.", msg_type="system")
            return
        raw = msg.raw
        position = pending.index(msg)
        options = [("edit", "Edit it (moves it back to the input)"), ("remove", "Remove it"),
                   ("back", "Back to the list")]
        lines = raw.splitlines() or [""]
        detail = [(line, "#8a8a8a") for line in lines[:QUEUE_DETAIL_LINES]]
        if len(lines) > QUEUE_DETAIL_LINES:
            detail.append((f"… {len(lines) - QUEUE_DETAIL_LINES} more lines", "dim"))

        async def act(choice: str) -> None:
            if choice == "back":
                self.action_show_queue()
                return
            if not self._steering.remove(msg_id):
                self._append("That message was already read.", msg_type="system")
                return
            self._steer_raws.pop(msg_id, None)
            if choice == "edit":
                self._put_back_in_input([raw])
            else:
                self._append(f"Removed steering message: {_one_line(raw, 60)}", msg_type="system")
            self._render_queue()
            if self._steering.pending() or self._prompt_queue:
                self.action_show_queue()
            else:
                self._close_picker()

        self._show_picker(f"Steering message {position + 1} of {len(pending)}", options, act, detail=detail)

    def _queued_prompt_actions(self, prompt: str) -> None:
        """One queued prompt: its text, then edit it, run it next, send it now (as a steering message),
        remove it, or go back."""
        if prompt not in self._prompt_queue:
            self._append("That prompt already started.", msg_type="system")
            return
        position = self._prompt_queue.index(prompt)
        options = [("edit", "Edit it (moves it back to the input)")]
        if position:
            options.append(("next", "Run it next"))
        options += [("send", "Send now (at the next tool boundary)"), ("remove", "Remove it"),
                    ("back", "Back to the list")]
        lines = prompt.splitlines() or [""]
        detail = [(line, "#8a8a8a") for line in lines[:QUEUE_DETAIL_LINES]]
        if len(lines) > QUEUE_DETAIL_LINES:
            detail.append((f"… {len(lines) - QUEUE_DETAIL_LINES} more lines", "dim"))

        async def act(choice: str) -> None:
            if choice == "back":
                self.action_show_queue()
                return
            if prompt not in self._prompt_queue:
                self._append("That prompt already started.", msg_type="system")
                return
            self._prompt_queue.remove(prompt)
            if choice == "next":
                self._prompt_queue.insert(0, prompt)
            elif choice == "edit":
                self._put_back_in_input([prompt])
            elif choice == "send":  # steering spec D2: read at the next boundary instead of after the turn
                self._steer_submit(prompt)
            else:
                self._append(f"Removed queued prompt: {_one_line(prompt, 60)}", msg_type="system")
            self._render_queue()
            if choice == "send":
                self.action_show_queue()  # back on the list, marked → now

        self._show_picker(f"Queued prompt {position + 1} of {len(self._prompt_queue)}", options, act, detail=detail)

    # ── Loader / timer / streaming helpers ─────────────────────────────────────

    def _show_loader(self, label: str = "Working...") -> Static:
        """Show (or update) loader message in conversation. Returns the Static widget."""
        self._loader_label = label
        existing = self._loader_widget
        if existing is not None and existing.parent is not None:
            existing.styles.display = "block"
            self._render_loader()
            return existing
        loader_widget = Static(self._loader_text(), classes="msg msg-loader", markup=False)
        self.query_one("#msg-container", Vertical).mount(loader_widget)
        self._loader_widget = loader_widget
        return loader_widget

    def _loader_text(self) -> Text:
        """Bold label, then the elapsed time and the steering / queued counts (steering spec D9: the count
        reads `· 1 steering · 2 queued`)."""
        text = Text(self._loader_label, style="bold")
        if self._turn_start_time is not None:
            text.append(f"  {_fmt_elapsed(time.time() - self._turn_start_time)}", style="bold #e5c07b")
        counts = [f"· {len(pending)} steering" for pending in (self._steering.pending(),) if pending]
        if self._prompt_queue:
            counts.append(f"· {len(self._prompt_queue)} queued")
        if counts:
            text.append("  " + " ".join(counts), style="#8a8a8a")
        return text

    def _render_loader(self) -> None:
        widget = self._loader_widget
        if widget is not None and widget.parent is not None:
            widget.update(self._loader_text())

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
        self._timer = self.set_interval(1.0, self._tick)

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        if self._turn_start_time is not None and not self.adapter._cancelled:
            self._render_loader()

    def _update_stream_widget(self, kind: str, text: str) -> None:
        """Update (or create) the live widget for this model call's answer or reasoning text."""
        self._stream_texts[kind] = text
        widget = self._stream_widgets.get(kind)
        if widget is None or widget.parent is None:
            css = "msg msg-thought" if kind == "thought" else "msg msg-agent"
            widget = SelectableStatic("" if kind == "thought" else text, classes=css, markup=False)
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
        if self._btw_open() and not self.adapter.has_pending_prompt():
            self._close_btw()  # closes the side answer; the turn keeps running (btw spec D6)
            return
        if self._picker_open():
            self._close_picker()
            return
        if self._settings_open():
            self._close_settings()
            return
        if self._agent_view_open():
            self._close_agent_view()
            return
        if not self._turn_in_progress and not self.adapter.has_pending_prompt():
            # Nothing to cancel (a stale flag would cancel the next turn's first step); Esc Esc
            # with an empty input takes back the last prompt
            self._escape_when_idle()
            return
        self.adapter._cancelled = True
        self.adapter._cancel_event.set()
        self.adapter.deny_pending_confirm()
        self.adapter.cancel_pending_ask()  # a question resolves as cancelled before the flag is polled (D7)
        self._loader_label = "Cancelling..."
        self._render_loader()

    # ── Ctrl+B: move the turn's running commands to the background (background shell D9) ──

    def _foreground_jobs(self) -> list[Job]:
        """Running model shell calls of this session and its subagents (never a `!command`)."""
        return [j for j in get_registry().jobs()
                if j.foreground and j.ended is None and session_of(j.owner) == self.session_id]

    def _foreground_agents(self) -> list["AgentRun"]:
        """This session's foreground subagent runs that haven't ended (running, waiting or queued)."""
        return [r for r in _agent_registry().runs(self.session_id)
                if not r.background and r.status in ("queued", "running", "waiting_approval")]

    def action_background_jobs(self) -> None:
        """Ctrl+B: move the turn's foreground shell jobs and subagents to the background (subagents D9)."""
        if not self._turn_in_progress or not (self._foreground_jobs() or self._foreground_agents()):
            self._flash_input_hint("nothing running to move to the background")
            raise SkipAction()  # the key still goes to the focused widget
        get_registry().detach_foreground(self.session_id)
        _agent_registry().detach_foreground(self.session_id)
        self._flash_input_hint("moved to the background · /jobs or /agents to see it")

    def _on_job_event(self, job: Job, event: JobEvent) -> None:
        if session_of(job.owner) == self.session_id:
            self.call_later(self._refresh_job_hint)
            self.call_later(self._render_jobs_bar)
            self.call_later(self._render_queue)  # the D10 "Ctrl+B to background it" hint follows jobs
            if event == "end" and not job.foreground and job.killed_by in (None, "user"):
                self.call_later(self._job_ended, job)  # the ones the owner gets a notice for (D7)

    def _job_ended(self, job: Job) -> None:
        """A background job ended on its own or by /jobs kill: a system line, and a notification
        when the terminal is in the background. The model hears on its next call, which is started
        now if the main agent's job ended on its own while the TUI is idle (D15)."""
        line = notice_line(job.id, job.status, job.exit_code, job.killed_by, fmt_runtime(job.runtime))
        self._append(line, msg_type="system")
        self._notify(f"dostuff: {line.removeprefix('⚙ ')}", 0.0)
        if job.owner == self.session_id and job.killed_by is None:
            self._schedule_wake()

    def _on_agent_event(self, run: "AgentRun", event: "AgentEvent") -> None:
        if run.owner != self.session_id:
            return
        self.call_later(self._refresh_job_hint)
        self.call_later(self._render_agents_bar)
        if event == "end" and run.background and run.stopped_by != "session_end":
            self.call_later(self._agent_ended, run)
            if run.stopped_by is None:  # ended on its own: the idle TUI resumes the agent (D7)
                self.call_later(self._schedule_wake)

    def _agent_ended(self, run: "AgentRun") -> None:
        """A background agent ended: a notification when the terminal is in the background (its system line
        came through the adapter; subagents spec D7)."""
        line = agent_notice_line(run.description, run.agent_type, run.status,
                                 fmt_agent_runtime((run.ended or 0) - (run.started or run.ended or 0)))
        self._notify(f"dostuff: {line.removeprefix('⚙ ')}", 0.0)

    # ── Wake-up: a turn for background jobs that ended while idle (spec D15) ─────

    def _wake_notices(self) -> list:
        """The main agent's queued notices, if any of them warrants a wake-up: a job (jobs.wake_on_end) or a
        background agent (agents.wake_on_end) that ended on its own. A /jobs kill or a stopped agent alone
        doesn't wake it (the user is right there); it rides along with one that does."""
        notices = get_notice_queue().pending(self.session_id)
        if any(n.killed_by is None for n in notices if n.kind == "job") and self.config.jobs_wake_on_end:
            return notices
        if any(n.wake for n in notices if n.kind == "agent") and self.config.agents_wake_on_end:
            return notices
        return []

    def _schedule_wake(self) -> None:
        """Check for a wake-up after JOB_WAKE_DELAY_S, so jobs and agents that end together start one turn."""
        if self._wake_timer is not None or not self._wake_notices():
            return
        self._wake_timer = self.set_timer(JOB_WAKE_DELAY_S, self._try_wake)

    def _wake_blocked(self) -> bool:
        """Busy, or the user is in the middle of something: a turn, typing, a prompt or picker,
        a session command (/compact, /clear, …) or a queued prompt that will run first."""
        if self._turn_in_progress or self._prompt_queue:
            return True
        if any(w.name not in JOB_WAKE_IDLE_WORKERS for w in self._pending_workers):
            return True
        return self._prompt_blocked()

    def _prompt_blocked(self) -> bool:
        """The user is in the middle of something a background agent's approval prompt must not cut into:
        typing, a prompt or picker, quitting (subagents D8)."""
        if self._quitting or self._quit_confirming:
            return True
        if self._picker_open() or self.query("#permission-prompt"):
            return True
        try:
            return bool(self.query_one("#input", EnterSubmits).text.strip())
        except Exception:
            return True

    def _try_wake(self) -> None:
        self._wake_timer = None
        notices = self._wake_notices()
        if not notices:
            return  # a user turn drained them meanwhile
        if self._wake_blocked():
            self._schedule_wake()  # retry; a prompt the user sends drains the notices instead
            return
        if self._wake_turns >= WAKE_LIMIT:
            if self._wake_turns == WAKE_LIMIT:  # say it once per streak
                self._wake_turns += 1
                self._append(f"⚙ Not resuming the agent: {WAKE_LIMIT} automatic turns in a row. "
                             "It sees the job results with your next message.", msg_type="system")
            return
        self._wake_turns += 1
        jobs = ", ".join(f'"{n.id}"' for n in notices if n.kind == "job")
        agents = ", ".join(f'"{n.description}"' for n in notices if n.kind == "agent")
        what = " and ".join(([f"background job {jobs}"] if jobs else []) + ([f"background agent {agents}"] if agents else []))
        self._append(f"⚙ Resuming the agent: {what} ended", msg_type="system")
        self._turn_in_progress = True
        text = format_notices(notices, self.config.max_tool_output_chars)
        self._start_worker(self._run_turn(text, wake=True), name=f"wake_{uuid.uuid4().hex[:6]}")

    def _refresh_job_hint(self) -> None:
        """The loader offers Ctrl+B only while a model shell call or a foreground subagent is running."""
        if not self._turn_in_progress or self._loader_label not in (WORKING_LABEL, WORKING_LABEL_JOBS):
            return
        self._loader_label = WORKING_LABEL_JOBS if self._foreground_jobs() or self._foreground_agents() else WORKING_LABEL
        self._render_loader()

    # ── Esc Esc: edit the previous prompt · Ctrl+R: search prompt history ──────

    def _last_prompt_index(self) -> int | None:
        """Index in steps_history of the last prompt the user sent (not a `!` command or a note)."""
        for i in range(len(self.steps_history) - 1, -1, -1):
            step = self.steps_history[i]
            content = step.get("content") if isinstance(step, dict) else None
            if (step.get("role") == "user" and isinstance(content, str) and content != EMPTY_RESPONSE_NUDGE
                    and not content.startswith("[/undo]") and _parse_shell_step(content) is None
                    and parse_stop_note(content) is None and parse_notices(content) is None):
                return i
        return None

    def _escape_when_idle(self) -> None:
        """Esc with no turn to cancel: Esc Esc (empty input) takes back the last prompt. Every press
        says what it did or how to quit, since a reflexive Esc is often someone trying to leave."""
        inp = self.query_one("#input", EnterSubmits)
        if inp.text.strip() or self._last_prompt_index() is None:
            self._last_escape = 0.0
            self._flash_input_hint(QUIT_HINT)
            return
        now = time.monotonic()
        if now - self._last_escape <= EDIT_PROMPT_ESC_SECONDS:
            self._last_escape = 0.0
            self._start_worker(self._edit_previous_prompt(), name="edit_prompt")
            return
        self._last_escape = now
        self._flash_input_hint(f"Esc again to edit the previous prompt · {QUIT_HINT}",
                               seconds=EDIT_PROMPT_ESC_SECONDS)

    async def _edit_previous_prompt(self) -> None:
        """Drop the last prompt and everything after it (answer, tool calls) and put the prompt back
        in the input. Files the turn changed stay as they are; /undo restores them."""
        index = self._last_prompt_index()
        if self._turn_in_progress or index is None:
            return
        dropped = self.steps_history[index:]
        # After a reload _sent_prompts is empty and the stored step may be framed (D6): unframe so
        # Esc-to-restore returns the typed text, never the frame (REQ-2.9).
        text = self._sent_prompts.get((self.session_id, index)) or unframe_steered(
            strip_attachments(dropped[0]["content"]))
        try:
            await self.store.drop_last_steps(self.session_id, len(dropped))
        except Exception as e:
            self._append(f"Could not remove the last prompt: {e}", msg_type="error")
            return
        del self.steps_history[index:]
        if self.current_session_history:
            del self.current_session_history[-min(len(dropped), len(self.current_session_history)):]
        changed_files = any(
            (tc.get("function") or {}).get("name") in ("write_file", "edit_file", "delete_file")
            for step in dropped for tc in (step.get("tool_calls") or []) if isinstance(tc, dict)
        )
        await self.query_one("#msg-container", Vertical).remove_children()
        self._render_history()
        self._append("Took back the last prompt and its answer: edit it and press Enter to send it again."
                     + (" Files that turn changed are left as they are; /undo restores them." if changed_files else ""),
                     msg_type="system")
        inp = self.query_one("#input", EnterSubmits)
        inp.text = text
        inp.recalled_text = text  # keep the "/" menu shut for a recalled command
        inp.move_cursor(inp.document.end)

    def action_history_search(self) -> None:
        """Ctrl+R: pick a past prompt from this directory's history (type to filter) to put in the input."""
        if self._settings_open() or self._picker_open() or self.adapter.has_pending_prompt():
            return
        inp = self.query_one("#input", EnterSubmits)
        prompts = list(dict.fromkeys(reversed(inp.prompt_history)))[:HISTORY_SEARCH_MAX]  # newest first, no repeats
        if not prompts:
            self._flash_input_hint("no prompt history in this directory yet")
            return

        def label(prompt: str) -> str:
            lines = prompt.strip().splitlines() or [""]
            first = lines[0] if len(lines[0]) <= 100 else lines[0][:99] + "…"
            return first + (f"  (+{len(lines) - 1} lines)" if len(lines) > 1 else "")

        async def choose(value: str) -> None:
            index = value[1:] if value.startswith("h") else ""
            text = prompts[int(index)] if index.isdigit() and int(index) < len(prompts) else value  # else: typed text
            inp.text = text
            inp.recalled_text = text
            inp.move_cursor(inp.document.end)

        self._show_picker("Search prompt history", [(f"h{i}", label(p)) for i, p in enumerate(prompts)], choose,
                          filterable=True)

    # ── Exit ───────────────────────────────────────────────────────────────────

    def action_help_quit(self) -> None:
        """Ctrl+C with nothing selected: Textual's "Do you want to quit?" toast, as an input-border hint."""
        self._flash_input_hint(QUIT_HINT)

    async def action_quit(self) -> None:
        """Ctrl+Q / /exit: clean up and quit; repeated presses are ignored.

        With background jobs running it asks first (in a worker: an action can't wait for a key).
        """
        if self._quitting or self._quit_confirming:
            return
        self.adapter.cancel_pending_ask()  # nothing waits on a dead event while we quit (D7)
        if self._running_jobs() or self._running_background_agents():
            self._quit_confirming = True
            self.run_worker(self._quit_after_confirm(), name="quit_confirm", exit_on_error=False)
            return
        self._quitting = True
        self.run_worker(self._run_cleanup_and_exit(), name="cleanup_worker", exit_on_error=False)

    async def _quit_after_confirm(self) -> None:
        try:
            if not await self._confirm_quit_background():
                return
        finally:
            self._quit_confirming = False
        self._quitting = True
        await self._run_cleanup_and_exit()

    # ── Background jobs (plan/2026-10-31-background-shell D8) ────────────────────

    def _running_jobs(self) -> list[Job]:
        """Running background jobs, not a subagent's: those die with their agent (counted as the agent)."""
        return [j for j in get_registry().jobs() if j.ended is None and not j.foreground and "/sub/" not in j.owner]

    async def _jobs_housekeeping(self) -> None:
        """Startup: drop old job logs; warn once about jobs a crashed earlier session may have left (D11)."""
        try:
            await asyncio.to_thread(prune_logs)
            orphans = await asyncio.to_thread(find_orphans, self.session_id)
        except Exception:
            return
        if orphans:
            count = f"{len(orphans)} background job{'s' if len(orphans) != 1 else ''}"
            listed = "; ".join(f"{command}, pid {pid}" for command, pid in orphans[:5])
            more = "; …" if len(orphans) > 5 else ""
            self._append(f"⚠ {count} from an earlier session may still be running ({listed}{more}). "
                         "Check them with your OS tools.", msg_type="error")

    def _running_background_agents(self) -> list["AgentRun"]:
        return [r for r in _agent_registry().runs()
                if r.background and r.status in ("queued", "running", "waiting_approval")]

    async def _confirm_quit_background(self) -> bool:
        """Ask before quitting stops the running background jobs and agents; True to go ahead (also with none)."""
        jobs, agents = self._running_jobs(), self._running_background_agents()
        if not jobs and not agents:
            return True
        parts = ([f"{len(jobs)} background job{'s' if len(jobs) != 1 else ''}"] if jobs else []) +             ([f"{len(agents)} agent{'s' if len(agents) != 1 else ''}"] if agents else [])
        count = " and ".join(parts)
        names = ", ".join([j.id for j in jobs] + [f'"{r.description}"' for r in agents])
        return await self.adapter.confirm(
            f"⚙ Quit and stop {count}? ({names})", "Quit?", f"Yes, stop {count} and quit", "No, keep working")

    def _carry_background(self, old_session_id: str) -> tuple[list[Job], list["AgentRun"]]:
        """/clear and /resume: the old session's running jobs (background shell D8) and background agents
        (subagents D14) keep running in the current one; queued notices move with them."""
        if old_session_id == self.session_id:
            return [], []
        jobs = get_registry().transfer(old_session_id, self.session_id)
        agents = _agent_registry().transfer(old_session_id, self.session_id)
        self.call_later(self._render_jobs_bar)
        self.call_later(self._render_agents_bar)
        self._schedule_wake()  # notices queued for the old session now belong to this one
        return jobs, agents

    def _report_carried(self, carried: tuple[list[Job], list["AgentRun"]]) -> None:
        jobs, agents = carried
        if jobs:
            count = f"{len(jobs)} background job{'s' if len(jobs) != 1 else ''}"
            self._append(f"⚙ {count} ({', '.join(j.id for j in jobs)}) kept running.", msg_type="system")
        if agents:
            count = f"{len(agents)} agent{'s' if len(agents) != 1 else ''}"
            names = ", ".join(f'"{r.description}"' for r in agents)
            self._append(f"⚙ {count} ({names}) kept running.", msg_type="system")

    async def _stop_agents(self) -> None:
        """Stop every subagent run, bounded (subagents CON-007); no notices, nobody is left to read them.
        Each run kills its own shell jobs; _stop_jobs catches any left over."""
        from dostuff.agent.subagents.registry import STOP_GRACE_S
        try:
            await asyncio.wait_for(_agent_registry().stop_all("session_end"), timeout=STOP_GRACE_S + KILL_WAIT_S)
        except Exception:
            pass

    async def _stop_jobs(self) -> None:
        """Kill every job, bounded; falls back to the synchronous kill."""
        registry = get_registry()
        try:
            await asyncio.wait_for(registry.kill_all(), timeout=KILL_WAIT_S + 1)
        except Exception:
            registry.kill_all_sync()

    async def _run_cleanup_and_exit(self) -> None:
        self._append("Saving session...")
        try:
            await self._cleanup_and_save_session()
        except Exception as e:
            self._append(f"Cleanup error: {e}", msg_type="error")
        finally:
            self._restore_title()
            self.exit()

    async def _cleanup_and_save_session(self) -> None:
        """Stop agents and jobs, cancel pending work, close MCP sessions and the DB, with bounded timeouts."""
        # 0. Background and foreground shell jobs: none outlives the app (background shell CON-008)
        get_registry().remove_listener(self._on_job_event)
        _agent_registry().remove_listener(self._on_agent_event)
        await self._stop_agents()
        await self._stop_jobs()
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
            if self.mcp_client is not None:
                await asyncio.wait_for(self.mcp_client.cleanup(), timeout=5.0)
        except asyncio.TimeoutError:
            self._append("MCP shutdown timed out.", msg_type="error")
        # 3. Worktree (before the DB closes: removing it points the session back at the main checkout)
        try:
            await asyncio.wait_for(self._close_worktree_on_exit(), timeout=30.0)
        except asyncio.TimeoutError:
            self.exit_note = f"Timed out cleaning up worktree {self.worktree.path if self.worktree else ''}."
        # 4. Paste files, when loop.paste_retention_days is 0 (otherwise the startup sweep ages them out)
        if self.config.paste_retention_days <= 0:
            self._pastes.delete_files()
        # 5. Session DB, after the agent rows and steps still queued for it (subagents D14)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(_agent_registry().flush(), timeout=5.0)
        await self.store.close()
        # 6. Force-stop any lingering litellm logging workers (background threads)
        try:
            litellm = sys.modules.get("litellm")  # don't import it just to stop it
            worker = getattr(litellm, "logging_worker", None)
            if worker is not None and hasattr(worker, "stop"):
                worker.stop()
        except Exception:
            pass
        self._append(f"Session saved. Goodbye! (cancelled {len(workers)} pending task(s))")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _mount_message(self, widget: Static) -> None:
        """Mount above the loader (so the loader stays at the bottom). While the agent view renders a
        transcript, `_message_target` points at that view's container instead of the conversation."""
        msg_container = self.query_one(self._message_target, Vertical)
        loader = self._loader_widget
        # Children, not is_mounted: a loader mounted moments ago (before the hooks' trust question) isn't yet
        if loader is not None and loader.parent is msg_container and loader in msg_container.children:
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
            msg = SelectableStatic(RichMarkdown(line), classes=css_class, markup=False)
        elif "\033[" in line:
            msg = Static(Text.from_ansi(line), classes=css_class, markup=False)
        else:
            msg = Static(line, classes=css_class, markup=False)
        self._mount_message(msg)

    def _append_tool_result(self, text: str, max_lines: int = TOOL_RESULT_PREVIEW_LINES,
                            msg_type: str = "tool-result") -> None:
        """A collapsible result (Ctrl+O / click); capped at the size the model sees."""
        text = truncate_tool_output(text, self.config.max_tool_output_chars)
        self._mount_message(ToolResult(text, max_lines, expanded=self._tool_results_expanded,
                                       classes=f"msg msg-{msg_type}"))

    def _append_call_result(self, text: str) -> None:
        """A tool call's result; one Ctrl+B moved to the background collapses to that fact."""
        job_id = detached_job_id(text) if text.startswith("bash_command: ") else None
        if job_id is None:
            self._append_tool_result(text)
        else:
            self._append_tool_result(f'moved to background as "{job_id}"\n{text}', max_lines=1)

    def _append_answers_block(self, pairs: list[tuple[str, str]]) -> None:
        """`ask_user`'s answers as one full `⎿` block, a row per pair (D9): no preview cap and no Ctrl+O,
        because the whole point is that the user can read back what was answered.

        Shared by the live path (`_emit_on_ui`) and by `/resume` (`_render_history_step`), so a stored
        session renders identically to the turn that made it."""
        text = Text()
        for i, (question, answer) in enumerate(pairs):
            text.append("  ⎿  " if i == 0 else "     ")
            text.append(_one_line_answer(question), style="#8a8a8a")
            text.append(" → ")
            text.append(_one_line_answer(answer))
            if i < len(pairs) - 1:
                text.append("\n")
        message = SelectableStatic(text, classes="msg msg-tool-result", markup=False)
        self._mount_message(message)

    def action_toggle_tool_results(self) -> None:
        """Ctrl+O: expand every truncated tool / shell result, or collapse them all again."""
        self._tool_results_expanded = not self._tool_results_expanded
        for widget in self.query(ToolResult):
            widget.set_expanded(self._tool_results_expanded)
        self._flash_input_hint("tool output expanded · ctrl+o to collapse" if self._tool_results_expanded
                               else "tool output collapsed · ctrl+o to expand")

    async def _connect_one_mcp_server(self, name: str, cfg: dict) -> None:
        timeout = float(cfg.get("connect_timeout") or MCP_CONNECT_TIMEOUT_S)
        assert self.mcp_client is not None  # connections start from _load_backend
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
        servers = self.mcp_client.servers if self.mcp_client is not None else {}
        self._append(f"MCP ready ({len(servers)} server(s)).", msg_type="system")

    async def _check_settings_files(self) -> None:
        """Pick up config.yaml and mcp_config.json edited outside dostuff (an editor, a shell) while it
        runs, idle or mid-turn, so a change applies without restarting the session."""
        if self._settings_check_running:
            return
        self._settings_check_running = True
        try:
            reloaded = await asyncio.to_thread(self.config.reload_if_changed)
            if reloaded:
                self._append("config.yaml changed on disk: reloaded.", msg_type="system")
                for warning in self.config.hooks_warnings:
                    self._append(warning, msg_type="error")
            error = self.config.load_error
            if error and error != self._reported_load_error:
                self._append(f"config.yaml not reloaded, keeping the previous settings: {error}", msg_type="error")
            self._reported_load_error = error
            mcp_changed = False
            if self.mcp_client is not None:
                signature = await asyncio.to_thread(_mcp_files_signature)
                mcp_changed = signature != self._mcp_files_signature
                self._mcp_files_signature = signature
            if reloaded or mcp_changed:
                self._apply_settings(mcp=mcp_changed)
        except Exception as e:
            self._append(f"Could not check the settings files: {e}", msg_type="error")
        finally:
            self._settings_check_running = False

    def _apply_settings(self, mcp: bool) -> None:
        """Pick up a settings change (Config is already reloaded) without a restart."""
        self.max_iterations = self.config.max_iterations
        self.query_one("#input", EnterSubmits).paste_collapse_lines = self.config.paste_collapse_lines
        self._render_status()
        if mcp:
            self._start_worker(self._sync_mcp_servers(), name="mcp_sync")

    async def _sync_mcp_servers(self) -> None:
        """Connect added or changed MCP servers and disconnect removed ones."""
        if not await self._wait_for_backend() or self.mcp_client is None:
            return
        servers: dict[str, dict] = {}
        for source in mcp_config_sources():
            try:
                servers.update(read_mcp_file(source))
            except Exception as e:
                self._append(f"Could not read {source}: {e}", msg_type="error")
                return
        old = self._mcp_server_configs
        gone = [n for n in old if n not in servers]
        changed = [n for n, cfg in servers.items() if old.get(n) != cfg]
        self._mcp_server_configs, self._mcp_configured = dict(servers), list(servers)
        for name in gone:
            await self.mcp_client.disconnect(name)
            self._append(f"MCP: disconnected {name}", msg_type="system")
        if changed:
            await asyncio.gather(*(self._connect_one_mcp_server(n, servers[n]) for n in changed))

    # Least important first: dropped in this order when the status bar doesn't fit
    _STATUS_DROP_ORDER = ("help", "session", "cwd", "tokens", "cost", "model", "context", "mode")

    def _update_status(self, context_window_percent: str = "") -> None:
        """Status bar: the ui.status_bar items (cwd, model, permission mode, context % by default)."""
        if context_window_percent:
            self.adapter._last_context_window_percent = context_window_percent
        self._render_status()

    def on_resize(self) -> None:
        self._render_status()

    def _status_items(self) -> dict[str, list[Text]]:
        """Each item's renderings, longest first; the bar uses the first one that fits."""
        cwd = os.getcwd()
        cwd_display = cwd.replace(str(Path.home()), "~")
        cwd_display = cwd_display if len(cwd_display) < 36 else "..." + cwd_display[-33:]
        mode = self.permissions.mode or self.config.permission_mode
        mode_style = {"auto": "bold #e06c75", "plan": "bold #8ab4f8"}.get(mode, "")
        cost = self.adapter._session_cost
        items: dict[str, list[Text]] = {
            "help": [Text("/help")],
            "cwd": [Text(cwd_display), Text(Path(cwd).name or cwd_display)],
            "session": [Text(self.session_id)],
            "model": [Text(get_model()), Text(get_model().split("/")[-1])],
            "mode": [Text(f"{mode} mode", style=mode_style)],
            "tokens": [Text((self.adapter._token_display or " ↑0 ↓0").strip())],
            "cost": [Text(format_cost(cost))] if cost else [],
            "context": [Text(p)] if (p := self.adapter._last_context_window_percent) else [],
        }
        return {name: items[name] for name in self.config.status_bar_items if items.get(name)}

    def _render_status(self) -> None:
        """Fit the bar to the terminal: shorten items, then drop the least important ones."""
        items = self._status_items()
        choice = {name: 0 for name in items}
        # the shells and agents items sit beside the status items, so they take from their room;
        # #status, #jobs-bar and #agents-bar have 1 cell of padding each side
        extras = [t for t in (self._jobs_bar_text(), self._agents_bar_text()) if t is not None]
        width = max(self.size.width - 2 - sum(t.cell_len + 2 for t in extras), 1)

        def line() -> Text:
            return Text("  ").join(items[name][choice[name]] for name in items)

        drop = [name for name in self._STATUS_DROP_ORDER if name in items]
        while line().cell_len > width:
            shorter = next((n for n in drop if choice[n] + 1 < len(items[n])), None)
            if shorter is not None:
                choice[shorter] += 1
            elif len(drop) > 1:
                name = drop.pop(0)
                del items[name], choice[name]
            else:
                break  # one item left: Static crops it
        try:
            self.query_one("#status", Static).update(line())
        except Exception:
            pass  # not mounted yet / shutting down

    def _jobs_bar_text(self) -> Text | None:
        """`⚙ 2 shells`, or None while this session has no background jobs running."""
        running = sum(1 for j in self._session_jobs() if j.ended is None)
        return Text(f"⚙ {running} shell{'s' if running != 1 else ''}", style="#e5c07b") if running else None

    def _render_jobs_bar(self) -> None:
        """The jobs item beside the status bar: shown while this session has background jobs running."""
        text = self._jobs_bar_text()
        try:
            bar = self.query_one("#jobs-bar", JobsBar)
        except Exception:
            return  # not mounted yet / shutting down
        bar.display = text is not None
        if text is not None:
            bar.update(text)
        self._render_status()  # the status items get more or less room

    def _agents_bar_text(self) -> Text | None:
        """`⚠ 1 agent waiting` while any of this session's agents waits for approval, else `⚙ 2 agents` while
        any runs (queued included); None when none do."""
        try:
            runs = [r for r in _agent_registry().runs(self.session_id)
                    if r.status in ("queued", "running", "waiting_approval")]
        except Exception:
            return None
        waiting = sum(1 for r in runs if r.status == "waiting_approval")
        if waiting:
            return Text(f"⚠ {waiting} agent{'s' if waiting != 1 else ''} waiting", style="#e5c07b")
        return Text(f"⚙ {len(runs)} agent{'s' if len(runs) != 1 else ''}", style="#e5c07b") if runs else None

    def _render_agents_bar(self) -> None:
        text = self._agents_bar_text()
        try:
            bar = self.query_one("#agents-bar", AgentsBar)
        except Exception:
            return  # not mounted yet / shutting down
        bar.display = text is not None
        if text is not None:
            bar.update(text)
        self._render_status()

    def action_show_agents(self) -> None:
        """The agents item was clicked: the /agents list, or close it when it's already showing."""
        if self._picker_open() and self.query_one("#picker", PermissionPrompt)._question == "Agents":
            self._close_picker()
            return
        if self._picker_open() or self._settings_open() or self.query("#permission-prompt"):
            return
        self._show_agents_list()

    def _jobs_picker_open(self) -> bool:
        """The jobs list, or one job's actions, is showing."""
        if not self._picker_open():
            return False
        question = self.query_one("#picker", PermissionPrompt)._question
        return question == "Background jobs" or question.startswith('Job "')

    def action_toggle_jobs(self) -> None:
        """The jobs item was clicked: open the list, or close it when it's already showing."""
        if self._jobs_picker_open():
            self._close_picker()
        else:
            self.action_show_jobs()

    def action_show_jobs(self) -> None:
        """The jobs line was clicked (or ↓ in an empty input): pick a job, running ones first."""
        if self._picker_open() or self._settings_open() or self.query("#permission-prompt"):
            return
        jobs = sorted(self._session_jobs(), key=lambda j: (j.ended is not None, j.started))
        if not jobs:
            return
        registry = get_registry()

        def label(job: Job) -> str:
            command = " ".join(job.command.split())
            command = command if len(command) <= 60 else command[:59] + "…"
            since = time.strftime("%H:%M:%S", time.localtime(job.started))
            return (f"{job.id}  {_job_status(job)}  since {since} ({fmt_runtime(job.runtime)})  "
                    f"{registry.owner_label(job.owner)}  $ {command}")

        self._show_picker("Background jobs", [(j.uid, label(j)) for j in jobs], self._job_actions)

    def _job_detail(self, job: Job) -> list[tuple[str, str]]:
        """The job's last JOB_DETAIL_LINES lines of output (stderr in red, at most half when there's
        stdout too) and its log path, for the job's picker: a few rows, not the whole screen."""
        out, err = job.tail("stdout", JOB_DETAIL_LINES), job.tail("stderr", JOB_DETAIL_LINES)
        if not out and not err:
            lines = [("(no output yet)", "dim")]
        else:
            err = err[-(JOB_DETAIL_LINES // 2 if out else JOB_DETAIL_LINES):] if err else []
            out = out[-(JOB_DETAIL_LINES - len(err)):]
            lines = [(line, "#8a8a8a") for line in out] + [(line, "#e06c75") for line in err]
        return lines + [(f"Full log: {job.out_path}", "dim")]

    async def _job_actions(self, uid: str) -> None:
        """A job picked from the jobs list: its recent output, then refresh it, stop it, or go back."""
        job = next((j for j in self._session_jobs() if j.uid == uid), None)
        if job is None:
            return
        options = [("refresh", "Refresh output")]
        if job.ended is None:
            options.append(("kill", "Stop it (the agent is told)"))
        options.append(("back", "Back to the list"))

        async def act(choice: str) -> None:
            if choice == "refresh":
                await self._job_actions(uid)
            elif choice == "kill":
                await self._kill_job(job)
            else:
                self.action_show_jobs()

        self._show_picker(f'Job "{job.id}" · {_job_status(job)} · $ {" ".join(job.command.split())[:60]}',
                          options, act, detail=self._job_detail(job))

    def _fmt_tokens(self, n: int) -> str:
        if n < 1000:
            return str(n)
        if n < 1_000_000:
            return f"{n/1000:.1f}k"
        return f"{n/1_000_000:.1f}M"

    def _remember_prompt(self, text: str) -> None:
        """Up/Down history: this run's list, and the project's history file."""
        inp = self.query_one("#input", EnterSubmits)
        if text.strip() and (not inp.prompt_history or inp.prompt_history[-1] != text):
            append_history(self._history_path, os.getcwd(), text)
        inp.add_history(text)

    def _clear_input(self) -> None:
        try:
            ta = self.query_one("#input", EnterSubmits)
            ta.text = ""
        except Exception:
            pass
