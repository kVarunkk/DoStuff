## Context

The `dostuff` package uses `typer` CLI (`cli.py`) with a basic agent loop (`core/agent.py`). A prototype (`textual_proxy.py`) confirms `textual` + `run_worker()` + `call_from_thread()` works.

## Goals / Non-Goals

**Goals:**
- Interactive TUI experience using `textual`.
- Component architecture matching Pi (`Header`, `Messages`, `Input`).
- Thread-safe agent integration without blocking the TUI event loop.

**Non-Goals:**
- Changing CLI commands or removing CLI mode.
- Web-based UI (separate scope).
- Full mouse/gesture support (out of scope for initial version).

## Decisions

- **Library choice (`textual`)**: Chosen over scratch components because it provides widget framework, theming, and async event loop integration out of the box. Trade-off: adds dependency.
- **Worker pattern (`run_worker`)**: Keeps agent loop separate from TUI render thread. `call_from_thread()` ensures UI updates are scheduled on the main event loop.
- **Component model**: Mirror Pi (`Header`, `Messages`, `Input`) for consistency. Keeps design portable.

## Risks / Trade-offs

- [Async threading complexity] → Mitigate with worker isolation and `call_from_thread()`.
- [Dependency size (`textual`)] → Acceptable; TUI is optional feature.
- [Breakage if `textual` updates APIs] → Lock version pin; monitor.

## Migration Plan

- Add `textual` to requirements / pyproject.toml.
- Default `dostuff` launches TUI (`cli.py` main callback runs TUI App instead of plain agent loop); CLI subcommands stay as-is.
- Prototype `textual_proxy.py` moved into `dostuff/cli_tui.py` or integrated directly into `cli.py` main.
- Prototype `textual_proxy.py` moved into `dostuff/cli_tui.py` or similar.

## Refinement Decisions (post-discussion, pre-coding)
- Resume history: load into #messages on on_mount
- Tool visibility (adapter.emit): `loop()` emits tool results via adapter.emit() in real-time
- Incremental display: Input.clear on submit; message shown immediately
- Shift+Enter: custom Input binding (Enter=submit, Shift+Enter=newline)
- /exit: save_memories_and_exit via current_session_history
- **HITL confirmations in TUI**: `loop()` currently uses `print` + `input` for HITL prompts (e.g. dangerous bash, delete_file). In TUI, this must use the input box (event-driven), not terminal stdin. Capture `ConfirmationRequired` in TUI worker, prompt via input widget, retry with `_confirmed=True` on accept.

### Adapter Pattern (Durable Solution)
Loop() uses adapter interface instead of direct print/input:
- adapter.emit(event_type, data) → TUI widget updates
- adapter.ask(message, resume_args) → confirmation prompt via input widget
- adapter.input() → input widget value
This decouples agent logic from TUI; adapter handles HITL and tool visibility in one mechanism.

### Visual Design Decisions (post-discussion)

**Layout:** Header removed; Screen transparent (terminal bg); 3 rows — Messages (4fr, borderless), Input (1fr, max 20%, top/bottom white borders only), Status bar (auto height, below input, no border). No custom background colors on any widget.

**Per-role colors (RichLog block markup A):** User=`bright_blue`; Agent/default=no color; Tool=`dim italic`; Confirm=`bold yellow`; Error=`bold red`; System=`dim`.

**Status bar content:** CWD + session-id (short) + resumed/new indicator; style `dim`.

**Token display:** Per-turn counts shown at end of each agent response (e.g., `↑234 ↓89` appended to message line); cumulative session total in status bar (`↑2.4k ↓890`). Human-readable format. Source: `agent_loop()` response usage field.

**Shortcut:** Ctrl+J for newline (Shift+Enter replaced due to framework event issues). Enter submits.
