## Why

The current `dostuff` CLI works but lacks an interactive full-screen experience. A TUI improves the agent experience: real-time message display, session context (header/status), and structured input, similar to Pi's own interactive mode.

## What Changes

- Add `textual` dependency for TUI framework.
- Create `textual_proxy.py`-style component-based interface (components: Header, Messages, Editor/Input).
- Wire async agent loop via `self.run_worker()` + `self.call_from_thread()` for thread-safe UI updates.
- Keep CLI commands (`init`, `doctor`, `session-list`) as subcommands; `dostuff` defaults to the TUI experience.

## Capabilities

### New Capabilities
- `tui`: Interactive terminal user interface for `dostuff`, built with `textual`, integrating the agent loop.

### Modified Capabilities
- (none — no existing spec-level behavior changes; this is additive)

## Impact

- New dependency: `textual` (Python TUI library).
- `dostuff` defaults to the TUI entry point; CLI subcommands (`init`, `doctor`, `session-list`) still work as separate commands.
- `.test_venv/textual_proxy.py` serves as prototype; final code lives in `dostuff/` package.
- Agent `core/agent.py` connects to TUI messages component via thread-safe callbacks.
