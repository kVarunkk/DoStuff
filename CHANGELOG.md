# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-09-05

### Added
- **`/exit` and `ctrl+q` unified**: Both now use `run_worker(action_quit())` — same fast path. Cleanup runs in a background thread with a "Saving session..." message displayed.
- **`[TOOL CALL]` / `[TOOL RESULT]` labels in history**: Historical messages now match live display format — tool calls show `[TOOL CALL] name(args)` and results show `[TOOL RESULT] result`, both with correct CSS colors (`.msg-tool` / `.msg-tool-result`).

### Changed
- **`cli.py` restructured**: No longer loads `Config()` before chdir. `session_id` is computed first, then session meta is loaded and `os.chdir()` switches to the resumed session's cwd, then `DostuffTUI` is launched. `Config()` is initialized inside the TUI only (after correct cwd).
- **`--user` flag removed**: Single-user only. `user_id` always comes from `Config().get_user_id()` (auto-creates if missing). No override per session.
- **`.env` lazy loading**: `load_dotenv()` moved from module-level to `Config.load()` — now reloads after `os.chdir()` when resuming a session from a different directory. Resolved the "d1 env applies to d2 session" issue.
- **`action_quit()` refactored**: Spawns a `cleanup_worker` (threaded) that calls `_cleanup_and_save_session()`, then schedules `self.exit()`. No blocking on the main event loop.

### Fixed
- **`TuiAdapter.emit("error")` not styled**: `"error"` event type was missing from `type_to_msg` map — fell through to default `"agent"` styling (no red). Added `"error": "error"` mapping; retry errors now show red background.
- **`ctrl+q` frozen TUI**: `ctrl+q` binding tried to `await action_quit()` directly on the event loop, blocking all input. Now uses `run_worker()` (same as `/exit`).
- **Session messages printed on mouse move**: `loop.py` had `print(msg)` fallback when adapter not set. Removed — TUI noise eliminated.
- **Loader inserted after messages**: Messages appended to bottom of stream, appearing *after* the loader. Fixed — loader stays at bottom; all new messages mount with `before=loader_widget`.
- **`call_from_thread` in same-thread context**: Cleanup worker now uses `thread=True` — `call_from_thread` for status messages works correctly without "same thread" RuntimeError.
- **Dead code removed**: `~/.dostuff/project/.env` path removed from `.env` lookup order (never used).

### Removed
- **`--user` flag**: Removed from `cli.py` and `DostuffTUI.__init__`. Single-user only.

## [0.1.2] - 2026-09-04

### Added
- Initial PyPI release (`dostuff` 0.1.2).
- Textual TUI with ASCII banner.
- Persistent session storage (SQLite).
- MCP integration (configurable servers).
- Semantic memory (ChromaDB).
- Multi-model support via litellm.
- `dostuff init`, `dostuff config`, `dostuff doctor`, `dostuff session-list` subcommands.
- Session resume (`--session <id>`).
- 3 retries on model call failures.
- ESC cancel binding (priority=True).
- Loader widget in conversation stream.
- Tool call / tool result colored messages.

## [0.1.1] - 2026-09-04

### Added
- Python 3.11+ compatibility fixes (f-string `.get()` syntax).
- `get_session_meta_sync()` added to `SQLiteSessionStore`.
- `README.md` updated for 3.11+.
- `FUTURE.md` created with session resume + 4 pending features.

### Fixed
- Python 3.10 stale chroma → wipe `~/.dostuff/data/chroma`.
- Fixed `dostuff` 1.0.0 PyPI conflict (another publisher owns it; pip version sorting picked 1.0.0 over 0.1.1).

## [0.1.0] - 2026-09-03

### Added
- First package build (`dostuff`).
- `pyproject.toml`: `name="dostuff"`, `requires-python=">=3.11"`, `chromadb>=0.4.0,<1.0.0`.
- `cli_tui.py` skeleton with Textual TUI framework.
- `dostuff/config.py`: `Config()` with `.env` loading at import time.
