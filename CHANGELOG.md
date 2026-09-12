# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2026-09-12

### Added

- **Streaming agent response** (`stream=True`): `_consume_stream()` accumulates `delta.content`/`delta.tool_calls`; adapter `emit("partial")`; live widget updates; `stream_options={"include_usage": True}`.
- **Usage tracking in stream mode**: `stream_options={"include_usage": True}` added; final usage-only chunk captured (`choices=[]`, `usage` non-null) after `continue` check removed.
- **Prompt queue during work**: `self._prompt_queue` + `_process_prompt_queue()`; queued prompts execute sequentially.
- **Live timer (`threading.Thread`)**: Updates status bar `⏳ {elapsed}s`; loader shows `Working...` (no freeze text, no timer embedded).
- **Loader persistence / removal**: `_freeze_final_time()` removed; loader stays mounted; messages mount `before=loader_widget`.
- **Manual `/compact` command**: `_compact_session()` loads DB truth, compacts with `0.15 * token_limit` budget (capped 20K), 60s timeout, loader, DB save (`compaction_notes` + `working_history`), and adapter sync.
- **Console trace disabled:** `ConsoleSpanExporter` removed from `tracing.py`; OTLP file trace (`~/.dostuff/data/traces/{sid}.jsonl`) planned for Jaeger.
- **Structured compaction summary:** `compact_context` prompt now requires sections (Goal, Constraints, Progress, Decisions, Next Steps, Critical Context, Turn Context); existing `compaction_notes` passed to LLM for merge/update.
- **ESC cancel confirmed done:** cancellation event wired through adapter + loop; stream interruption verified.
- **Context window percentage**: status bar shows `%` of context window filled from turn-level `prompt_tokens` / `token_limit`; persisted in DB via `last_input_tokens` and restored on resume.
- **Auto compaction**: threshold set to 50% of context window; uses turn-level prompt tokens; compacts before iteration loop.
- **Context window percentage**: status bar shows `%` of context window filled from turn-level `prompt_tokens` / `token_limit`; persisted in DB via `last_input_tokens` and restored on resume.
- **Auto compaction**: threshold set to 50% of context window; uses turn-level prompt tokens; compacts before iteration loop.
- **Token persistence**: DB `prompt_tokens`/`completion_tokens`/`total_tokens` saved per turn; `last_input_tokens` column added for context percentage.
- **Session last-used time**: `sessions.last_used` updated on every `save_session_meta`; shown in `session-list`.

### Fixed

- **Tool call accumulation in stream**: Captured `fn.name` from `fn.name` (not `arguments` fallback); `tc_key` uses `index` (primary) or `id` (fallback); instance-bound `lambda` for `model_dump()`.
- **Rate limit / usage omission**: `stream_options` added; universal provider limitation confirmed (stream chunks omit usage except final chunk).
- **Token counts hidden when 0**: `token_str` conditional; non-streamed turns show real usage, streamed turns don't until final chunk.
- **`GeneratorExit` / `BaseExceptionGroup` suppression**: Added to `_consume_stream()` for `mcp` library anyio generator cleanup during stream end.
- **MCP `streamable_http_client` crash (`RuntimeError: cancel scope`)**: Suppressed in stream loop; cleanup handled.
- **Package sync**: Source files (`cli_tui.py`, `loop.py`, `call_agent.py`, `config.py`, `extract_semantic_memories.py`, `extract_episodic_memory.py`) copied to installed `dostuff/` package after each edit.
- **`.env` lazy loading** (`Config.load()`), `--user` removal, loader persistence, live timer, queue, `ctrl+q` freeze, `call_from_thread` fix, `_fmt_tokens` consolidation already in 0.1.3; stream completes 0.1.3 work.

### Changed

- **Status bar formatting**: `📁 {cwd} • sess: {sid} • ↑{p} ↓{c}{loader}`; "new"/"resumed" keywords removed; loader no longer freezes final time.
- **Loader behavior**: `_clear_loader()` added then removed (caused abrupt app close when called from `run_worker` thread without `call_from_thread`).
- **`stream=True` architecture verified**: `await acompletion(..., stream=True)` (not raw coroutine); `types.SimpleNamespace` used (no `FakeResponse`); `_update_streaming_widget()` replaces and accumulates cumulative text.

### Removed

- `--user` flag removed (already in 0.1.3).
- `_freeze_final_time()` removed (loader no longer freezes; kept visible).

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
- **Loader inserted after messages**: Messages appended to bottom of stream, appearing _after_ the loader. Fixed — loader stays at bottom; all new messages mount with `before=loader_widget`.
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
