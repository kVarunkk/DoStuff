# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`web_fetch` tool**: GET a URL; HTML becomes Markdown-ish text (headings, lists, links, `<pre>`; scripts/styles dropped), JSON/XML/text as-is, binary refused; `max_chars` (default 20 000), downloads stop at 5 MB. 4xx/5xx HTML pages return only the header and a note.
- **`web_search` tool**, no API key: SearXNG if `search.searxng_url` is set, else the `ddgs` package (`dostuff[search]`), else DuckDuckGo's HTML page. Title / URL / snippet per result, `max_results` 1-20.
- New permission kind `network` for `web_fetch` / `web_search`: asks in `ask` / `allowlist`; "always" and `permissions.allow` work per domain (`web_fetch:docs.python.org`) or for every search (`web_search`). Both are in the subagents' default tools.
- **Settings tools** (`tools/settings.py`): `read_settings`, `edit_config` (exact-text edit of `config.yaml`, keeps comments, refuses invalid YAML, warns on unknown keys), `set_mcp_server` / `remove_mcp_server`; `scope` is `user` or `project`. Changes apply live (`Config.reload()`, status bar, iteration cap, MCP servers reconnected). New permission kind `settings` asks in every mode with no "always"; writes inside a `.dostuff` folder are treated the same. Not available to subagents.
- **`@path` mentions**: `@src/app.py` attaches the file, `@src` a folder listing (`@"a b.txt"` for spaces); non-paths stay plain text, `.env*` and binaries skipped. `@` opens a completion menu in the TUI; also works in `dostuff -p`.
- **Cost per turn and per session**, priced with litellm's table (cached tokens included); stored in `sessions.cost_usd` (schema v3), shown in the status bar, `/status` and at the end of each turn; `cost_usd` in `dostuff -p --json`.
- **TUI**: Shift+Enter (also Ctrl+Enter, Alt+Enter, Ctrl+J, `\` + Enter) inserts a newline; Up/Down prompt history saved per directory; Ctrl+O expands/collapses truncated tool results (click toggles one); select text to copy (OSC 52, plus `clip.exe` on Windows) while the input keeps focus.
- **Memory commands** `/episodes` and `/forget <key or id>`.
- **Configurable status bar** (`ui.status_bar`): it fits the terminal by shortening the cwd/model and dropping the least important items, and redraws on resize.
- `dostuff --version` (`-V`); `python -m dostuff` (and `python -m dostuff.cli`) runs the CLI.

### Changed
- **chromadb is optional** (`pip install "dostuff[memory]"`, included in `full`). Without it the memory tools aren't offered, recall is skipped, and `search_mcp_tools` uses keyword search.
- **Faster startup** (~9 s → under 1 s to the first frame): litellm, chromadb and mcp load in a thread after the first frame.
- Chroma's embedding model downloads at startup with a message instead of silently during the first turn; `dostuff doctor` reports it.
- **Project instructions are combined**: global `~/.dostuff/AGENTS.md` (or CLAUDE.md), then one file per folder from the outermost parent down to the cwd.
- Memory recall leaves out facts below 0.25 similarity; the memory resolver skips the model call when nothing related exists; `save_episode` skips near-duplicates (≥ 0.85). The unused `context_window` episode field is gone.
- System prompt: a "The web" section (search before guessing URLs / packages / config values, never present placeholders as real) and guidance for adding MCP servers.
- Tool result previews are capped by characters (8 × 200) as well as lines.
- `pyproject.toml` Homepage points at the repo root; CI also tests Python 3.14 and installs `.[dev,memory]`.

### Fixed
- **MCP OAuth with confidential clients** (Supabase 422 at the token endpoint): the registration's `client_secret` and auth method are saved and sent (`client_secret_post` or HTTP Basic); cached registrations missing a needed secret are redone. Scope follows the MCP spec (`WWW-Authenticate`, else the resource's `scopes_supported`). OAuth errors include the server's message.
- `python -m dostuff.cli ...` exited 0 with no output.

## [0.2.0] - 2026-10-23

Fixes from the v0.1.6 package review (`plan/2026-10-23-package-review-fixes/`).

### Security
- **Permission layer** (`agent/permissions.py`): approval is decided before dispatch, never by the model. `_confirmed` / `overwrite` are gone from tool signatures and dropped if a model sends them. Modes `permissions.mode: ask | allowlist | auto`; prompt accepts `y` (once), `a` (always this session), anything else denies; a denial is returned to the model as the tool result.
- **Shell allow-list** replaces the bypassable `DANGEROUS_PATTERNS` denylist: a command runs without asking only if every segment is read-only and it has no redirection/substitution.
- **File sandbox**: every file tool rejects paths outside the project root (no more silent re-rooting of `/etc/passwd`); `.env*` and `~/.dostuff/` can't be read.
- **Tracing** defaults to off in `config.example.yaml`; span content is redacted unless `tracing.capture_content: true`.
- **Memory provenance**: facts carry `source` (`user_message` / `assistant`); the agent is instructed never to save facts seen only in tool output. `/memories` lists saved facts.
- **MCP OAuth**: random `state` verified, RFC 8707 `resource` sent, 120 s callback timeout, `?error=` handled, port 8085 always released, refresh uses the registered `client_id`. Optional keyring token storage (`dostuff[keyring]`).

### Added
- **Memory tools** `save_memory` (key/value fact; overlapping facts are updated via the ADD/UPDATE/NOOP resolver) and `save_episode` (DECISION / PREFERENCE / CORRECTION / EVENT / LEARNING): the agent saves memories during the session instead of an LLM extraction pass at exit. Saves don't ask for approval, are shown in the TUI, and aren't available to subagents.
- Tools: `edit_file` (exact replacement, unique-match check, `replace_all`), `grep`, `glob` (pure Python, skip `.git`/venvs/`node_modules`/`build`/`dist`); `read_file` gains `offset`/`limit` and line numbers.
- `<environment>` block in the system prompt every turn (cwd, OS, shell, date, git branch/status); restored tool-usage guidance.
- Config keys: `loop.context_window`, `loop.max_tool_output_chars` (default 30 000), `memory.episodic_scope` (`project`), `models[].api_key_env`; MCP server `connect_timeout`.
- `dostuff init --force`; `--session <unknown id>` asks before starting a new session.
- **First-run setup**: launching `dostuff` with no global config and no `MODEL` asks for provider, model and API key (hidden; saved to `~/.dostuff/.env`), writes `~/.dostuff/config.yaml` and offers a test request. `dostuff setup` re-runs it (old config kept as `config.yaml.bak`). No TTY → one-line hint only.
- **Headless mode** `dostuff -p "<prompt>"`: one turn, answer on stdout, tool activity on stderr (`-q` quiets). `--json`, piped stdin appended (`-p -` reads the prompt from stdin), `--session`, `--permission-mode`. Approvals asked on a TTY, else denied. Ctrl+C exits 130.
- **Sessions**: `dostuff resume [<id>]` (latest session in this directory), `dostuff sessions [--all] [-n N]`, `sessions rename`, `sessions rm`, `sessions prune [--older-than 30d] [--dry-run]`; ids accept a unique prefix. Session titles (schema v2). `dostuff export [<id>] [-o file|-]` / `/export` write Markdown.
- **Models**: `/model` picker (saved models, then each provider's catalogue, filterable); picking an unsaved model adds it to the global config. `/model <name> [--active]`; `dostuff models add <provider/model> [--active] [--project] [--api-key-env VAR]` and `dostuff models list`. Model lists come from litellm's catalogue, OpenRouter's public API and the local Ollama server; config edits keep comments.
- **TUI commands**: `/` command menu (filter, Tab completes, Enter runs); `/resume` picker; `/rename`; `/status`; `/permissions [mode]` (session-only, shown in the status bar); `/undo [force]` restores files from per-turn checkpoints of `write_file` / `edit_file` / `delete_file` (incl. subagents; not shell changes).
- **`!command`** runs a shell command directly from the input (no approval); command and output are saved so the model sees them next turn.
- **`todo_write` tool**: visible task list for multi-step work, rendered as a checklist.
- **Prompt caching** for Claude models (anthropic, and Claude on bedrock / vertex_ai / openrouter); cached tokens shown in the status bar. `loop.prompt_caching: false` turns it off.
- `dostuff mcp list / add / remove` (stdio or `--url`; `--project` writes the project file).
- `bash_command` `timeout` parameter (default 60 s, max 110 s). Config keys `loop.max_retries` (default 6) and `loop.retry_max_delay` (default 60 s).
- Tests for schemas, permissions, subagents, file/search tools, loop robustness, DB migration, config, memory, MCP end-to-end, CLI commands and the TUI; GitHub Actions CI (pytest + mypy).

### Changed
- **TUI runs turns, `/compact`, `/clear` and exit cleanup as async workers** on Textual's loop (no threads); the timer uses `set_interval`. Reasoning text streams into one widget per model call; streamed answers aren't duplicated; all tool calls of a resumed step are rendered.
- **Agent loop**: compaction is checked after every model call; retries only rate-limit/timeout/5xx/connection errors, with jittered exponential backoff that honours `Retry-After` (context overflow compacts then retries once; auth/bad-request fail fast); an interrupted stream is marked in the UI; an empty response gets one nudge; malformed tool-call JSON becomes a tool error; read-only tool calls run in parallel, mutating ones in order; tool output over the cap keeps head + tail; on ESC, finished tool results are kept.
- `consume_stream` keeps provider tool-call ids and `finish_reason`. `call_agent` uses `drop_params=True` and sends `stream_options` / Gemini thinking params only when they apply.
- Compaction summarises in chunks (each request < 50 % of the window) and always makes progress. Unknown models default to a 128k window with a one-time warning.
- **Session DB**: working history is stored as rows in `working_steps` (one INSERT per step; compaction flags rows). Existing DBs are migrated on first open (`PRAGMA user_version` 1); the old column is emptied, not dropped. One connection per store.
- **Memory**: episodic recall is per user (and per project directory by default) instead of per session; per-fact JSON guards; UPDATE with an unknown key becomes ADD; tz-aware timestamps; one shared Chroma client; Chroma calls run in threads.
- **MCP**: tools are keyed `server__tool`; servers connect concurrently with per-server timeouts; each connection lives in its own task (clean shutdown, no cross-task errors); `search_mcp_tools` only returns connected servers' tools and falls back to token-overlap search; tool calls re-authenticate once on HTTP 401.
- **Config**: `.env` read from `~/.dostuff/.env` then `./.dostuff/.env` (project overrides user; real env wins), loaded once; `get_config()` singleton; model read at call time; `model.api_key_env` honoured.
- Version comes from package metadata (`importlib.metadata`), used by the banner and `dostuff doctor`.
- Default identity is neutral ("DoStuff, a coding agent").
- Packaging: `mcp>=2.0`, `httpx2` declared, `setuptools>=77`; extras `keyring`, `dev`. Templates ship in the package (`dostuff/templates/`).
- **`/clear` starts a new session**; the old one stays resumable. Sessions are only created with their first step (no empty sessions per launch).
- **Approval prompts are a selector** (arrows / number / `y` `a` `n`); several calls in one response are approved together ("allow all", "review each", "no to all"). Denying takes an optional note that is passed to the model.
- **TUI look**: no message backgrounds, Claude Code-style prefixes (`>`, `●`, `⎿`), grey band on user messages, terminal background shows through, auto-scroll follows streaming output, Gemini reasoning rendered as formatted text.
- Project MCP config is merged over the global one per server; `mcp.config_path` is honoured; a broken JSON file is skipped with a message.
- `dostuff doctor` shows ✓/✗ checks with fixes and exits 1 on failure; `config` / `doctor` no longer create files.
- README quick start is `pip install dostuff` → `dostuff`.

### Fixed
- Subagents: keyword call into `loop`, `tool_names` enforced, `delete_file`/`delegate_to_subagent` never granted, valid seed message, parent adapter/cancel state passed through, `loop.subagent_max_iterations`.
- Tool schemas generated with pydantic from the real signatures (`dict` → `object`, arrays have `items`, nullable unions).
- Subagents are no longer killed by the 120 s tool timeout, and return partial results (a wrap-up or a step digest) when they stop early.
- Long-running shell commands froze the TUI on Windows: commands now run detached with no stdin, and the whole process tree is killed on timeout or ESC.
- Unrelated past episodes were recalled in every session; recall now needs a minimum similarity.
- Exit no longer waits on memory extraction.

### Removed
- `learn_from_session.py` (dead, referenced a non-existent tool), `extract_text.py`, exit-time memory extraction (`save_memories_and_exit.py`, `extract_semantic_memories.py`, `extract_episodic_memory.py`, `format_transcript.py`) and `memory.exit_timeout_s`, `InMemorySessionStore`, `SQLiteSessionStore.save()`, `WORKSPACE_ROOT` / `SKILLS_ROOT` / `system_prompt_removed`, `random_file.txt`, `build/`, `dostuff_test.egg-info/`.

## [0.1.6] - 2026-09-21

### Added
- **Configurable max iterations:** `loop.max_iterations` (default `MAX_ITERATIONS` = 100) via `config.yaml`; `cli_tui` passes to `loop`; no hardcoded fallback.

## [0.1.5] - 2026-09-05

### Fixed
- **DB migration `last_input_tokens`:** `session_store.py` `_init_db` now runs `ALTER TABLE sessions ADD COLUMN last_input_tokens INTEGER DEFAULT 0`; fixes `OperationalError: no such column` on fresh install / resume with older DB.

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