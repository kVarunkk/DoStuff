<div align="center">

# DoStuff

**A pip-installable AI agent harness with a Textual TUI, persistent sessions, and MCP tool support.**

[![PyPI](https://img.shields.io/badge/PyPI-published-3776ab?style=flat-square&logo=pypi)](https://pypi.org/project/dostuff/)

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![litellm](https://img.shields.io/badge/litellm-any--provider-blueviolet?style=flat-square)](https://docs.litellm.ai/docs/providers)
[![MCP](https://img.shields.io/badge/MCP-compatible-success?style=flat-square)](https://modelcontextprotocol.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![textual](https://img.shields.io/badge/UI-textual-ff6b6b?style=flat-square)](https://textual.textualize.io)

[Install](#install) • [Quick start](#quick-start) • [Tools & permissions](#tools-and-permissions) • [Configuration](#configuration) • [TUI controls](#tui-controls) • [MCP servers](#mcp-servers) • [Skills](#skills) • [Troubleshooting](#troubleshooting)

</div>

DoStuff is a terminal AI agent that runs anywhere `pip` runs. It gives you a full interactive session with persistent memory, multi-provider LLM support via [litellm](https://docs.litellm.ai), and pluggable [MCP](https://modelcontextprotocol.io) tool servers. State lives in `~/.dostuff/`, so any project you launch `dostuff` from shares the same memory and sessions.

> [!TIP]
> If you have used `claude-code` or `aider`, this will feel familiar. DoStuff is opinionated, minimal, and easy to extend.

---

## Features

- **Textual TUI** with a `/` command menu, `@file` completion, `!shell` commands, an in-app config editor, a configurable status bar and a live turn timer
- **Headless mode** — `dostuff -p "<prompt>"` runs one prompt and prints the answer (`--output-format json|stream-json`, `--max-turns`, `--allowed-tools` for scripts and CI)
- **Multi-provider LLMs** via [litellm](https://docs.litellm.ai/docs/providers) — OpenAI, Anthropic, Gemini, Groq, Mistral, Ollama, OpenRouter, OpenCode Zen / Go
- **Persistent sessions** in SQLite; resume with `dostuff -c`, `dostuff resume` or `dostuff --session <id>`, branch one off with `/fork`
- **Git worktrees** — `/worktree` or `dostuff -w` runs a session on its own branch and checkout; subagents can run in their own worktree too
- **Long-term memory** (semantic + episodic) backed by ChromaDB — optional, `pip install "dostuff[memory]"`
- **MCP server support** — stdio (npx, uvx) and HTTP transports
- **`@file` mentions** — type `@` in the prompt to pick a file or folder; its content (or listing) is sent with the message
- **Project instructions** — `AGENTS.md` / `CLAUDE.md` from `~/.dostuff/`, the repo root and every folder down to the current one, combined
- **Coding tools** — `read_file` (line numbers, offset/limit; it also opens images and PDFs for a model that can view them, and Word / Excel / PowerPoint files as text), `edit_file` (exact replacement), `multi_edit` (several replacements in one file, all or nothing), `write_file`, `grep`, `glob`, `list_files`, `delete_file`, `bash_command`, `web_fetch` (read a URL as text), `web_search` (free, no API key)
- **`ask_user`** — the agent asks you a real question — up to 4, with 2–4 options each, multi-select and a free-text "Other" — instead of guessing; the answers come back to it as one result and stay readable in `/resume`. See [Asking the user](#asking-the-user)
- **`/btw` side questions** — ask about the conversation while a turn runs (or idle) without stopping or queuing it: one model call, no tools run, answered in a panel above the input and never added to the history; `f` hands the question to a background agent that can check it with tools. See [Side questions](#side-questions-btw)
- **Mid-turn steering (`/steer`)** — send a correction to the turn that is running and the model reads it at the next tool boundary instead of waiting for the turn; plain messages steer by default, and `ui.mid_turn_messages: queue` batches them after the turn instead. See [Mid-turn steering](#mid-turn-steering-steer)
- **Permission layer** — mutating tool calls ask for approval (allow once / always this session / deny with a note); read-only shell commands run without asking; modes `ask` / `allowlist` / `auto`; `/undo` reverts the agent's file changes
- **Hooks** — shell commands on `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop`, `SubagentStart` and `SubagentStop` that can block a tool call or prompt, feed a formatter's output back to the model, or keep the turn going; Claude Code's hook I/O contract, and project hooks run only once you trust them. See [Hooks](#hooks)
- **Background jobs** — `bash_command(run_in_background=true)` starts dev servers, watchers and long test runs as jobs the agent reads with `bash_output` (`wait_for` a ready line) and stops with `kill_shell`; it hears when one ends. `Ctrl+B` moves a running command to the background, `/jobs` lists them, and none outlives the session. See [Background jobs](#background-jobs)
- **File sandbox** — file tools stay inside the project directory, plus read-only access to `.dostuff` (the project's and `~/.dostuff/`) and to discovered skill folders; `.env*` files, `mcp_config.json` and `.dostuff/data` (past sessions and prompts, saved MCP client secrets) are never readable
- **Skills loader** — drop a `SKILL.md` in the right folder, restart, done; run one with `/<skill-name> [args]`, name one mid-message to allow it for that message, or keep one manual-only with `disable-model-invocation`. See [Skills](#skills)
- **Subagents** — the `agent` tool starts agents in parallel, in the foreground or the background, from built-in types (`explore`, `plan`, `general-purpose`), your own Markdown agent files (Claude Code's `.claude/agents/` work too) or a `fork` of the conversation; continue one with `send_message`, stop it with `stop_agent`, see them all in `/agents`. See [Subagents](#subagents)
- **Per-tool timeout** (120s) to prevent hung servers; long tool output is truncated (head + tail) before it reaches the model
- **Retries with backoff** for rate limits, timeouts and 5xx errors (6 by default, exponential with jitter, honors `Retry-After`; `loop.max_retries` / `loop.retry_max_delay`); auth / bad-request errors fail fast
- **Token and cost tracking** — per-turn and cumulative totals in `/status` and the line after each turn (`✻ Worked for 1m 5s · $0.01`); cost in USD from litellm's price table (hidden when the model has no known price). The table bundled with litellm is used, so startup doesn't wait on a download; set `LITELLM_LOCAL_MODEL_COST_MAP=False` to fetch the latest one from GitHub instead
- **Long-term memory tools** — the agent saves facts (`save_memory`) and notable moments (`save_episode`) as they come up; both are recalled in later sessions
- **Graceful exit** — `Ctrl+Q` or `/exit` saves the session, cleans up, and stops background workers
- **Clean Cancellation** — `Escape` can cancel in-flight agent turns cleanly without corrupting the session
- **2-state session split** — `session_steps` (full audit) / `working_history` (LLM payload); DB persists both plus `compaction_notes` + `last_input_tokens`
- **Auto & Manual Compaction** — Manual `/compact` runs compaction and saves it. Auto-compaction triggers automatically during the agent loop if context usage (`last_input_tokens`) exceeds `loop.auto_compact_threshold` (default 50%) of the model's token limit; `off` disables it (an overflowing window is still compacted), and `/compact auto <N|off>` changes it for the session. It keeps at least 15% or 20k tokens of the most recent steps in the working history, summarizing the rest into `compaction_notes` which are appended to system instructions.
- **Tracing (opt-in)** — OTLP tracing support built into `lib/tracing.py`; prompt and tool content are redacted from spans by default

---

## Install

Install from PyPI:

```bash
pip install dostuff
```

This installs the `dostuff` command on your `PATH`. Done.

Optional features come as extras:

| Install | Adds |
|---|---|
| `pip install dostuff` | Everything except the extras below, including MCP (with OAuth tokens kept in the OS keyring) |
| `pip install "dostuff[memory]"` | Long-term memory (facts and past decisions across sessions) and semantic MCP tool search, via ChromaDB. Large download |
| `pip install "dostuff[search]"` | Steadier `web_search` through the `ddgs` package (several engines) instead of DuckDuckGo's HTML page |
| `pip install "dostuff[docs]"` | Word, Excel and PowerPoint files (`.docx`, `.xlsx`, `.pptx`), and PDFs for models that can't read them, converted to text for `@path` and `read_file` (`pypdf`, `python-docx`, `openpyxl`, `python-pptx`) |
| `pip install "dostuff[images]"` | Pillow: large images are downscaled to fit the model's limit, and `Alt+V` reads the clipboard directly |
| `pip install "dostuff[full]"` | `memory` + `search` + `docs` + `images` |
| `pip install "dostuff[dev]"` | `pytest` and `mypy`, for working on dostuff itself |

Extras combine, e.g. `pip install "dostuff[memory,search]"`. Without `memory` everything else
works: the memory tools are hidden from the model and MCP tool search matches keywords instead of
meaning. `dostuff doctor` shows whether memory is on.

### Other install methods

For an isolated install with [pipx](https://pypa.github.io/pipx/):

```bash
pipx install dostuff
pipx install "dostuff[full]"   # with extras
```

For development (editable mode, from a clone):

```bash
git clone https://github.com/kVarunkk/DoStuff.git
cd DoStuff
pip install -e .           # core
pip install -e ".[memory]" # + long-term memory (chromadb)
pip install -e ".[search]" # + ddgs web search
pip install -e ".[full]"   # + memory + search
pip install -e ".[dev]"    # + pytest + mypy

python -m pytest -q tests
python -m mypy dostuff --ignore-missing-imports
```

Verify:

```bash
dostuff --help
```

---

## Quick start

```bash
cd your_project
dostuff
```

The first launch runs a short setup: pick a provider and model, paste the API key (input is hidden),
and optionally send a one-token test request. It writes `~/.dostuff/config.yaml` and stores the key in
`~/.dostuff/.env`. Re-run it anytime with `dostuff setup`, and check your install with `dostuff doctor`.

Setup never runs without a terminal (CI, pipes, containers): set `MODEL` and the provider key as
environment variables instead.

In the TUI:

```text
> hi there
[agent replies]

> /exit
[session saved, goodbye]
```

> [!NOTE]
> Skipped setup? DoStuff still starts with built-in defaults (`openai/gpt-4o-mini`) and asks again on the next launch. See [Troubleshooting](#troubleshooting).

---

## CLI reference

| Command                  | Description                                     |
| ------------------------ | ----------------------------------------------- |
| `dostuff`                | Launch the interactive TUI                      |
| `dostuff --version`      | Print the version (`-V`)                        |
| `dostuff --session <id>` | Resume a prior session (asks before starting a new one if the ID is unknown) |
| `dostuff -c` / `--continue` | Resume the latest session in the current directory (same as `dostuff resume`; works with `-p` too) |
| `dostuff -w` / `--worktree-name <name>` | Start the session in a new git worktree (`.dostuff/worktrees/<name>`, branch `dostuff/<name>`) |
| `dostuff init [--force]` | Create `.dostuff/` config/skills in current dir (won't overwrite without `--force`) |
| `dostuff setup`          | Pick a model, save its API key, write `~/.dostuff/config.yaml` (runs automatically on first launch) |
| `dostuff config`         | Show resolved config paths and values           |
| `dostuff doctor`         | Health check: config, API key, MCP, data dir; prints a fix for each problem |
| `dostuff memory [check\|repair] [-y]` | Check that long-term memory's indexes can answer queries (`check`, the default), or rebuild a broken one after "Error finding id" errors (`repair`, which backs it up to `~/.dostuff/data/chroma-backups/` first). Stop dostuff before repairing |
| `dostuff resume [<id>]`  | Resume the latest session started in the current directory (or the given one; a unique ID prefix is enough). `--fork` resumes a copy, `--worktree` moves it into a new worktree |
| `dostuff sessions [--all] [-n N]` | List sessions in the current directory with their title or first prompt (`--all`: every directory) |
| `dostuff sessions rename <id> <title>` | Name a session |
| `dostuff sessions rm <id>... [-y]` | Delete sessions and their `/undo` checkpoints |
| `dostuff sessions prune [--older-than 30d] [--dry-run] [-y]` | Delete empty sessions, and sessions not used for the given time (`45m`, `12h`, `30d`, `2w`) |
| `dostuff export [<id>] [-o file\|-]` | Save a session (default: latest here) as Markdown: prompts, answers, tool calls and results |
| `dostuff -p "<prompt>"`  | Headless: run one prompt, print the answer to stdout, exit. Progress goes to stderr. Piped stdin is appended to the prompt (`-p -` reads the prompt from stdin). `--session <id>` continues a session, `--output-format json` (or `--json`) prints `{result, session_id, usage, ...}`, `--output-format stream-json` prints one JSON event per line (`init`, `assistant`, `tool_call`, `tool_result`, `usage`, `error`, then `result`), `-q` hides progress, `--permission-mode ask\|allowlist\|auto\|plan`. `--max-turns N` stops after N model calls (exit code 1). `--allowed-tools` (comma-separated or repeated) runs matching calls without asking in any mode, in the `permissions.allow` syntax (`"npm test"`, `web_fetch:docs.python.org`) or a tool name for every call of it (`write_file`, `bash_command`). Without a terminal, other calls that need approval are denied. `--trust-project-hooks` runs the project's [hooks](#project-hooks-and-trust) without a saved trust answer |
| `dostuff mcp list`       | Show MCP servers and the file each comes from |
| `dostuff mcp add <name> [-e K=V] -- <command> [args...]` | Add a stdio server; `--url <url> [-H "Name: value"]` for HTTP; `--project` writes `./.dostuff/mcp_config.json` |
| `dostuff mcp remove <name> [--project]` | Remove a server |
| `dostuff models list`    | Show the configured models (`*` = in use) and the file they come from |
| `dostuff models add [<provider/model>] [--active] [--api-key-env VAR] [--project]` | With no name: pick a provider, then a model from its list (type to filter, or any name). Add a model to `models:` in `~/.dostuff/config.yaml` (comments are kept); `--active` makes it the default, also for one already listed; `-P <provider>` lets you give a bare model name |

---

## TUI controls

| Key / command | Action                              |
| ------------- | ----------------------------------- |
| `Enter`       | Submit message                      |
| `/`           | Open the command menu (type to filter, `Tab` completes, `Enter` runs) |
| `!<command>`  | Run a shell command directly (no approval); the command and its output are shown to the model on the next turn |
| `# <note>`    | Add `- <note>` to `AGENTS.md`: you pick this project's or the global one (`~/.dostuff/AGENTS.md`); the agent follows it from the next message |
| `Shift+Enter` | Insert a newline (multi-line input). If your terminal sends it as plain Enter, use `\` then `Enter`, `Ctrl+Enter`, `Alt+Enter` or `Ctrl+J` |
| `@path`       | Attach a file (numbered lines) or a folder listing to the message. Typing `@` lists matching project paths: `Tab` / `Enter` completes, a folder opens for the next part. `@"a b.txt"` for spaces. Also works with `dostuff -p`. `.env` files and binaries are never attached. An image (PNG, JPEG, GIF or WebP, told apart by its content, not its extension) is sent to the model as an image when the model can read images, and shows as `[Image #1 · shot.png · 1280×720]`; on a model that can't, or over 3.75 MB, or more than 20 in one message, nothing is sent and your prompt goes back in the input (`-p` exits with code 1). With `dostuff[images]`, an image wider or taller than 1568 px, or over 3.75 MB, is downscaled first (a GIF only when it is over 3.75 MB). A PDF goes to the model as a document when the model reads PDFs (up to 32 MB and 100 pages), shown as `[PDF · report.pdf · 12 pages]`; otherwise its text is attached. Word, Excel and PowerPoint files (`.docx`, `.xlsx`, `.pptx`) are attached as text: headings and tables as markdown, one table per sheet (values, not formulas), one section per slide with its notes. A converted file over 200 lines (`loop.paste_file_lines`) is saved to `.dostuff/attachments/` and the model reads it from there. Converting needs `dostuff[docs]`; without it, or for the old `.doc` / `.xls` / `.ppt` formats, the prompt goes back with the reason. Images and PDFs are copied to `.dostuff/attachments/` and removed after `loop.attachment_retention_days` (30). For a model dostuff can't tell about, set `vision: true` / `pdf_input: true` (or `false`) on its `models:` entry |
| `Tab`         | While typing a slash command, accept the dim ghost text after the cursor: the rest of the command name, then its flag (`/model ` → `--active`, `/fork ` → `--worktree`). `Right` accepts it too |
| Paste        | A paste over 10 lines (`ui.paste_collapse_lines`) or 2,000 chars shows as `[Pasted text #1 +243 lines]` and is expanded when you send it. Over 200 lines (`loop.paste_file_lines`) it's saved to `.dostuff/pastes/` and the model reads it from there, so it doesn't fill the context; files are removed after `loop.paste_retention_days` (7). Dragging an image or document into the terminal pastes its path; dostuff turns it into an `@path` mention, or, for an image outside the project, attaches it like `Alt+V` |
| `Alt+V`       | Paste an image from the clipboard (a screenshot, an image copied in a browser). It shows as `[Image #1]` in the input, and is sent with your message under the same rules as an `@image` mention (a model that can read images, 3.75 MB, 20 per message). `Ctrl+V` stays the terminal's own paste, which carries only text. Uses Pillow when installed (`dostuff[images]`), otherwise PowerShell on Windows, `osascript` on macOS, and `wl-paste` or `xclip` on Linux. Checked on Windows; macOS and Linux are untested so far |
| `Up` / `Down` | On the first / last line of the input: previous / next prompt typed in this directory (saved in `~/.dostuff/data/prompt_history.jsonl`) |
| `Ctrl+R`      | Search the prompts typed in this directory (type to filter, `Enter` puts the prompt in the input) |
| `Esc` `Esc`   | With an empty input: take back your last prompt. It and everything after it (answer, tool calls) leave the conversation and the prompt goes back in the input to edit and resend. Files that turn changed stay as they are; `/undo` restores them |
| `Ctrl+O`      | Expand / collapse every truncated tool and shell result (click one to toggle just that one) |
| `Ctrl+B`      | While the agent waits on a shell command or a foreground subagent (the loader says `Ctrl+B to background`): move them to the background; the agent gets what a command printed so far and carries on, and hears from the agent when it ends. Not for `!command`s. See [Background jobs](#background-jobs) and [Subagents](#subagents) |
| `Shift+Tab`   | Cycle the permission mode for this session: ask → auto → plan → allowlist (shown in the status bar) |
| `Ctrl+Backspace` / `Ctrl+W` / `Alt+Backspace` | Delete the previous word; `Ctrl+Delete` / `Alt+Delete` the next one |
| `/agents` / `/agents types` / `/agents reload` | This session's agents (running first): pick one to read its transcript, answer its approval prompt or stop it / the agent types with their source, model and tools (`/agents types <name>`: one type's system prompt and config) / re-read the agent files |
| `/exit`       | Save the session and quit           |
| `/clear`      | Start a new session; the old one stays resumable |
| `/resume [<id>]` | Pick another session from this directory (or switch to the given one) |
| `/fork [title] [--worktree]` | Copy this session into a new one and switch to it; the original is unchanged |
| `/worktree [name]` / `/worktree exit` | Move the session into a new git worktree on branch `dostuff/<name>` / leave it (removed if unchanged, otherwise keep or remove) |
| `/permissions [mode]` | Show or change the permission mode for this session (`ask`, `allowlist`, `auto`, `plan`) |
| `/copy`       | Copy the model's last answer to the clipboard |
| `/btw <question>` | Ask a side question about the conversation, even while a turn runs: answered in a panel above the input, without tools, never added to the history. See [Side questions](#side-questions-btw) |
| `/steer <message>` | Send a correction to the running turn: the model reads it at the next tool boundary and carries on, without the turn ending. See [Mid-turn steering](#mid-turn-steering-steer) |
| `/notifications [on\|off\|bell\|osc9]` | Turn turn-finished / approval notifications on or off for this session (default from `ui.notifications`) |
| `/model [<provider/model>] [--active]` | Pick a model: your saved ones (`models:` in config.yaml) first, then "More <provider> models…" for each provider's list (type to filter; Enter on no match uses what you typed). A model that isn't saved yet is added to `~/.dostuff/config.yaml`. A plain `/model` switch lasts only for this session; add `--active` (e.g. `/model --active`, or `/model anthropic/claude-sonnet-5-5 --active`) to make it the default that new sessions start with |
| `/config [mcp] [project]` | Edit a settings file right in the TUI: pick `config.yaml` or the MCP servers, user (`~/.dostuff`) or project (`./.dostuff`), or name one (`/config mcp project`). `Ctrl+S` checks the file, saves it and applies it without a restart (an invalid file stays open with the error); `Esc` closes, twice if there are unsaved changes. A new file starts with every option commented out. `.env` files aren't offered |
| `/status`     | Session, directory, model, permission mode, tokens, context use, MCP servers |
| `/context`    | What fills the context window: system prompt, turn context (the environment, recalled memories and hook context sent with each prompt), tools, MCP tools, project instructions, compaction summary and messages, with tokens, % and a bar |
| `/memory [project\|global]` | Edit `AGENTS.md` (this project's or `~/.dostuff/AGENTS.md`) in the same editor as `/config`; `Ctrl+S` saves and the agent follows it from the next message |
| `/doctor`     | The `dostuff doctor` checks (config, API key, MCP config, memory, hooks), with a fix for each problem |
| `/tools`      | Every tool the agent has: when it asks (in `ask` mode), whether subagents get it and what it does, then each connected MCP server's tools (reached through `call_mcp_tool`) |
| `/hooks [r]`  | List the configured hooks per event: file, matcher, timeout, `fail_closed`, and whether each runs (`user` / `trusted` / `untrusted` / `disabled` / `broken`). `/hooks r` re-reads the config files. See [Hooks](#hooks) |
| `/queue` / `/queue clear` | List the prompts waiting for the current turn: edit one (back into the input), run it next or remove it / drop them all |
| `/jobs [<id>]` / `/jobs kill <id>` | List this session's background jobs (id, owner, status, runtime, command) / show a job's last 40 lines of output and its log files / stop one (the agent is told). See [Background jobs](#background-jobs) |
| `/<skill> [args]` | Run a skill: every discovered skill is a command (listed in the `/` menu and `/help`). See [Skills](#skills) |
| `… /<skill> …` | Mid-message, makes that skill available for this message; the `/` menu offers skills as you type. See [Skills](#skills) |
| `/<command> [args]` | Run a custom command from `.dostuff/commands/` or `~/.dostuff/commands/`. See [Custom commands](#custom-commands) |
| `/rename <title>` | Name this session (shown by `dostuff sessions`) |
| `/export [<path>]` | Save this session as Markdown (default `dostuff-session-<id>.md`) |
| `/undo [force]` | Revert the agent's file changes from its last file-editing turn |
| `/compact`    | Manually trigger context compaction; `/compact auto <1-100\|off>` sets the auto-compaction threshold for this session |
| `/memories`   | List the facts saved about you (and where each came from) |
| `/episodes`   | List saved episodes (decisions, corrections, fixes) recalled here, newest first |
| `/forget <key or id>` | Delete a saved fact (by key) or episode (by the id shown in `/episodes`) |
| `Ctrl+Q`      | Graceful quit with save             |
| `Escape`      | Cancel in-flight turn (also answers "no" to an open approval prompt) |
| `/help`       | Show available in-TUI commands      |

**Visual elements:**

- **Messages** — `>` for your prompts (on a grey band), `●` for agent replies and tool calls, `⎿` for tool results; the terminal background shows through
- **Sticky prompt** — once a prompt or `!command` scrolls off the top, it stays pinned there on one row while you are in its turn, as the answer streams or when you scroll back through an old one, until the next prompt reaches the top. Click it to go back to the start of the turn; `↓ Jump to bottom` returns you to the latest message
- **Approvals** — a selector (arrows, a number, or `y` = allow once, `a` = always this session, `n` = deny with an optional note); several calls in one response can be approved together
- **Turn summary** — `✻ Worked for 1m 5s · $0.01 · done 12:59 PM` after each turn
- **Diffs** — `write_file` / `edit_file` / `multi_edit` changes shown as a coloured diff
- **Errors** — red background
- **Status bar** — `cwd • model • permission mode • context usage %` by default; `help`, `session`, `tokens` and `cost` can be added. Choose and order the items with `ui.status_bar` in `config.yaml`; when the terminal is too narrow (or zoomed in), long items are shortened and the least important dropped (help, session, cwd, tokens, cost, model, context, mode) instead of being cut off
- **Shells** — while background jobs run, a `⚙ 2 shells` item sits at the end of the status bar whatever `ui.status_bar` says. Click it (or press `↓` in an empty input) to list the session's jobs with when each started and how long it has run, and click it again to close the list; pick one (click or Enter) to see its last 8 lines of output and log path in the same small panel, refresh them, or stop it
- **Loading** — bold `Working...` with a live elapsed timer and queued-prompt count, or streamed thoughts displayed live
- **Queued prompts** — prompts that have to wait for the turn sit above the input, one row each, and run in order: skill and custom commands in both modes, plain messages with `ui.mid_turn_messages: queue`. Click the list (or `/queue`) to pick one and edit it (back into the input), run it next or remove it, or clear them all (`/queue clear`). `Esc` cancels the turn and moves the queued prompts back into the input instead of running them
- **Steering messages** — `/steer <message>` (and every plain message by default; `ui.mid_turn_messages: queue` makes them wait instead) waits for the next tool boundary instead of the end of the turn: the bar leads with `→ N for the agent · read at the next tool boundary`, `/queue` lists them first with Edit / Remove (and *Send now* turns a queued prompt into one), and `Esc` puts a pending one back in the input ahead of the queue. See [Mid-turn steering](#mid-turn-steering-steer)
- **Terminal title** — the tab / window title shows the run status: `⏳ dostuff · <folder>` while working, `✋` while an approval waits, `✓` when done; the old title comes back on exit. `ui.terminal_title: false` turns it off
- **Notifications** (off by default) — `ui.notifications: bell` (terminal bell) or `osc9` (desktop notification in iTerm2, WezTerm, Ghostty, kitty …) when a turn finishes or an approval is waiting, if it took longer than `ui.notify_after_seconds` (30) or the terminal is in the background. `/notifications` switches it per session

---

## Tools and permissions

`/tools` lists these in the TUI, plus the tools of the connected MCP servers.

| Tool | Kind | Asks? |
| ---- | ---- | ----- |
| `read_file`, `list_files`, `grep`, `glob`, `get_current_datetime`, `search_mcp_tools`, `get_mcp_tool_details` | read | never |
| `bash_output`, `kill_shell` | read | never (they only touch the agent's own background jobs) |
| `write_file`, `edit_file`, `multi_edit`, `delete_file` | write | yes (shows a diff; `multi_edit` shows one diff for all its edits) |
| `bash_command` | shell | yes, unless every part of the command is on the read-only allow-list (`ls`, `cat`, `git status`, `git diff`, …) with no redirection or substitution |
| `call_mcp_tool` | mcp | yes |
| `web_fetch` | network | yes; `a` (always) allows that domain for the session |
| `web_search` | network | yes; `a` (always) allows every search for the session. Free: DuckDuckGo by default, the `ddgs` package if installed (`pip install "dostuff[search]"`), or your own SearXNG via `search.searxng_url` |
| `agent`, `send_message`, `stop_agent` | — | no; each subagent's own calls ask, titled with the agent |
| `ask_user` | read | never (it is the agent talking to you; subagents never get it) |
| `read_settings` | read | never (MCP `env` / header values are shown as `<redacted>`) |
| `edit_config`, `set_mcp_server`, `remove_mcp_server`, `create_agent_type` | settings | **yes, in every mode but `auto`, with no "always" option** (shows the diff or the server's command) |

File tools writing under a `.dostuff` directory, and `bash_command` calls whose command or `cwd` names `.dostuff`, are
settings changes too, so `ask` and `allowlist` always prompt for them. Paths into subagent worktrees
(`.dostuff/worktrees/…`) are exempt, apart from the secrets copied there. The shell check matches the text `.dostuff`,
so a name the command builds indirectly is not caught.

Reading is not a settings change: a read-only shell command on `.dostuff` (`ls .dostuff/agents`, `cat
~/.dostuff/config.yaml`) runs like any other, plan mode included. Only the part of a pipeline that names the path has to
be a read, so `grep FAIL ~/.dostuff/jobs/<session>/<job>.out | sed … | sort -u` is fine. The same paths `read_file` can
open qualify, so in `ask` and `allowlist` these still ask: `.env*` files and `mcp_config.json` (in a `.dostuff` dir, or
the legacy one at the project root), a wildcard in the path, a command that changes directory, a `cwd` other than the
project root, and `grep` / `rg` / `Select-String` over all of a `.dostuff` dir, `.dostuff/worktrees` or one worktree,
which would reach `.env`.

`permissions.mode` in `config.yaml`:

- `ask` (default) — as in the table above.
- `allowlist` — as `ask`, plus the patterns in `permissions.allow` run without asking (a shell command prefix such as `npm test`, a tool name such as `write_file`, `call_mcp_tool:<server>__<tool>`, `web_fetch:<domain>`, or `web_search`).
- `auto` — nothing asks, so the agent never stops to wait for approval. That includes settings changes: it can add hooks and MCP servers, create agents and skills, and change most of `.dostuff`, so use it only where you would let it do all of that. It refuses two things, and the call fails at once so the agent carries on: secrets (a `.env*` file, `mcp_config.json` or `.dostuff/data` named in a command, or a search that would reach one) and its own limits (`permissions`, `hooks.project_trust`, `trusted_hooks.json`). `config.yaml` changes only through `edit_config`, which can check them. There is no classifier reviewing commands, and the shell checks match text, so this is a guard against mistakes, not a sandbox. A `PreToolUse` hook that asks still prompts, because you configured it, and so does the one-time trust check for project hooks. A red warning is shown at startup.
- `plan` — read-only: reads and read-only shell commands run; file changes, other shell commands and settings changes are denied with a "plan mode" note, and the model is told to investigate and answer with a plan. Web and MCP calls ask as in `ask`.

In the TUI, `Shift+Tab` cycles the mode for the session (ask → auto → plan → allowlist); `/permissions <mode>` picks one.

Answering `a` (always) approves that tool, or for the shell those commands, for the rest of the session.

`edit_file` and `multi_edit` grant the same thing, so a rule for either covers both: in `permissions.allow`, in
`--allowed-tools`, and for `a` (always). It stops there: neither covers `write_file` or `delete_file`.

**Changing settings by asking.** Instead of editing `~/.dostuff/config.yaml` or `mcp_config.json` by hand, ask
the agent: "hide the session id in the status bar", "set max_iterations to 50 for this project", "add the GitHub
MCP server". It reads the settings with `read_settings`, changes `config.yaml` with `edit_config` (an exact-text
edit that keeps your comments, checked to be valid YAML) and adds, updates or removes servers with
`set_mcp_server` / `remove_mcp_server`, in the user files (`~/.dostuff/`) or the project's (`./.dostuff/`).
Changes apply at once, without a restart (MCP servers connect or disconnect; tracing changes need a restart).
Because these files hold the permission rules and an MCP server is a command that runs on your machine, every
change asks, except in `auto` mode, and so do `write_file` / `edit_file` / `multi_edit` on anything in a `.dostuff` folder.
API keys stay in `.env` files, which the agent can't read or change. Subagents don't get these tools.

> [!IMPORTANT]
> File tools are sandboxed to the project directory (the directory the session runs in): paths that resolve
> outside it are rejected, except that `.dostuff` (the project's and `~/.dostuff/`) and discovered skill folders
> can be read (never written), and `.env*` files, `mcp_config.json` and `.dostuff/data` cannot be read. **The shell is not sandboxed**:
> `bash_command` can touch anything your user can, which is why non-read-only commands ask first.
> The shell is Git Bash / `bash` when found, otherwise PowerShell on Windows or `/bin/sh`.

> [!NOTE]
> **Cancelling a running tool.** ESC stops waiting for tools at once and the turn ends; results that already
> finished are kept. A running `bash_command` is killed with its children; tools that run in a worker thread (file tools) cannot
> be interrupted mid-call, so one that was still running may finish in the background; its result is recorded as
> `Cancelled by user (may still be running)`.

---

## Background jobs

Dev servers, file watchers and long test runs don't fit a shell call that has to finish within its timeout. The agent
starts them as **background jobs** instead and keeps working while they run:

| Tool | What it does |
|---|---|
| `bash_command(command, run_in_background=true, background_id="dev")` | Starts the job and returns at once with its id, pid and log files. `background_id` is optional (`bg-1`, `bg-2`, … otherwise); starting an id that is still running is refused. At most 8 running jobs per agent |
| `bash_output("dev", filter=…, wait_for=…, timeout=…)` | The job's status and the stdout / stderr it printed since the last read. `wait_for` is a regex: wait (1–110 s) until a new line matches, e.g. a server's `ready` line, instead of sleeping. `filter` keeps only matching lines |
| `kill_shell("dev")` | Kills the job and everything it started, and shows its last lines |

The approval prompt for a background start reads `Run shell command (background job "dev"):`; `bash_output` and
`kill_shell` never ask, because they only reach the calling agent's own jobs. Hooks see `run_in_background` and
`background_id` in `PreToolUse`, and can deny `kill_shell` like any tool. Plan mode allows a background job only for a
read-only command.

**When a job ends** on its own (or you stop it with `/jobs kill`), the agent is told before its next model call with a
`<background-job id="test" status="exited" code="1" runtime="42s">` message holding the last 20 lines of each stream,
so it doesn't have to poll to notice a crashed server. In the TUI the same moment shows as `⚙ job "test" exited (1)
after 42s`, with a notification when the terminal is in the background (`ui.notifications`). If the agent is idle,
the TUI starts a turn for it right away (`⚙ Resuming the agent: …`), so it can look at the failed test run or crashed
server on its own. It waits while you are typing, an approval is open or a command like `/compact` runs; if you send a
prompt first, that turn gets the message instead. A `/jobs kill` alone doesn't wake it, and after 3 such automatic
turns in a row it waits for your next message. Turn this off with `jobs: {wake_on_end: false}` in `config.yaml`; the
agent then sees the message with your next prompt.

**In the TUI:**

- `Ctrl+B` while the agent waits on a shell command (the loader says `Ctrl+B to background`) moves that command, and
  any a subagent is running in the same turn, to the background. The agent gets what it printed so far and carries on;
  the transcript shows `⎿ moved to background as "bg-3"`. `!command`s can't be backgrounded.
- `/jobs` lists the session's jobs: id, owner (`main` or `subagent: <role>`), status, runtime and command.
  `/jobs <id>` shows the last 40 lines of each stream and the log paths, `/jobs kill <id>` stops one.
- While any run, a `⚙ 2 shells` item shows at the end of the status bar; click it (or `↓` in an empty input) to pick a job
  from a list (status, start time, runtime, command) and show its output or stop it.

**Logs** go to `~/.dostuff/jobs/<session>/<job>.out` and `.err` (you can `tail -f` them; `/jobs <id>` prints the
paths). Directories older than 7 days are removed at startup.

**Jobs live until their owner ends; in the TUI that is the app, not the conversation:**

| Event | What happens to the jobs |
|---|---|
| Subagent run ends (done, error or cancelled) | its jobs are killed |
| `/compact` | kept |
| `/clear` | kept, no prompt: they move to the new session, which gets their end notices and can use `bash_output`/`kill_shell` on them |
| `/resume` another session | same as `/clear`: the running jobs follow you into the resumed session |
| Quit (`Ctrl+Q`, `/exit`) | "Quit and stop 2 background jobs?", then they are killed |
| Headless run (`dostuff -p`) ends, or `Ctrl+C` | all killed, one stderr line each (`stream-json`: `{"type": "job", …}` events) |

> [!WARNING]
> If DoStuff is killed hard (task manager, `kill -9`, power loss), its jobs keep running: they are detached from the
> terminal by design. Each session records its jobs in `~/.dostuff/jobs/<session>/jobs.json`, and the next start
> warns `1 background job from an earlier session may still be running (npm run dev, pid 4812)`. It doesn't kill
> them, since a pid may have been reused by then; check with your OS tools.

---

## Asking the user

`ask_user` lets the agent stop and ask instead of guessing: a library to use, a product decision, a name. It is
the one tool that talks to you, it never needs approval, and subagents never get it (they report back instead).

The questions arrive as a panel in place of the input:

```
 ✓ DB · ▌ Auth ·   Name
   Which auth methods should the API support?
❯   1. JWT   stateless, no extra service
    2. OAuth   a provider to configure
    3. Other (type your own)
   ────────────────────────────────────────────────────────
   Submit answers  (answer every question first)
 ↑/↓ select · Enter confirm · ←/→ change question · Esc cancel
```

The top line is the tab strip: `▌` marks the question you are on (bold and orange), `✓` marks one you have
answered, and both sit in the same slot so the tabs never shift as you answer. The submit row is dim until
every question has an answer, then it brightens and the parenthetical hint goes away.

- **Nothing submits on one key.** A pick records the answer and moves to the next question; `Submit answers` is
  the only thing that sends, and it stays inert (`answer every question first`) until every question has one.
  `←`/`Shift+Tab` goes back, so an earlier answer can be changed before submitting.
- **`2-4` options, one line each**, with the trade-off where it matters. A multi-select question toggles with
  `space` or its number, and the joined answer comes back in option order.
- **`Other`** is always the last row, numbered after the options: pick it and the input takes a typed answer (a
  pasted block is collapsed to one line). A question with no options shows as just that row, so it still appears
  in the tab strip; **Enter** (or `o`) on it takes the input, and `←`/`Tab` still move between questions — so
  landing on a free-text question never traps you in the input. A call where *every* question is free text skips
  the panel entirely and the input takes each answer in turn, with the last Enter as the submission.
- **Esc** cancels the questions and ends the turn, like Esc on an approval prompt. **Ctrl+Q** / `/exit` with a
  question open cancel it first and then quit.
- Up to 4 questions per call. The prompt tells the model when to ask: when the answer changes what it builds and
  the repo can't tell it; never to confirm a plan, to ask permission, or for anything `grep` / `read_file` answers.

The answers reach the model as one tool result, and the transcript shows the same pairs as one block — live and
after `/resume`:

```
  ⎿  Which database should I use? → Postgres
     Which auth methods should the API support? → JWT, OAuth
```

Nothing is truncated and Ctrl+O has nothing to expand, unlike an ordinary tool result: reading back what you
answered is the point. What the model actually receives is `format_answers`' output:

```text
Asked the user 2 questions; they answered:
1. Which database should I use?
   → Postgres
2. Which auth methods should the API support?
   → JWT, OAuth
```

A cancelled, refused or unavailable call renders as an ordinary tool result, with the model told what to do
(state the assumption and continue). `/export` writes the result text verbatim.

**`dostuff -p`** asks on the terminal when both stdin and stderr are a TTY: each question and its numbered options
go to stderr, and it reads one line per question (`2`, `1,3`, or free text; an empty line is `(no answer)`). There
is no submit row, because a line reader has no way back. With piped stdin (or stderr redirected to a file) the tool
is still listed but answers "No user is available in this run", and the model states an assumption and carries on.

---

## Side questions (`/btw`)

A turn can run for minutes, and halfway through you may want to know something *about* it: which file it just
edited, whether a test already exists, what you asked for. Typed into the input, that question would wait in the
queue until the turn ends; `Esc` would stop the work. `/btw <question>` answers it now, while the turn keeps running:

```
╭──────────────────────────────────────────────────────────────────────────────────╮
│ /btw — where does the retry happen?                                              │
│ The retry loop is in call_agent, wrapped by classify_model_error; loop.py        │
│ decides whether to compact or back off from its return value.                    │
│ Esc/Enter/Space close · f fork out · c copy · ↑18.4k (18.2k cached) ↓61 ·        │
│ $0.0021                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────╯
```

The panel sits above the input and the conversation stays visible underneath. It shows `Asking…` until the answer
arrives; `↑`/`↓` scroll a long one.

- **No tools run.** The model sees the conversation up to the last completed step and answers from that alone, in
  one response. If it asks for a tool or writes out a tool call instead, nothing runs and a note under the answer
  says so.
- **Nothing is written.** The question and the answer never enter the transcript, the session history, `/export` or
  the next compaction, and their cost is not added to the session: it is shown in the panel's footer instead, with
  the prompt-cache hits: the side call sends the turn's own system prompt and tools, so a provider with prompt
  caching can serve most of it from the cache. Hooks don't fire for it either.
- **No follow-up turns.** One question, one answer. Later `/btw`s in the same session see the earlier side questions
  and answers (the last 20); `/clear`, `/resume` and `/fork` start a fresh side thread.

Keys while the panel has focus: `Esc`, `Enter` or `Space` close it (Esc closes the panel, it does **not** cancel the
turn), `c` copies the answer, and `f` forks it out. One side question at a time; a second `/btw` while one is still
asking is refused. If an approval prompt or an `ask_user` question opens while the panel is up, the prompt takes the
focus and the first `Esc`.

**Fork out (`f`).** When the conversation isn't enough to answer, `f` starts a background `fork` agent with the
question, the answer to check and the same view of the conversation the side question had. It runs with the main
agent's tools in its own slot (`/agents` shows it), costs what any background agent costs, and its report comes back
into the conversation like every background agent's. `Esc` on the main turn doesn't stop it. It is carried to the new
session by `/clear` and `/resume`; `/fork` leaves it with the original session and says so.

With no turn running, `/btw` works from the stored conversation and the last turn's system prompt. `dostuff -p` has
no `/btw` (only skills and custom commands run there), so `/btw` in a `-p` prompt is ordinary text for the model.

---

## Mid-turn steering (`/steer`)

A message typed while a turn runs normally waits its turn: it joins the queue and runs as its own prompt when the
current one ends. `/steer <message>` sends a **correction** instead — the model reads it at the next tool boundary and
carries on with it, without the turn ending and without anything being interrupted:

```
> /steer actually target ./dostuff, not the repo root
```

It lands in the transcript as a user message right after the tool output it was reacting to — where the model read it
— and the next model call sees it as the newest user message. A final answer that arrives while a correction is still
waiting does not end the turn: the model reads it and redoes or extends the answer. That check runs before the `Stop`
hooks, so they see the answer that really ends the turn.

What the model reads is framed, not bare: the stored user step is your text wrapped in two fixed lines —
`The user sent a new message while you were working:` above it, `Address the message above, then continue the task
you were working on.` below. The loop has no notion of "the task", only the history, so without that instruction a
standalone question tends to be answered with a final response and the turn ends with the interrupted work left
mid-flight; Claude Code wraps mid-turn messages the same way. Everything you *see* stays raw — the transcript line,
Esc-to-restore, and what a leftover re-runs — the frame reaches only the model, plus the stored transcript that
`/resume` replays.

### Two modes

`ui.mid_turn_messages` in `config.yaml` decides what a **plain** message does while a turn runs (default `steer`):

| Submitted | `queue` | `steer` (default) |
|---|---|---|
| Plain message | waits in the queue, runs after the turn | steering: read at the next tool boundary |
| `/steer <message>` | steering | steering |
| `/<skill> args`, custom command | waits in the queue | waits in the queue |
| `/btw`, `/queue`, `/jobs`, … | run now | run now |
| `/clear`, `/resume`, `/model`, … | refused while a turn runs | refused while a turn runs |
| `!command` | refused while a turn runs | refused while a turn runs |
| `#note` (quick memory) | run now | run now |

`steer` is the default because it is Claude Code's behaviour: a correction is read now, at the next tool boundary,
instead of after the turn. `queue` stays one key away for anyone who prefers batching — three instructions typed in a
row run as three prompts while you are away, and a queued message is never read mid-task when you meant "next".
`/steer` works in both modes, and with no turn running it is
an ordinary prompt (it starts a turn), so the habit is never an error. Skills and custom commands stay in the queue in
both modes: a skill is a turn's prompt, not a correction.

### What you see

- **On submit** the input clears and the bar above it leads with the steering group, `→ 1 for the agent · read at the next tool boundary`, above the usual `N queued · click or /queue …` rows; the loader counts them separately, `· 1 steering · 2 queued`.
- **`/queue`** lists the steering messages first, each marked `→`, with *Edit* and *Remove* (once the loop has read
  one, the picker says "That message was already read."), and every queued prompt gained *Send now (at the next tool
  boundary)*, which turns it into a correction.
- **Attachments are prepared in order**: `@mentions` are expanded and pastes resolved right after you send, in the
  order you sent them, so a message holding a big file never jumps ahead of the one after it. The `Attached …` lines
  show as they finish, as for a prompt.
- **Hooks and the sandbox still apply**: `UserPromptSubmit` runs for each correction at the boundary, so a hook can
  block it — it is shown like a blocked prompt and never reaches the model — or add context, written as a
  note right before the correction (the system prompt never changes mid-turn, so the prompt cache holds); a preparation
  failure is reported instead of half-delivered.

### Latency, and `Ctrl+B`

Delivery waits for the tool call in flight; nothing is interrupted to deliver sooner, so a two-minute `bash_command`
delays a correction two minutes. While a message waits behind a running command the bar says why —
`waiting for a running command · Ctrl+B to background it` — and `Ctrl+B` moves the command (or a foreground subagent)
to the background, which returns the call to the loop at once: the message is read at the next boundary while the
command keeps running. `Esc` remains the way to stop.

### Leftovers

A correction can outlive its turn — sent after the final answer's last check, still being prepared, the turn hit the
iteration cap, the model call failed, or you pressed `Esc`. Nothing is lost:

- **Cancelled** (`Esc`) — the raw text goes back into the input **ahead of** the queued prompts: stopping the work
  includes the corrections to it.
- **Otherwise** — it runs next: `1 message arrived after the turn ended; running it next`, then it is re-prepared from
  what you typed and runs as a normal turn.

Either way the text you typed is written to the history exactly once — whether the model read it mid-turn or it ran as
the next turn.

`dostuff -p` has no `/steer` (only skills and custom commands run there), so a `/steer` line in a `-p` prompt is
ordinary text for the model.

---

## Subagents

A subagent is a separate agent with its own context, system prompt and tools. The main agent hands it a task with the
`agent` tool and gets one report back, so the searching, reading and trial and error stay out of the main
conversation. Several `agent` calls in one response run in parallel. The main agent follows up with `send_message`
and can stop one with `stop_agent`.

| Tool | What it does |
|---|---|
| `agent(description, prompt, subagent_type, model, run_in_background, isolation)` | Start an agent. `description` is 3–5 words for the UI; `prompt` is the whole task (the agent sees nothing else); `subagent_type` picks the type (default `general-purpose`); `model` overrides the type's; `isolation="worktree"` runs it in its own git worktree. The result is its report, ending with `agent_id: <id>` |
| `send_message(to, message, run_in_background)` | Continue an agent. One that has ended runs again with its whole history plus the message; one still running reads it before its next step |
| `stop_agent(agent_id)` | Stop a running agent; its shell jobs are killed and its worktree is closed out |

A subagent can never start, message or stop agents, delete files, save memories, write the todo list or change
settings, whatever its type asks for. At most `agents.max_concurrent` (4) run at once; more are queued.

### Agent types

| Type | Tools | For |
|---|---|---|
| `general-purpose` | every subagent tool | research and multi-step tasks, including changing files. The default |
| `explore` | read-only (its `bash_command` only runs read-only commands) | finding code and answering where things are; model: `agents.explore_model` |
| `plan` | read-only | investigating and returning a step-by-step plan |
| `fork` | the main agent's | a copy of the main agent that inherits the whole conversation (same system prompt, so a Claude model reuses the prompt cache). Costly: only when the task needs the conversation. Always the main model |

Your own types are Markdown files with YAML frontmatter. They are searched in this order, and the first file with a
given `name` wins: `.dostuff/agents/` and `.claude/agents/` in the project, then `~/.dostuff/agents/` and
`~/.claude/agents/`. `agents.read_claude_agents: false` skips both `.claude` folders. A file can replace a built-in
type, except `fork`. Files are re-read when they change; `/agents reload` forces it.

You can also ask the agent for one ("make a code-reviewer agent"): its `create_agent_type(name, description, prompt,
tools, model, max_iterations, scope, replace)` tool writes the file to `.dostuff/agents/` (`scope="user"`:
`~/.dostuff/agents/`) after showing it to you. Like the settings tools, it asks in every mode. The new type can be
started at once.

```markdown
---
name: code-reviewer
description: Reviews a diff for bugs and missing tests. Use after a change.
tools: Read, Grep, Glob, Bash   # omit for every subagent tool
model: fast                     # inherit (default), an alias from agents.models, or a litellm model string
max_iterations: 30              # optional; default agents.max_iterations
---
You are a code reviewer. Report findings as a list, most severe first.
```

`tools:` takes DoStuff names or Claude Code's, so Claude Code agent files work unchanged:

| Claude Code | DoStuff |
|---|---|
| `Read` | `read_file` |
| `Write` | `write_file` |
| `Edit`, `MultiEdit` | `edit_file`, `multi_edit` |
| `Glob` / `Grep` / `LS` | `glob` / `grep` / `list_files` |
| `Bash` | `bash_command` (plus `bash_output`, `kill_shell`) |
| `WebFetch` / `WebSearch` | `web_fetch` / `web_search` |
| `mcp__*` | `search_mcp_tools`, `get_mcp_tool_details`, `call_mcp_tool` |

Unknown and forbidden names are dropped with a warning naming the file; `/agents types` lists each type with its
source, resolved model, tools and any warnings, and `/agents types <name>` shows one type's whole config: its
system prompt, model, tools and max iterations (for `fork`, the task message it gets). `model:` aliases come from `agents.models`; Claude Code's `sonnet`,
`opus` and `haiku` are looked up there too and fall back to the main model (with a warning in the report) when they
aren't defined. The call's `model=` wins over the type's `model:`.

### Background agents

With `run_in_background=true` the call returns the `agent_id` at once and the main agent keeps working. When the agent
ends, its report arrives as a `<background-agent id=… status=…>` message before the main agent's next model call, and
the TUI shows `⚙ agent "<description>" (<type>) finished after 2m 10s`. If the TUI is idle it starts a turn for it
(`agents.wake_on_end`, at most 3 automatic turns in a row together with background jobs). `Ctrl+B` moves running
foreground agents to the background. Approval prompts from a background agent are titled
`[agent: <description> · <type>]` and wait while you type. The status line shows `⚙ 2 agents` (or
`⚠ 1 agent waiting` for approval); click it for `/agents`.

`dostuff -p` waits for background agents and runs a turn on their reports before it prints the answer, for at most 3
such turns; agents still running then are stopped and named on stderr. `--output-format stream-json` prints an
`{"type": "agent", …}` event when an agent starts and when it ends.

Quitting asks before it stops running agents. `/clear` and `/resume` keep background agents running in the new
session.

### Transcripts and `/agents`

Every agent's steps are saved in the session database, apart from the main conversation. `/agents` lists this
session's agents (id, status, runtime, description and type, tokens, cost); pick one to read its whole transcript,
answer its approval prompt, stop it or show its type's system prompt and config. After a restart or `/resume` the agents come back as records (a run that was
going is `interrupted`) that `send_message` can continue.

### Hooks and config

`SubagentStart` fires when a run starts and can add context to its task; `SubagentStop` fires on its final answer.
Both match the agent type, and tool hooks inside an agent see `agent_id` and `agent_type`. See [Hooks](#hooks).

```yaml
agents:
  max_iterations: 100       # model calls per run; default: the main agent's loop.max_iterations
  max_concurrent: 4         # running or waiting for approval; more are queued
  wake_on_end: true         # a finished background agent resumes the idle TUI
  read_claude_agents: true  # also load .claude/agents/
  explore_model: inherit    # model of the built-in explore type
  models:                   # aliases for model: and model=
    fast: anthropic/claude-haiku-5-5
```

---

## Hooks

Hooks are shell commands DoStuff runs at fixed points of a session: block a tool call, feed a formatter's output back
to the model, keep the turn going until the tests pass. They are configured under `hooks:` in `config.yaml` and use
Claude Code's I/O contract (stdin JSON, exit codes, JSON output), so an existing Claude Code hook script works once its
matcher and tool argument names are renamed (see [the table below](#coming-from-claude-code)).

> [!WARNING]
> Hooks run with your privileges, **unsandboxed**, and are not permission-checked: they are your config, not model
> output. Read a project's hooks before you trust them.

| Event | When | Matcher is matched against | What a hook can do |
|---|---|---|---|
| `SessionStart` | Before the first turn of a session, and after a compaction | the source: `startup`, `resume`, `clear`, `compact` | Add context for the rest of the session |
| `UserPromptSubmit` | After you send a prompt, before it is stored or sent | — | Reject the prompt, or add context sent with it (hidden from the transcript, and stays in the model's history) |
| `PreToolUse` | Before each tool call, before any approval prompt | the tool name | `allow`, `deny` or `ask`; rewrite the arguments |
| `PostToolUse` | After a tool call ran | the tool name | Append text to the result the model sees |
| `Stop` | When the agent is about to give its final answer | — | Block, so the turn continues with your reason |
| `SubagentStart` | When a subagent run starts (an `agent` call, or a `send_message` that continues one) | the agent type | Add context to the run's task. Can't block |
| `SubagentStop` | Same as `Stop`, for a subagent's final answer | the agent type | Same as `Stop` |

### Config

`matcher` is a Python regex that must match the whole name (`edit_file` does not match `edit_file_v2`); empty or `*`
matches everything. `if` (tool events only) is searched in a `bash_command`'s `command`. Inline flags such as `(?i)`
go first in a pattern. `timeout` (default 30 s, 1–300) and `fail_closed` can be set on a group or on one hook.

A tool matcher that matches `edit_file` also fires for `multi_edit`, so a guard or a formatter on `edit_file` holds for
the same change made through `multi_edit`. It does not work the other way: a `multi_edit` matcher skips `edit_file`. The
payload stays true to the call, with `tool_name` set to `multi_edit` and `tool_input` to `{path, edits}`. A script that
reads `tool_input.path` keeps working; one that reads `tool_input.old_str` gets nothing there and needs to read `edits`.
A hook that rewrites `old_str` / `new_str` makes a `multi_edit` call fail with a tool error, and nothing is written.

This is the example from the spec, and the two scripts it runs are below:

```yaml
hooks:
  project_trust: ask          # ask | allow | deny  (D4) — only read from ~/.dostuff/config.yaml
  stop_block_limit: 3         # Stop blocks honoured per turn before the loop is allowed to end (1–10)
  SessionStart:
    - matcher: "startup|resume|clear"   # regex, full match against the source; empty or "*" = all
      hooks:
        - type: command
          command: "git status --short"
  PreToolUse:
    - matcher: "bash_command"           # regex, full match against the tool name
      if: "(?i)^\\s*git\\s+push"        # optional regex, searched in tool_input.command (shell tool only)
      fail_closed: true
      timeout: 10
      hooks:
        - type: command
          name: block-main-push
          command: "python .dostuff/hooks/block_main_push.py"
  PostToolUse:
    - matcher: "write_file|edit_file"
      hooks:
        # the script reads tool_input.path from the stdin JSON and runs the formatter on it;
        # its stdout is fed back to the model (D5)
        - type: command
          name: format
          command: "python .dostuff/hooks/format_file.py"
  Stop:
    - hooks:
        - { type: command, name: tests, command: "python .dostuff/hooks/run_tests.py" }
```

- **`fail_closed: true`** (`PreToolUse` and `UserPromptSubmit` only) is for guardrails: if the hook crashes or times
  out, the call is denied (or the prompt rejected) instead of going ahead. A `fail_closed` hook with a config mistake,
  such as a bad regex, is kept and denies everything it matches; any other invalid hook is dropped with a warning.
- **`stop_block_limit`** (1–10) bounds how many times `Stop` hooks can continue one turn; after that the turn ends with
  a note.
- User hooks come first, then the project's are added after them; a project file can't remove or reorder yours.

```python
# .dostuff/hooks/block_main_push.py: PreToolUse on bash_command; the group's `if` already matched `git push`
import json
import subprocess
import sys

payload = json.load(sys.stdin)
words = payload["tool_input"].get("command", "").replace(":", " ").split()
branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True,
                        cwd=payload["cwd"]).stdout.strip()
if "main" in words or branch == "main":
    print("Don't push to main: push a feature branch and open a pull request.", file=sys.stderr)
    sys.exit(2)  # exit 2 = deny; stderr is the reason the model sees
```

```python
# .dostuff/hooks/format_file.py: PostToolUse on write_file|edit_file; stdout is appended to the tool result
import json
import subprocess
import sys

payload = json.load(sys.stdin)
path = payload["tool_input"].get("path", "")
if path.endswith(".py"):
    result = subprocess.run(["ruff", "format", path], capture_output=True, text=True, cwd=payload["cwd"])
    print((result.stdout + result.stderr).strip())
```

```python
# .dostuff/hooks/run_tests.py: Stop; exit 2 sends the failures back and the turn continues
import json
import subprocess
import sys

payload = json.load(sys.stdin)
result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x"], capture_output=True, text=True,
                        cwd=payload["cwd"])
if result.returncode not in (0, 5):  # 5: no tests collected
    print("Tests fail; fix them before finishing:\n" + result.stdout[-3000:], file=sys.stderr)
    sys.exit(2)
```

### Input and output

Each hook runs through the same shell as `bash_command`, in the project directory (a worktree subagent's own
worktree), with no terminal. stdin is one JSON object:

```json
{"hook_event_name": "PreToolUse", "session_id": "…", "turn_id": "…", "cwd": "…", "turn_type": "interactive_loop",
 "tool_name": "bash_command", "tool_input": {"command": "git push origin main"}, "tool_use_id": "…"}
```

| Event | Extra keys |
|---|---|
| `PreToolUse`, `PostToolUse` | `tool_name`, `tool_input`, `tool_use_id`; `PostToolUse` also `tool_response` (the result, already truncated) |
| `UserPromptSubmit` | `prompt` (the text you typed; images and documents appear as `[Image: name]` / `[PDF: name]`), `attachments` (`[{"name", "kind", "path"}]`, empty when there are none) |
| `SessionStart` | `source` |
| `Stop`, `SubagentStop` | `last_assistant_message`, `stop_hook_active` (true once a Stop hook already continued this turn) |
| `SubagentStart` | `agent_id`, `agent_type`, `background`, `prompt` |

Inside a subagent, `PreToolUse`, `PostToolUse` and `SubagentStop` payloads also carry `agent_id` and `agent_type`.

The environment adds `DOSTUFF_HOOK_EVENT`, `DOSTUFF_SESSION_ID`, `DOSTUFF_TURN_ID`, `DOSTUFF_PROJECT_ROOT` and
`DOSTUFF_TURN_TYPE`. Output is capped at `loop.max_tool_output_chars`.

| Exit code | Meaning |
|---|---|
| `0` | Success. Plain stdout is context (`SessionStart`, `UserPromptSubmit`, `PostToolUse`); JSON on stdout can carry a decision (below) |
| `2` | Block. stderr (or stdout) is the reason: `PreToolUse` denies the call, `PostToolUse` appends it to the result, `UserPromptSubmit` rejects the prompt, `Stop` continues the turn |
| other | The hook failed: an error line in the transcript, and the run goes on (unless `fail_closed`) |

JSON on stdout (exit 0), in either Claude Code spelling:

```json
{"decision": "block", "reason": "…"}
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow|deny|ask",
                        "permissionDecisionReason": "…", "updatedInput": {"command": "…"},
                        "additionalContext": "…"}}
```

When several hooks match, they run one after another, each on the original input: the strictest decision wins
(`deny` > `ask` > `allow`), contexts are joined in config order, and the last `updatedInput` wins. Plan mode still
applies: a hook `allow` never runs a call plan mode refused, and settings changes still ask. Rewritten arguments are
checked again, so a rewrite can't skip the prompt the new call needs. Each hook's effect shows in the transcript as a
line like `⚓ PreToolUse bash_command → deny (reason)` (a `hook` event in `--output-format stream-json`). ESC kills a
running hook.

### Project hooks and trust

Hooks in `./.dostuff/config.yaml` come with the repo, so they are code from whoever wrote it. They run only once you
trust them, set by `hooks.project_trust` in **`~/.dostuff/config.yaml`** (a project file can't set it):

- `ask` (default): before the first project hook would run, the TUI lists every project hook's command and file and
  asks. A yes is saved in `~/.dostuff/trusted_hooks.json` for that project, until the hooks block or any file in
  `.dostuff/hooks/` changes; then it asks again. A no skips them for the session.
- `allow` runs them without asking; `deny` never runs them.
- Headless (`dostuff -p`) can't ask: it honours a saved yes, runs them with `--trust-project-hooks`, and otherwise
  skips them with a warning. If a skipped hook is `fail_closed`, it refuses to start (exit code 2) instead.

Keep hook scripts in `.dostuff/hooks/`: trust covers the files there, and the model can't edit anything under
`.dostuff/` without a settings prompt. A script elsewhere in the repo can be changed after you trusted it.

`/hooks` lists every hook per event with its file, matcher, timeout, `fail_closed` and whether it runs (`user`,
`trusted`, `untrusted`, `disabled` or `broken`); `/hooks r` re-reads the config files. `dostuff doctor` reports
invalid hooks, commands not found on `PATH` and project scripts outside `.dostuff/hooks/`.

### Coming from Claude Code

The stdin envelope, exit codes, JSON output, event names and the `matcher` + `hooks: [{type: command, command}]`
nesting are the same. Tool names and some argument names differ, so rename the matcher and any `tool_input` keys:

| Claude Code | DoStuff `tool_name` | `tool_input` keys that differ |
|---|---|---|
| `Bash` | `bash_command` | none (`command`, plus `cwd`, `timeout`) |
| `Write` | `write_file` | `file_path` → `path` |
| `Edit` | `edit_file` | `file_path` → `path`, `old_string`/`new_string` → `old_str`/`new_str` |
| `MultiEdit` | `multi_edit` | `file_path` → `path`, and inside `edits`, `old_string`/`new_string` → `old_str`/`new_str` |
| `Read` | `read_file` | `file_path` → `path` |
| `Grep` / `Glob` | `grep` / `glob` | per tool |
| `AskUserQuestion` | `ask_user` | `multiSelect` → `multi_select`; the answers come back as the tool's result |
| `mcp__<server>__<tool>` | `call_mcp_tool` | server and tool are inside `tool_input` |
| `/btw` | `/btw` | not a tool: a side question fires no hook (no `UserPromptSubmit`, `PreToolUse` or `Stop`) |

Not supported yet: `http`, `prompt` and `agent` hook types, and the `Notification`, `PreCompact` and `SessionEnd`
events.

---

## Configuration

Config is layered, with later sources overriding earlier ones:

1. `~/.dostuff/config.yaml` — global defaults
2. `<cwd>/.dostuff/config.yaml` — per-project overrides
3. Environment variables — highest priority

### `config.yaml`

```yaml
mcp:
  config_path: ~/.dostuff/mcp_config.json

tracing:
  enabled: false # default OFF
  exporter: "otlp" # otlp | none
  capture_content: false # true = keep user input / tool args / tool output in spans

# Multi-model selection: active marks the selected model
models:
  - name: "openai/gpt-4o-mini"
    provider: "openai"
    active: true
  - name: "gemini/gemini-3.1-flash-lite"
    provider: "gemini"
    active: false
    # api_key_env: "MY_GEMINI_KEY"  # optional: env var holding this model's key

loop:
  max_iterations: 100
  # context_window: 128000       # override when dostuff/litellm don't know the model
  max_tool_output_chars: 30000   # longer tool results keep head + tail
  max_retries: 6                 # retries on rate limits, timeouts, 5xx (0 = none)
  retry_max_delay: 60            # cap on each wait, seconds; Retry-After is honored

memory:
  episodic_scope: project   # recall past episodes from this directory (project) or anywhere (user)

permissions:
  mode: ask                 # ask | allowlist | auto | plan
  allow: []

ui:
  mid_turn_messages: steer  # steer (default): a plain message sent while a turn runs is read by the model at
                            # the next tool boundary, inside the running turn | queue: it waits and runs after
                            # the turn. /steer sends one either way (see Mid-turn steering)

agents:                     # see Subagents
  max_iterations: 100       # model calls per run; default: loop.max_iterations (older: loop.subagent_max_iterations)
  max_concurrent: 4
  models:
    fast: anthropic/claude-haiku-5-5
```

See [`config.example.yaml`](https://github.com/kVarunkk/DoStuff/blob/package-structure/dostuff/templates/config.example.yaml) for a full template. `dostuff setup` writes it for you.

### `.env`

**Never put secrets in `config.yaml`.** `edit_config` refuses them, `/config` warns, and `dostuff doctor` flags any already there. Use `~/.dostuff/.env` and/or project-local `<cwd>/.dostuff/.env`. Both are read; project values override user values, and real environment variables override both:

```bash
# ~/.dostuff/.env (chmod 600 on Unix)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
```

### Override at runtime

```bash
MODEL=gemini/gemini-3.1-flash-lite dostuff
OTEL_ENABLED=true OTEL_EXPORTER=otlp dostuff
```

---

## Models

The model resolution maps model names (using [litellm format](https://docs.litellm.ai/docs/providers): `provider/model-name`) or derives them if the provider prefix is missing:

| Provider  | Example Model name                     | API key env         |
| --------- | -------------------------------------- | ------------------- |
| OpenAI    | `openai/gpt-4o-mini`                   | `OPENAI_API_KEY`    |
| Anthropic | `anthropic/claude-sonnet-5-5`          | `ANTHROPIC_API_KEY` |
| Gemini    | `gemini/gemini-3.1-flash-lite`         | `GEMINI_API_KEY`    |
| Groq      | `groq/llama-3.1-70b-versatile`         | `GROQ_API_KEY`      |
| Ollama    | `ollama/llama3.1`                      | _(none)_            |
| OpenCode Zen | `opencode/claude-sonnet-5-5`        | `OPENCODE_API_KEY`  |
| OpenCode Go  | `opencode-go/kimi-k3`               | `OPENCODE_API_KEY`  |

If `name` has no `/`, the provider prefix is dynamically inferred from the model name prefix or configured `provider` fields.

Any model litellm supports works; there is nothing to register in code. To add one to your list:

```bash
dostuff models add                                         # pick a provider, then a model from its list
dostuff models add anthropic/claude-sonnet-5-5            # add it (pick it with /model)
dostuff models add groq/llama-3.3-70b-versatile --active  # add it and make it the default
```

or `/model` inside the TUI. Model lists come from litellm's catalog (chat models, without dated snapshots that have an alias), from OpenRouter's and OpenCode's public model lists, and for Ollama from the models you've pulled (the server must be running). A name that isn't listed still works. Then put the provider's key in `~/.dostuff/.env`. A project's `./.dostuff/config.yaml` with its own `models:` list replaces the global list in that directory.

OpenCode models (`opencode/<id>` for Zen, `opencode-go/<id>` for a Go subscription) use the ids from the [OpenCode docs](https://opencode.ai/docs/zen/). litellm has no OpenCode provider, so DoStuff sends each model to the gateway endpoint for its family: Claude models (and a few Qwen / MiniMax models) to the Anthropic Messages endpoint, GPT / Grok / Muse models to the Responses endpoint, Gemini models to the Gemini endpoint, and all other models to chat completions. One console key in `OPENCODE_API_KEY` works for Zen and Go. OpenCode only allows its free models from its own app, so use paid models or Go.

The gateways don't publish a model's context window in `/v1/models`, so `helpers/models/opencode.py` carries one per model, taken from [models.dev](https://models.dev)'s `opencode` and `opencode-go` entries (the same id can differ per gateway: `opencode/minimax-m3` is 512k, `opencode-go/minimax-m3` is 1M). `/context`, the status bar and auto-compaction size themselves from it; set `loop.context_window` to override.

---

## MCP servers

MCP servers extend the agent with new tools. Configure them in `~/.dostuff/mcp_config.json`.

**Discovery:** if `mcp.config_path` is set in `config.yaml`, only that file is used. Otherwise `~/.dostuff/mcp_config.json` is loaded and `<cwd>/.dostuff/mcp_config.json` is merged over it (a project server replaces a global one with the same name). `<cwd>/mcp_config.json` (legacy) is used only when neither exists.

Manage servers with `dostuff mcp list` / `add` / `remove` instead of editing the JSON by hand.

### Example config

```json
{
  "mcpServers": {
    "tavily": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@tavily/mcp-server"]
    },
    "github": {
      "transport": "stdio",
      "command": "uvx",
      "args": ["mcp-server-github"],
      "env": { "GITHUB_TOKEN": "${MY_WORK_GITHUB_KEY}" }
    },
    "remote": {
      "transport": "http",
      "url": "https://mcp.example.com/sse",
      "headers": { "Authorization": "Bearer ${SEARCH_API_KEY}" }
    }
  }
}
```

**Keys go in `.dostuff/.env`, not in this file.** `${NAME}` in `headers` and `env` is replaced from `.env` (or the real environment) when the server connects:

- **HTTP server:** you pick the name. It only links the two files, so `SEARCH_API_KEY=...` in `.env` matches `${SEARCH_API_KEY}` above.
- **Local (stdio) server:** the server picks the name, so copy it from the server's docs. It inherits `.env`, so a key under that exact name needs no `env` entry (`tavily` above reads `TAVILY_API_KEY`). Use `env` to pass a key you saved under another name (`github` above).

Only the exact form `${NAME}` is replaced, and only in `headers` and `env`. If a variable is missing or empty the server doesn't connect, and the error names it; after adding it, restart dostuff. `dostuff doctor` reports missing variables and plain keys left in the file, and `/config` and `dostuff mcp add` warn about them. On the command line use single quotes, so your shell doesn't expand the name itself: `dostuff mcp add search --url https://mcp.example.com -H 'Authorization: Bearer ${SEARCH_API_KEY}'`.

A server's `url` receives whatever its headers name, so read a project's `.dostuff/mcp_config.json` before you run dostuff in a repository you don't trust, just as you would for a `command` it runs.

| Field       | Required for | Notes                          |
| ----------- | ------------ | ------------------------------ |
| `transport` | optional     | `stdio` (default) or `http`    |
| `command`   | stdio        | Executable (`npx`, `uvx`, ...) |
| `args`      | stdio        | Argument list                  |
| `env`       | stdio        | Env vars for the child process, added to what it inherits; `${NAME}` is read from `.env` |
| `url`       | http         | Remote MCP server URL          |
| `headers`   | http         | HTTP headers; `${NAME}` is read from `.env` |
| `connect_timeout` | optional | Seconds before giving up on this server (default 150, enough for a browser OAuth login) |

MCP servers connect concurrently in the background, so the TUI never blanks during boot. Each server is announced in the chat area upon successful or failed connection.

MCP tools are named `<server>__<tool>` (e.g. `github__create_issue`), so two servers can expose a tool with the same name. `search_mcp_tools` only returns tools of servers that are connected right now.

**OAuth (HTTP servers):** a browser login opens when the server answers 401. The flow uses PKCE, a random `state` and an RFC 8707 `resource` indicator; the local callback on port 8085 times out after 120 s. Tokens are stored in the OS keyring (memory only if no keyring backend is available), never in a plain file. Expired access tokens are refreshed with the refresh token, before a request or after a 401, without opening the browser; you sign in again only when the refresh token itself is rejected.

To force a re-registration, delete:

```bash
rm ~/.dostuff/data/mcp_client_registrations.json
```

---

## Project instructions

Standing instructions for the agent go in `AGENTS.md` (or `CLAUDE.md`). Every file that applies is
added to the system prompt, most general first, so the closest one wins where they disagree:

1. `~/.dostuff/AGENTS.md` — your instructions for every project
2. one per folder from the top of the path down to the current directory (e.g. the repo root, then a sub-package)

In a folder with both, `AGENTS.md` is used. They're read once at startup, and again after you change
one from the TUI: `# <note>` adds a line to this project's or the global file, and `/memory` opens one in an editor.

---

## Skills

Skills are directories containing a `SKILL.md` (frontmatter with `name` and `description` + instructions body).

**Discovered from (in order):**

1. `<cwd>/.dostuff/skills/`
2. `<cwd>/.agents/skills/`
3. `~/.dostuff/skills/`
4. `~/.agents/skills/`
5. `<cwd>/skills/`

**Conflict resolution:** Skills are deduplicated by name: if two share a name, the first one found (in the order above) wins and the other is dropped, from the system prompt and from the slash commands alike.

**How a skill reaches the model:** the system prompt lists each skill's `name`, `description` and `dir` (its absolute folder), and only those. When the model decides to use a skill it reads the `SKILL.md` in that dir with `read_file`, so a body costs nothing until it is needed. Frontmatter is parsed as YAML, so quoted and folded (`>`, `|`) descriptions arrive as plain text; a file with invalid YAML still loads, with its fields read line by line.

File tools can read a discovered skill's folder wherever it lives, including the files next to `SKILL.md` (`scripts/`, `reference/`), but never write there. `.env*` files stay unreadable inside a skill folder too, and of `~/.dostuff/` only discovered skills under `~/.dostuff/skills/` can be read.

**Example structure:**

```text
~/.agents/skills/
└── my-skill/
    └── SKILL.md        # required
```

**`SKILL.md`:**

```markdown
---
name: my-skill
description: One-line description of what this skill does
---

# My Skill

Detailed instructions for the agent go here.
```

**Running a skill yourself:** each skill is also a slash command, `/<name> [arguments]` (e.g. `/my-skill src/app.py`), in the TUI and with `dostuff -p "/my-skill ..."`. The turn's message is the skill's `SKILL.md` body plus your arguments (which can use `@path` mentions), so the model follows it straight away instead of deciding whether to read it. The command shows in the `/` menu marked `(skill)` and in `/help`. Built-in commands win a name clash, the first skill found (in the order above) wins between skills, and names with spaces can't be run this way; the model can still use those skills on its own.

**Manual-only skills:** add `disable-model-invocation: true` to the frontmatter and the model is not told about the skill. It is left out of the system prompt and every subagent's prompt, so it only runs when you ask for it, with `/<name>` or a mention (below). It stays in the `/` menu and `/help`, and the `Skills loaded` line at startup marks it `(manual only)`. Only `false`, `no`, `off`, `0` or an empty value leave a skill visible to the model; anything else, a typo included, hides it.

> [!IMPORTANT]
> `disable-model-invocation` is **not a security boundary**. It stops the model from discovering and choosing the
> skill; it does not block anything. The skill's folder stays readable by the file tools, a project skill is an
> ordinary project file, and the shell can read whatever your user can. To guarantee a dangerous command never runs,
> put a `PreToolUse` [hook](#hooks) on that command.

**Mentioning a skill:** write `/<name>` inside a message, as in `go ahead and /deploy`, to make that skill available for that message. This is how the model gets to use a manual-only skill, and it works for any skill. The `SKILL.md` is not expanded: the model gets a note after your message with the skill's name, description and folder, and judges from your wording, so `don't run /deploy yet` works. You keep seeing the message as you typed it, plus a line, `Skill available for this message: deploy`; if that line is missing, the name wasn't recognised. It works in prompts, queued prompts, steering messages and `dostuff -p`. The permission lasts for that message, and subagents don't inherit it.

A mention follows a space or a new line, can be any case, and may end in one `.`, `,`, `!`, `?`, `:` or `;`. So `` `/deploy` ``, `(/deploy)` and `/deploy/x` are not mentions, and neither is a path like `/tmp/notes.md`. Custom commands can't be mentioned. Only what you type counts: a `/deploy` inside an `@file` or a paste collapsed to `[Pasted text #1 …]` grants nothing. A message that starts with a skill or custom command is that command's run, so `/review then /deploy` grants nothing for `deploy`.

While you type, `/` after a space opens the menu with skills only. Tab and Enter there replace just that word: your sentence stays and nothing is sent until the next Enter.

Run `dostuff init` to bootstrap a local `.dostuff/skills/` folder. Restart the agent after adding new skills.

### Custom commands

A Markdown file in `.dostuff/commands/` (this project) or `~/.dostuff/commands/` (every project) is a slash
command named after the file: `.dostuff/commands/review.md` is `/review`. Running `/review src/app.py` sends the
file's text as your message, with `$ARGUMENTS` replaced by `src/app.py` (with no `$ARGUMENTS` in the file, the
arguments are added at the end). An optional frontmatter `description:` is shown in the `/` menu, otherwise the
first line is. Works in the TUI and with `dostuff -p "/review src/app.py"`.

```markdown
---
description: Review a file for bugs
---
Review $ARGUMENTS for bugs and missing tests. List problems by severity.
```

Built-in commands win a name clash, then project commands, then user commands, then skills. Restart dostuff
after adding a command.

---

## Sessions and memory

- **DB:** `~/.dostuff/data/sessions.db` (SQLite)
- **Resume:** `dostuff resume` (latest in the current dir), `dostuff resume <id>` or `dostuff --session <id>`
- **List:** `dostuff sessions` (current dir) / `dostuff sessions --all`
- **Memory stores:** `~/.dostuff/data/chroma/` (needs `pip install "dostuff[memory]"`; without it, saved memories are kept on disk but not used)
  - Semantic (key/value facts)
  - Episodic (events with summaries and timestamps)
- **Saved when:** the agent decides — `save_memory` stores a key/value fact (an overlapping fact is updated, not duplicated) and `save_episode` stores a decision, correction or lesson (one that rewords an episode already saved in the directory is skipped). Each save is shown in the TUI; nothing is extracted at exit. Subagents can't save memories.
- **Two step tables** — `session_steps` (full audit, shown on resume) and `working_steps` (what the model sees; compaction flags old rows instead of rewriting them). Each step is one INSERT. DBs from 0.1.6 and earlier are migrated on first open (`PRAGMA user_version`).
- **Memory recall** — episodic memories come from your past sessions in the same directory (`memory.episodic_scope: user` for all directories). The agent is told never to save facts that only appeared in tool output; only facts and episodes related to the prompt are recalled (an unrelated prompt like "hi" gets none). List them with `/memories` and `/episodes`, delete one with `/forget`.
- **Index repair** — if Chroma's vector index gets out of sync with its stored rows (chroma-core/chroma#7032), recall and saving fall back to a slower scan instead of failing, and the TUI warns at startup. `dostuff memory repair` rebuilds the index; rows it can't rebuild are left out but kept in the backup, and a failed rebuild leaves the original as it was.
- **Token persistence** — `prompt_tokens` / `completion_tokens` / `total_tokens` / `last_input_tokens` saved to DB; resume restores percent from DB meta

Long-term memory survives `pip install . --upgrade` because it lives in the user's home directory, not in the package.

---

## Tracing (opt-in)

Tracing is **OFF by default** to avoid spurious endpoint connection errors.

```yaml
# ~/.dostuff/config.yaml
tracing:
  enabled: true
  exporter: "otlp"
  endpoint: "localhost:4317"
```

Or via env:

```bash
OTEL_ENABLED=true OTEL_EXPORTER=otlp dostuff
```

---

## Troubleshooting

> [!WARNING]
> TUI says **"No API key found for model …"**
>
> Run `dostuff setup`, or pick the model in the `models` list of `~/.dostuff/config.yaml` (or set `MODEL=provider/model`) and add the key it names to `~/.dostuff/.env`. `dostuff doctor` shows what is missing.

> [!WARNING]
> **Connection closed** (MCP tool)
>
> The MCP server disconnected. Check the command and logs. Force a reconnect:
>
> ```bash
> rm ~/.dostuff/data/mcp_client_registrations.json
> dostuff
> ```

> [!WARNING]
> **MarkupError** when a tool returns rich or complex structures
>
> Fixed in the current version by disabling unneeded Rich parsing on tool logs. Update:
>
> ```bash
> pip install --upgrade dostuff
> ```

> [!WARNING]
> **Tracing errors / endpoint retrying**
>
> Tracing is OFF by default. If you enabled it and have no collector running, set:
>
> ```bash
> OTEL_ENABLED=false dostuff
> ```

> [!WARNING]
> **Session not resuming**
>
> Use `dostuff sessions` (or `/resume` in the TUI) to find the session ID. Each session also stores the working directory it was started from and switches to it before launching the TUI.

> [!WARNING]
> **`pip install .` fails**
>
> Requires Python 3.11+. Upgrade pip first:
>
> ```bash
> pip install --upgrade pip
> ```

---

## Project structure

```text
dostuff/
├── agent/                  # agent loop, tool dispatch, token tracking
│   ├── call_agent.py       # litellm agent call wrapper with retry/retry-delay logic
│   ├── loop.py             # main execution loop, auto-compaction and tool coordination
│   ├── permissions.py      # permission modes, shell allow-list, session approvals
│   └── run_tool.py         # authorize (permission check) + dispatch of one tool call
├── helpers/                # sub-modules for agent, mcp, memory, skills, tools, ui
│   ├── agent/              # state, exit handlers, system prompts, token limit utilities
│   ├── mcp/                # config loading, oauth
│   ├── memory/             # memory resolver, session export, shell steps
│   ├── skills/             # skill discovery logic
│   ├── tools/              # schemas and path resolution
│   └── ui/                 # event emission shims
├── lib/                    # core logic
│   ├── checkpoints.py      # per-turn file checkpoints for /undo
│   ├── exceptions.py       # validation, confirmation, and workflow exceptions
│   ├── mcp/                # mcp transport client & local registrations
│   ├── memory/             # semantic, episodic, and session stores
│   ├── model.py            # model and provider resolution
│   ├── tracing.py          # opentelemetry tracing bootstrap
│   └── worktree.py         # git worktree create / exit / cleanup
├── memory/                 # DB management
│   └── __init__.py         # DB initiation exports
├── tools/                  # defined tools
│   ├── bash/               # terminal command execution tools
│   ├── files/              # read, write, edit, delete, list
│   ├── search/             # grep and glob (pure Python, default excludes)
│   ├── mcp/                # mcp tool search, detail, and execution wrappers
│   ├── memory/             # save_memory / save_episode
│   ├── settings.py         # read_settings, edit_config, set/remove_mcp_server
│   ├── web_fetch.py        # read a URL as text
│   ├── web_search.py       # free web search (SearXNG / ddgs / DuckDuckGo)
│   ├── todo_write.py       # visible task list
│   ├── agent.py            # the agent tool (subagents: dostuff/agent/subagents/)
│   ├── create_agent_type.py # write a new agent type file
│   └── send_message.py, stop_agent.py # continue or stop an agent
├── cli.py                  # CLI typer entrypoints and doctor checkup utilities
├── cli_tui.py              # Textual TUI Application and adapter translates state
├── config.py               # layered config loader supporting project/global yaml and multi-model configuration
├── setup_wizard.py         # first-run / `dostuff setup` onboarding
├── templates/              # config.example.yaml and env.example shipped with the package
└── skills.py               # skills loader re-exports
```

---

## Project metadata

- **Author:** [varun](mailto:varunkumawatleap2@gmail.com)
- **Repository:** https://github.com/kVarunkk/DoStuff
- **Issues:** https://github.com/kVarunkk/DoStuff/issues
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **PyPI:** https://pypi.org/project/dostuff/
