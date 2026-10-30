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
- **Coding tools** — `read_file` (line numbers, offset/limit), `edit_file` (exact replacement), `write_file`, `grep`, `glob`, `list_files`, `delete_file`, `bash_command`, `web_fetch` (read a URL as text), `web_search` (free, no API key)
- **Permission layer** — mutating tool calls ask for approval (allow once / always this session / deny with a note); read-only shell commands run without asking; modes `ask` / `allowlist` / `auto`; `/undo` reverts the agent's file changes
- **Hooks** — shell commands on `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop` and `SubagentStop` that can block a tool call or prompt, feed a formatter's output back to the model, or keep the turn going; Claude Code's hook I/O contract, and project hooks run only once you trust them. See [Hooks](#hooks)
- **File sandbox** — file tools stay inside the project directory and never read `.env*` files or `~/.dostuff/`
- **Skills loader** — drop a `SKILL.md` in the right folder, restart, done; run one with `/<skill-name> [args]`
- **Subagents** — `delegate_to_subagent` runs a bounded task with a restricted tool set; its approval prompts and ESC go through the same TUI
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
| `pip install "dostuff[full]"` | `memory` + `search` |
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
| `@path`       | Attach a file (numbered lines) or a folder listing to the message. Typing `@` lists matching project paths: `Tab` / `Enter` completes, a folder opens for the next part. `@"a b.txt"` for spaces. Also works with `dostuff -p`. `.env` files and binaries are never attached |
| `Tab`         | While typing a slash command, accept the dim ghost text after the cursor: the rest of the command name, then its flag (`/model ` → `--active`, `/fork ` → `--worktree`). `Right` accepts it too |
| Paste        | A paste over 10 lines (`ui.paste_collapse_lines`) or 2,000 chars shows as `[Pasted text #1 +243 lines]` and is expanded when you send it. Over 200 lines (`loop.paste_file_lines`) it's saved to `.dostuff/pastes/` and the model reads it from there, so it doesn't fill the context; files are removed after `loop.paste_retention_days` (7) |
| `Up` / `Down` | On the first / last line of the input: previous / next prompt typed in this directory (saved in `~/.dostuff/data/prompt_history.jsonl`) |
| `Ctrl+R`      | Search the prompts typed in this directory (type to filter, `Enter` puts the prompt in the input) |
| `Esc` `Esc`   | With an empty input: take back your last prompt. It and everything after it (answer, tool calls) leave the conversation and the prompt goes back in the input to edit and resend. Files that turn changed stay as they are; `/undo` restores them |
| `Ctrl+O`      | Expand / collapse every truncated tool and shell result (click one to toggle just that one) |
| `Shift+Tab`   | Cycle the permission mode for this session: ask → auto → plan → allowlist (shown in the status bar) |
| `Ctrl+Backspace` / `Ctrl+W` / `Alt+Backspace` | Delete the previous word; `Ctrl+Delete` / `Alt+Delete` the next one |
| `/exit`       | Save the session and quit           |
| `/clear`      | Start a new session; the old one stays resumable |
| `/resume [<id>]` | Pick another session from this directory (or switch to the given one) |
| `/fork [title] [--worktree]` | Copy this session into a new one and switch to it; the original is unchanged |
| `/worktree [name]` / `/worktree exit` | Move the session into a new git worktree on branch `dostuff/<name>` / leave it (removed if unchanged, otherwise keep or remove) |
| `/permissions [mode]` | Show or change the permission mode for this session (`ask`, `allowlist`, `auto`, `plan`) |
| `/copy`       | Copy the model's last answer to the clipboard |
| `/notifications [on\|off\|bell\|osc9]` | Turn turn-finished / approval notifications on or off for this session (default from `ui.notifications`) |
| `/model [<provider/model>] [--active]` | Pick a model: your saved ones (`models:` in config.yaml) first, then "More <provider> models…" for each provider's list (type to filter; Enter on no match uses what you typed). A model that isn't saved yet is added to `~/.dostuff/config.yaml`. A plain `/model` switch lasts only for this session; add `--active` (e.g. `/model --active`, or `/model anthropic/claude-sonnet-5-5 --active`) to make it the default that new sessions start with |
| `/config [mcp] [project]` | Edit a settings file right in the TUI: pick `config.yaml` or the MCP servers, user (`~/.dostuff`) or project (`./.dostuff`), or name one (`/config mcp project`). `Ctrl+S` checks the file, saves it and applies it without a restart (an invalid file stays open with the error); `Esc` closes, twice if there are unsaved changes. A new file starts with every option commented out. `.env` files aren't offered |
| `/status`     | Session, directory, model, permission mode, tokens, context use, MCP servers |
| `/context`    | What fills the context window: system prompt, tools, MCP tools, project instructions, compaction summary and messages, with tokens, % and a bar |
| `/memory [project\|global]` | Edit `AGENTS.md` (this project's or `~/.dostuff/AGENTS.md`) in the same editor as `/config`; `Ctrl+S` saves and the agent follows it from the next message |
| `/doctor`     | The `dostuff doctor` checks (config, API key, MCP config, memory, hooks), with a fix for each problem |
| `/hooks [r]`  | List the configured hooks per event: file, matcher, timeout, `fail_closed`, and whether each runs (`user` / `trusted` / `untrusted` / `disabled` / `broken`). `/hooks r` re-reads the config files. See [Hooks](#hooks) |
| `/<skill> [args]` | Run a skill: every discovered skill is a command (listed in the `/` menu and `/help`). See [Skills](#skills) |
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
- **Approvals** — a selector (arrows, a number, or `y` = allow once, `a` = always this session, `n` = deny with an optional note); several calls in one response can be approved together
- **Turn summary** — `✻ Worked for 1m 5s · $0.01 · done 12:59 PM` after each turn
- **Diffs** — `write_file` / `edit_file` changes shown as a coloured diff
- **Errors** — red background
- **Status bar** — `cwd • model • permission mode • context usage %` by default; `help`, `session`, `tokens` and `cost` can be added. Choose and order the items with `ui.status_bar` in `config.yaml`; when the terminal is too narrow (or zoomed in), long items are shortened and the least important dropped (help, session, cwd, tokens, cost, model, context, mode) instead of being cut off
- **Loading** — bold `Working...` with a live elapsed timer and queued-prompt count, or streamed thoughts displayed live
- **Terminal title** — the tab / window title shows the run status: `⏳ dostuff · <folder>` while working, `✋` while an approval waits, `✓` when done; the old title comes back on exit. `ui.terminal_title: false` turns it off
- **Notifications** (off by default) — `ui.notifications: bell` (terminal bell) or `osc9` (desktop notification in iTerm2, WezTerm, Ghostty, kitty …) when a turn finishes or an approval is waiting, if it took longer than `ui.notify_after_seconds` (30) or the terminal is in the background. `/notifications` switches it per session

---

## Tools and permissions

| Tool | Kind | Asks? |
| ---- | ---- | ----- |
| `read_file`, `list_files`, `grep`, `glob`, `get_current_datetime`, `search_mcp_tools`, `get_mcp_tool_details` | read | never |
| `write_file`, `edit_file`, `delete_file` | write | yes (shows a diff) |
| `bash_command` | shell | yes, unless every part of the command is on the read-only allow-list (`ls`, `cat`, `git status`, `git diff`, …) with no redirection or substitution |
| `call_mcp_tool` | mcp | yes |
| `web_fetch` | network | yes; `a` (always) allows that domain for the session |
| `web_search` | network | yes; `a` (always) allows every search for the session. Free: DuckDuckGo by default, the `ddgs` package if installed (`pip install "dostuff[search]"`), or your own SearXNG via `search.searxng_url` |
| `delegate_to_subagent` | — | no; the subagent's own calls ask |
| `read_settings` | read | never (MCP `env` / header values are shown as `<redacted>`) |
| `edit_config`, `set_mcp_server`, `remove_mcp_server` | settings | **always, in every mode, with no "always" option** (shows the diff or the server's command) |

File tools writing under a `.dostuff` directory, and `bash_command` calls whose command or `cwd` names `.dostuff`, are
settings changes too, so they always ask. That way auto mode can't quietly add a hook to `~/.dostuff/config.yaml`. Paths
into subagent worktrees (`.dostuff/worktrees/…`) are exempt. The shell check matches the text `.dostuff`, so a name the
command builds indirectly is not caught.

`permissions.mode` in `config.yaml`:

- `ask` (default) — as in the table above.
- `allowlist` — as `ask`, plus the patterns in `permissions.allow` run without asking (a shell command prefix such as `npm test`, a tool name such as `write_file`, `call_mcp_tool:<server>__<tool>`, `web_fetch:<domain>`, or `web_search`).
- `auto` — nothing asks. A red warning is shown at startup.
- `plan` — read-only: reads and read-only shell commands run; file changes, other shell commands and settings changes are denied with a "plan mode" note, and the model is told to investigate and answer with a plan. Web and MCP calls ask as in `ask`.

In the TUI, `Shift+Tab` cycles the mode for the session (ask → auto → plan → allowlist); `/permissions <mode>` picks one.

Answering `a` (always) approves that tool, or for the shell those commands, for the rest of the session.

**Changing settings by asking.** Instead of editing `~/.dostuff/config.yaml` or `mcp_config.json` by hand, ask
the agent: "hide the session id in the status bar", "set max_iterations to 50 for this project", "add the GitHub
MCP server". It reads the settings with `read_settings`, changes `config.yaml` with `edit_config` (an exact-text
edit that keeps your comments, checked to be valid YAML) and adds, updates or removes servers with
`set_mcp_server` / `remove_mcp_server`, in the user files (`~/.dostuff/`) or the project's (`./.dostuff/`).
Changes apply at once, without a restart (MCP servers connect or disconnect; tracing changes need a restart).
Because these files hold the permission rules and an MCP server is a command that runs on your machine, every
change asks, even in `auto` mode, and so do `write_file` / `edit_file` on anything in a `.dostuff` folder.
API keys stay in `.env` files, which the agent can't read or change. Subagents don't get these tools.

> [!IMPORTANT]
> File tools are sandboxed to the project directory (the directory the session runs in): paths that resolve
> outside it are rejected, and `.env*` files and `~/.dostuff/` cannot be read. **The shell is not sandboxed**:
> `bash_command` can touch anything your user can, which is why non-read-only commands ask first.
> The shell is Git Bash / `bash` when found, otherwise PowerShell on Windows or `/bin/sh`.

> [!NOTE]
> **Cancelling a running tool.** ESC stops waiting for tools at once and the turn ends; results that already
> finished are kept. Tools that run in a worker thread (file tools, `bash_command`) cannot be interrupted
> mid-call, so one that was still running may finish in the background; its result is recorded as
> `Cancelled by user (may still be running)`.

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
| `UserPromptSubmit` | After you send a prompt, before it is stored or sent | — | Reject the prompt, or add context for this turn |
| `PreToolUse` | Before each tool call, before any approval prompt | the tool name | `allow`, `deny` or `ask`; rewrite the arguments |
| `PostToolUse` | After a tool call ran | the tool name | Append text to the result the model sees |
| `Stop` | When the agent is about to give its final answer | — | Block, so the turn continues with your reason |
| `SubagentStop` | Same, for a `delegate_to_subagent` run | — | Same |

### Config

`matcher` is a Python regex that must match the whole name (`edit_file` does not match `edit_file_v2`); empty or `*`
matches everything. `if` (tool events only) is searched in a `bash_command`'s `command`. Inline flags such as `(?i)`
go first in a pattern. `timeout` (default 30 s, 1–300) and `fail_closed` can be set on a group or on one hook. This is
the example from the spec, and the two scripts it runs are below:

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
| `UserPromptSubmit` | `prompt` |
| `SessionStart` | `source` |
| `Stop`, `SubagentStop` | `last_assistant_message`, `stop_hook_active` (true once a Stop hook already continued this turn) |

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
| `Read` | `read_file` | `file_path` → `path` |
| `Grep` / `Glob` | `grep` / `glob` | per tool |
| `mcp__<server>__<tool>` | `call_mcp_tool` | server and tool are inside `tool_input` |

Not supported yet: `http`, `prompt` and `agent` hook types, and the `Notification`, `PreCompact`, `SessionEnd` and
`SubagentStart` events.

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
  subagent_max_iterations: 25
  # context_window: 128000       # override when dostuff/litellm don't know the model
  max_tool_output_chars: 30000   # longer tool results keep head + tail
  max_retries: 6                 # retries on rate limits, timeouts, 5xx (0 = none)
  retry_max_delay: 60            # cap on each wait, seconds; Retry-After is honored

memory:
  episodic_scope: project   # recall past episodes from this directory (project) or anywhere (user)

permissions:
  mode: ask                 # ask | allowlist | auto | plan
  allow: []
```

See [`config.example.yaml`](https://github.com/kVarunkk/DoStuff/blob/package-structure/dostuff/templates/config.example.yaml) for a full template. `dostuff setup` writes it for you.

### `.env`

**Never put secrets in `config.yaml`.** Use `~/.dostuff/.env` and/or project-local `<cwd>/.dostuff/.env`. Both are read; project values override user values, and real environment variables override both:

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
      "args": ["-y", "@tavily/mcp-server"],
      "env": { "TAVILY_API_KEY": "tvly-..." }
    },
    "github": {
      "transport": "stdio",
      "command": "uvx",
      "args": ["mcp-server-github"],
      "env": { "GITHUB_TOKEN": "ghp_..." }
    },
    "remote": {
      "transport": "http",
      "url": "https://mcp.example.com/sse",
      "headers": { "Authorization": "Bearer xxx" }
    }
  }
}
```

| Field       | Required for | Notes                          |
| ----------- | ------------ | ------------------------------ |
| `transport` | optional     | `stdio` (default) or `http`    |
| `command`   | stdio        | Executable (`npx`, `uvx`, ...) |
| `args`      | stdio        | Argument list                  |
| `env`       | stdio        | Env vars for the child process |
| `url`       | http         | Remote MCP server URL          |
| `headers`   | http         | HTTP headers                   |
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

**Conflict resolution:** No deduplication in the system prompt. If two skills share the same name (from different directories), both are loaded and passed to the model — the agent may see the same skill twice with different `location` paths. To avoid confusion, keep names unique across the search paths.

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
│   └── delegate_to_subagent.py # bounded subagents (optionally in a worktree)
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
