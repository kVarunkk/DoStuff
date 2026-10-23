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

- **Textual TUI** with per-role colored messages, status bar, and a loading/timer spinner
- **Multi-provider LLMs** via [litellm](https://docs.litellm.ai/docs/providers) — OpenAI, Anthropic, Gemini, Groq, Mistral, Ollama, OpenRouter
- **Persistent sessions** in SQLite; resume with `dostuff resume` or `dostuff --session <id>`
- **Long-term memory** (semantic + episodic) backed by ChromaDB
- **MCP server support** — stdio (npx, uvx) and HTTP transports
- **Coding tools** — `read_file` (line numbers, offset/limit), `edit_file` (exact replacement), `write_file`, `grep`, `glob`, `list_files`, `delete_file`, `bash_command`
- **Permission layer** — mutating tool calls ask for approval (`y` once / `a` always this session / `n` deny); read-only shell commands run without asking; modes `ask` / `allowlist` / `auto`
- **File sandbox** — file tools stay inside the project directory and never read `.env*` files or `~/.dostuff/`
- **Skills loader** — drop a `SKILL.md` in the right folder, restart, done
- **Subagents** — `delegate_to_subagent` runs a bounded task with a restricted tool set; its approval prompts and ESC go through the same TUI
- **Per-tool timeout** (120s) to prevent hung servers; long tool output is truncated (head + tail) before it reaches the model
- **Retries with backoff** for rate limits, timeouts and 5xx errors (6 by default, exponential with jitter, honors `Retry-After`; `loop.max_retries` / `loop.retry_max_delay`); auth / bad-request errors fail fast
- **Token tracking** — per-turn and cumulative totals displayed live
- **Long-term memory tools** — the agent saves facts (`save_memory`) and notable moments (`save_episode`) as they come up; both are recalled in later sessions
- **Graceful exit** — `Ctrl+Q` or `/exit` saves the session, cleans up, and stops background workers
- **Clean Cancellation** — `Escape` can cancel in-flight agent turns cleanly without corrupting the session
- **2-state session split** — `session_steps` (full audit) / `working_history` (LLM payload); DB persists both plus `compaction_notes` + `last_input_tokens`
- **Auto & Manual Compaction** — Manual `/compact` runs compaction and saves it. Auto-compaction triggers automatically during the agent loop if context usage (`last_input_tokens`) exceeds 50% of the model's token limit. It keeps at least 15% or 20k tokens of the most recent steps in the working history, summarizing the rest into `compaction_notes` which are appended to system instructions.
- **Tracing (opt-in)** — OTLP tracing support built into `lib/tracing.py`; prompt and tool content are redacted from spans by default

---

## Install

Install from PyPI:

```bash
pip install dostuff
```

This installs the `dostuff` command on your `PATH`. Done.

### Other install methods

For an isolated install with [pipx](https://pypa.github.io/pipx/):

```bash
pipx install dostuff
```

For development (editable mode, from a clone):

```bash
git clone https://github.com/<you>/dostuff.git
cd dostuff
pip install -e .           # core
pip install -e ".[full]"   # + onnxruntime + grpcio
pip install -e ".[keyring]" # persist MCP OAuth tokens in the OS keyring
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
| `dostuff --session <id>` | Resume a prior session (asks before starting a new one if the ID is unknown) |
| `dostuff init [--force]` | Create `.dostuff/` config/skills in current dir (won't overwrite without `--force`) |
| `dostuff setup`          | Pick a model, save its API key, write `~/.dostuff/config.yaml` (runs automatically on first launch) |
| `dostuff config`         | Show resolved config paths and values           |
| `dostuff doctor`         | Health check: config, API key, MCP, data dir; prints a fix for each problem |
| `dostuff resume [<id>]`  | Resume the latest session started in the current directory (or the given one; a unique ID prefix is enough) |
| `dostuff sessions [--all] [-n N]` | List sessions in the current directory with their title or first prompt (`--all`: every directory) |
| `dostuff sessions rename <id> <title>` | Name a session |
| `dostuff sessions rm <id>... [-y]` | Delete sessions and their `/undo` checkpoints |
| `dostuff sessions prune [--older-than 30d] [--dry-run] [-y]` | Delete empty sessions, and sessions not used for the given time (`45m`, `12h`, `30d`, `2w`) |
| `dostuff export [<id>] [-o file\|-]` | Save a session (default: latest here) as Markdown: prompts, answers, tool calls and results |
| `dostuff -p "<prompt>"`  | Headless: run one prompt, print the answer to stdout, exit. Progress goes to stderr. Piped stdin is appended to the prompt (`-p -` reads the prompt from stdin). `--session <id>` continues a session, `--json` prints `{result, session_id, usage, ...}`, `-q` hides progress, `--permission-mode ask\|allowlist\|auto`. Without a terminal, calls that need approval are denied |
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
| `Ctrl+J`      | Insert a newline (multi-line input) |
| `/exit`       | Save the session and quit           |
| `/clear`      | Start a new session; the old one stays resumable |
| `/resume [<id>]` | Pick another session from this directory (or switch to the given one) |
| `/model [<provider/model>] [--active]` | Pick a model: your saved ones (`models:` in config.yaml) first, then "More <provider> models…" for each provider's list (type to filter; Enter on no match uses what you typed). A model that isn't saved yet is added to `~/.dostuff/config.yaml`. The switch lasts for this session; `--active` also makes it the default |
| `/status`     | Session, directory, model, permission mode, tokens, context use, MCP servers |
| `/rename <title>` | Name this session (shown by `dostuff sessions`) |
| `/export [<path>]` | Save this session as Markdown (default `dostuff-session-<id>.md`) |
| `/undo [force]` | Revert the agent's file changes from its last file-editing turn |
| `/compact`    | Manually trigger context compaction |
| `/memories`   | List the facts saved about you (and where each came from) |
| `Ctrl+Q`      | Graceful quit with save             |
| `Escape`      | Cancel in-flight turn (also answers "no" to an open approval prompt) |
| `/help`       | Show available in-TUI commands      |

**Visual elements:**

- **User messages** — blue background
- **Agent messages** — neutral
- **Tool calls and results** — teal/green background
- **Confirmations** — yellow background (respond `y` = allow once, `a` = always this session, anything else = deny)
- **Diffs** — `write_file` / `edit_file` changes shown as a coloured diff
- **Errors** — red background
- **Status bar** — `cwd • session-id • active model • prompt/completion token rates • context usage % • processing timer`
- **Loading** — `Working...` or streamed thoughts displayed live

---

## Tools and permissions

| Tool | Kind | Asks? |
| ---- | ---- | ----- |
| `read_file`, `list_files`, `grep`, `glob`, `get_current_datetime`, `search_mcp_tools`, `get_mcp_tool_details` | read | never |
| `write_file`, `edit_file`, `delete_file` | write | yes (shows a diff) |
| `bash_command` | shell | yes, unless every part of the command is on the read-only allow-list (`ls`, `cat`, `git status`, `git diff`, …) with no redirection or substitution |
| `call_mcp_tool` | mcp | yes |
| `delegate_to_subagent` | — | no; the subagent's own calls ask |

`permissions.mode` in `config.yaml`:

- `ask` (default) — as in the table above.
- `allowlist` — as `ask`, plus the patterns in `permissions.allow` run without asking (a shell command prefix such as `npm test`, a tool name such as `write_file`, or `call_mcp_tool:<server>__<tool>`).
- `auto` — nothing asks. A red warning is shown at startup.

Answering `a` (always) approves that tool, or for the shell those commands, for the rest of the session.

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
  # context_window: 128000       # override when litellm doesn't know the model
  max_tool_output_chars: 30000   # longer tool results keep head + tail
  max_retries: 6                 # retries on rate limits, timeouts, 5xx (0 = none)
  retry_max_delay: 60            # cap on each wait, seconds; Retry-After is honored

memory:
  episodic_scope: project   # recall past episodes from this directory (project) or anywhere (user)

permissions:
  mode: ask                 # ask | allowlist | auto
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
| Anthropic | `anthropic/claude-3-5-sonnet-20240620` | `ANTHROPIC_API_KEY` |
| Gemini    | `gemini/gemini-3.1-flash-lite`         | `GEMINI_API_KEY`    |
| Groq      | `groq/llama-3.1-70b-versatile`         | `GROQ_API_KEY`      |
| Ollama    | `ollama/llama3.1`                      | _(none)_            |

If `name` has no `/`, the provider prefix is dynamically inferred from the model name prefix or configured `provider` fields.

Any model litellm supports works; there is nothing to register in code. To add one to your list:

```bash
dostuff models add                                         # pick a provider, then a model from its list
dostuff models add anthropic/claude-sonnet-5-5            # add it (pick it with /model)
dostuff models add groq/llama-3.3-70b-versatile --active  # add it and make it the default
```

or `/model` inside the TUI. Model lists come from litellm's catalog (chat models, without dated snapshots that have an alias), from OpenRouter's public model list, and for Ollama from the models you've pulled (the server must be running). A name that isn't listed still works. Then put the provider's key in `~/.dostuff/.env`. A project's `./.dostuff/config.yaml` with its own `models:` list replaces the global list in that directory.

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

**OAuth (HTTP servers):** a browser login opens when the server answers 401. The flow uses PKCE, a random `state` and an RFC 8707 `resource` indicator; the local callback on port 8085 times out after 120 s. Tokens are kept in memory, or in the OS keyring if you install `dostuff[keyring]`, never in a plain file. A tool call that fails with 401 re-authenticates once and retries.

To force a re-registration, delete:

```bash
rm ~/.dostuff/data/mcp_client_registrations.json
```

---

## Skills

Skills are directories containing a `SKILL.md` (frontmatter with `name` and `description` + instructions body).

**Discovered from (in order):**

1. `<cwd>/.dostuff/skills/`
2. `<cwd>/.agents/skills/`
3. `~/.dostuff/skills/`
4. `~/.agents/skills/`
5. `<cwd>/skills/`

**Conflict resolution:** No deduplication. If two skills share the same name (from different directories), both are loaded and passed to the model — the agent may see the same skill twice with different `location` paths. To avoid confusion, keep names unique across the search paths.

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

Run `dostuff init` to bootstrap a local `.dostuff/skills/` folder. Restart the agent after adding new skills.

---

## Sessions and memory

- **DB:** `~/.dostuff/data/sessions.db` (SQLite)
- **Resume:** `dostuff resume` (latest in the current dir), `dostuff resume <id>` or `dostuff --session <id>`
- **List:** `dostuff sessions` (current dir) / `dostuff sessions --all`
- **Memory stores:** `~/.dostuff/data/chroma/`
  - Semantic (key/value facts)
  - Episodic (events with summaries and timestamps)
- **Saved when:** the agent decides — `save_memory` stores a key/value fact (an overlapping fact is updated, not duplicated) and `save_episode` stores a decision, correction or lesson. Each save is shown in the TUI; nothing is extracted at exit. Subagents can't save memories.
- **Two step tables** — `session_steps` (full audit, shown on resume) and `working_steps` (what the model sees; compaction flags old rows instead of rewriting them). Each step is one INSERT. DBs from 0.1.6 and earlier are migrated on first open (`PRAGMA user_version`).
- **Memory recall** — episodic memories come from your past sessions in the same directory (`memory.episodic_scope: user` for all directories). The agent is told never to save facts that only appeared in tool output; list saved facts with `/memories`.
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
│   ├── memory/             # transcript formatting and memory extractors
│   ├── skills/             # skill discovery logic
│   ├── tools/              # schemas and path resolution
│   └── ui/                 # event emission shims
├── lib/                    # core logic
│   ├── exceptions.py       # validation, confirmation, and workflow exceptions
│   ├── mcp/                # mcp transport client & local registrations
│   ├── memory/             # semantic, episodic, and session stores
│   ├── model.py            # model and provider resolution
│   └── tracing.py          # opentelemetry tracing bootstrap
├── memory/                 # DB management
│   └── __init__.py         # DB initiation exports
├── tools/                  # defined tools
│   ├── bash/               # terminal command execution tools
│   ├── files/              # read, write, edit, delete, list
│   ├── search/             # grep and glob (pure Python, default excludes)
│   └── mcp/                # mcp tool search, detail, and execution wrappers
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
