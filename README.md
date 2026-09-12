<div align="center">

# DoStuff

**A pip-installable AI agent harness with a Textual TUI, persistent sessions, and MCP tool support.**

[![PyPI](https://img.shields.io/badge/PyPI-published-3776ab?style=flat-square&logo=pypi)](https://pypi.org/project/dostuff/)

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![litellm](https://img.shields.io/badge/litellm-any--provider-blueviolet?style=flat-square)](https://docs.litellm.ai/docs/providers)
[![MCP](https://img.shields.io/badge/MCP-compatible-success?style=flat-square)](https://modelcontextprotocol.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![textual](https://img.shields.io/badge/UI-textual-ff6b6b?style=flat-square)](https://textual.textualize.io)

[Install](#install) • [Quick start](#quick-start) • [Configuration](#configuration) • [TUI controls](#tui-controls) • [MCP servers](#mcp-servers) • [Skills](#skills) • [Troubleshooting](#troubleshooting)

</div>

DoStuff is a terminal AI agent that runs anywhere `pip` runs. It gives you a full interactive session with persistent memory, multi-provider LLM support via [litellm](https://docs.litellm.ai), and pluggable [MCP](https://modelcontextprotocol.io) tool servers. State lives in `~/.dostuff/`, so any project you launch `dostuff` from shares the same memory and sessions.

> [!TIP]
> If you have used `claude-code` or `aider`, this will feel familiar. DoStuff is opinionated, minimal, and easy to extend.

---

## Features

- **Textual TUI** with per-role colored messages, status bar, and a loading/timer spinner
- **Multi-provider LLMs** via [litellm](https://docs.litellm.ai/docs/providers) — OpenAI, Anthropic, Gemini, Groq, Mistral, Ollama, OpenRouter
- **Persistent sessions** in SQLite; resume with `dostuff --session <id>`
- **Long-term memory** (semantic + episodic) backed by ChromaDB
- **MCP server support** — stdio (npx, uvx) and HTTP transports
- **Skills loader** — drop a `SKILL.md` in the right folder, restart, done
- **Per-tool timeout** (120s) to prevent hung servers
- **Token tracking** — per-turn and cumulative totals displayed live
- **Graceful exit** — `Ctrl+Q` or `/exit` saves, cleans up, and stops background workers
- **Clean Cancellation** — `Escape` can cancel in-flight agent turns cleanly without corrupting the session
- **2-state session split** — `session_steps` (full audit) / `working_history` (LLM payload); DB persists both plus `compaction_notes` + `last_input_tokens`
- **Auto & Manual Compaction** — Manual `/compact` runs compaction and saves it. Auto-compaction triggers automatically during the agent loop if context usage (`last_input_tokens`) exceeds 50% of the model's token limit. It keeps at least 15% or 20k tokens of the most recent steps in the working history, summarizing the rest into `compaction_notes` which are appended to system instructions.
- **Tracing (opt-in)** — OTLP tracing support built into `lib/tracing.py`

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
```

Verify:

```bash
dostuff --help
```

---

## Quick start

```bash
# 1. Set up config and secrets
mkdir -p ~/.dostuff
cp config.example.yaml ~/.dostuff/config.yaml
cp .env.example ~/.dostuff/.env
nano ~/.dostuff/.env   # add keys (e.g., OPENAI_API_KEY=sk-...)

# 2. cd to your project root
cd your_project

# 3. Launch the TUI
dostuff
```

In the TUI:

```text
> hi there
[agent replies]

> /exit
[session saved, goodbye]
```

> [!NOTE]
> No config? No `.env`? The TUI shows a clear message and lets you quit without crashing. See [Troubleshooting](#troubleshooting).

---

## CLI reference

| Command                  | Description                                     |
| ------------------------ | ----------------------------------------------- |
| `dostuff`                | Launch the interactive TUI                      |
| `dostuff --session <id>` | Resume a prior session                          |
| `dostuff init`           | Create `.dostuff/` config/skills in current dir |
| `dostuff config`         | Show resolved config paths and values           |
| `dostuff doctor`         | Health check (config, data dir, user ID)        |
| `dostuff session-list`   | List all past sessions with their working dirs  |

---

## TUI controls

| Key / command | Action                              |
| ------------- | ----------------------------------- |
| `Enter`       | Submit message                      |
| `Ctrl+J`      | Insert a newline (multi-line input) |
| `/exit`       | Save memories and quit              |
| `/clear`      | Clear chat and session history      |
| `/compact`    | Manually trigger context compaction |
| `Ctrl+Q`      | Graceful quit with save             |
| `Escape`      | Cancel in-flight turn               |
| `/help`       | Show available in-TUI commands      |

**Visual elements:**

- **User messages** — blue background
- **Agent messages** — neutral
- **Tool calls and results** — teal/green background
- **Confirmations** — yellow background (respond with `y` or `n`)
- **Errors** — red background
- **Status bar** — `cwd • session-id • active model • prompt/completion token rates • context usage % • processing timer`
- **Loading** — `Working...` or streamed thoughts displayed live

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
  enabled: true # default OFF
  exporter: "otlp" # otlp | none

# Multi-model selection: active marks the selected model
models:
  - name: "openai/gpt-4o-mini"
    provider: "openai"
    active: true
  - name: "gemini/gemini-3.1-flash-lite"
    provider: "gemini"
    active: false
```

See [`config.example.yaml`](https://github.com/kVarunkk/DoStuff/blob/package-structure/config.example.yaml) for a full template.

### `.env`

**Never put secrets in `config.yaml`.** Use `~/.dostuff/.env` or project-local `<cwd>/.dostuff/.env`:

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

---

## MCP servers

MCP servers extend the agent with new tools. Configure them in `~/.dostuff/mcp_config.json`.

**Discovery order:** `mcp.config_path` → `~/.dostuff/mcp_config.json` → `<cwd>/.dostuff/mcp_config.json` → `<cwd>/mcp_config.json`.

**Conflict resolution:** No merge — the first found config wins. If `~/.dostuff/mcp_config.json` exists, the project-level `.dostuff/mcp_config.json` is **silently ignored**. To use a custom path, set `mcp.config_path` in `config.yaml` — that takes highest priority.

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

MCP connections start in a background worker, so the TUI never blanks during boot. Each server is announced in the chat area upon successful or failed connection.

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
- **Resume:** `dostuff --session <id>`
- **List:** `dostuff session-list`
- **Memory stores:** `~/.dostuff/data/chroma/`
  - Semantic (key/value facts)
  - Episodic (events with summaries and timestamps)
- **Saved on:** `/exit` or `Ctrl+Q`
- **Compacted working_history** — preserved in `working_history`; `session_steps` kept as full audit; `compaction_notes` merged with existing notes on each compact
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
> TUI says **"No LLM configured"**
>
> Set `model.name` in `~/.dostuff/config.yaml` or define the `models` list, and add the API key to `~/.dostuff/.env`. See [Quick start](#quick-start).

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
> Use `dostuff session-list` to find the session ID. Each session also stores the working directory it was started from and switches to it before launching the TUI.

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
│   └── run_tool.py         # executes tool calls safely with optional timeouts
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
│   ├── files/              # file write, delete, list, and read utilities
│   └── mcp/                # mcp tool search, detail, and execution wrappers
├── cli.py                  # CLI typer entrypoints and doctor checkup utilities
├── cli_tui.py              # Textual TUI Application and adapter translates state
├── config.py               # layered config loader supporting project/global yaml and multi-model configuration
└── skills.py               # skills loader re-exports
```

---

## Project metadata

- **Author:** [varun](mailto:varunkumawatleap2@gmail.com)
- **Repository:** https://github.com/kVarunkk/DoStuff
- **Issues:** https://github.com/kVarunkk/DoStuff/issues
