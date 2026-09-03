<div align="center">

# DoStuff

**A pip-installable AI agent harness with a Textual TUI, persistent sessions, and MCP tool support.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![litellm](https://img.shields.io/badge/litellm-any--provider-blueviolet?style=flat-square)](https://docs.litellm.ai/docs/providers)
[![MCP](https://img.shields.io/badge/MCP-compatible-success?style=flat-square)](https://modelcontextprotocol.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![textual](https://img.shields.io/badge/UI-textual-ff6b6b?style=flat-square)](https://textual.textualize.io)

[Install](#install) • [Quick start](#quick-start) • [Configuration](#configuration) • [TUI controls](#tui-controls) • [MCP servers](#mcp-servers) • [Skills](#skills) • [Troubleshooting](#troubleshooting)

</div>

DoStuff is a terminal AI agent that runs anywhere `pip` runs. It gives you a full interactive session with persistent memory, multi-provider LLM support via [litellm](https://docs.litellm.ai), and pluggable [MCP](https://modelcontextprotocol.io) tool servers. State lives in `~/.dostuff/`, so any project you launch `dostuff` from shares the same memory and sessions.

![TUI demo showing agent loop with tool call and response](./docs/demo.png)

> [!TIP]
> If you have used `claude-code` or `aider`, this will feel familiar. DoStuff is opinionated, minimal, and easy to extend.

---

## Features

- **Textual TUI** with per-role colored messages, status bar, and a loading spinner
- **Multi-provider LLMs** via [litellm](https://docs.litellm.ai/docs/providers) — OpenAI, Anthropic, Gemini, Groq, Mistral, Ollama, OpenRouter
- **Persistent sessions** in SQLite; resume with `dostuff --session <id>`
- **Long-term memory** (semantic + episodic) backed by ChromaDB
- **MCP server support** — stdio (npx, uvx) and HTTP transports
- **Skills loader** — drop a `SKILL.md` in the right folder, restart, done
- **Per-tool timeout** (120s) to prevent hung servers
- **Token tracking** — per-turn and cumulative totals
- **Graceful exit** — `Ctrl+Q` or `/exit` saves, cleans up, and stops background workers

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
nano ~/.dostuff/.env   # add OPENAI_API_KEY=sk-...

# 2. Edit ~/.dostuff/config.yaml and set model.name
#    (see config.example.yaml)

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

| Command                  | Description                                    |
| ------------------------ | ---------------------------------------------- |
| `dostuff`                | Launch the interactive TUI                     |
| `dostuff --session <id>` | Resume a prior session                         |
| `dostuff --user <id>`    | Override the user ID                           |
| `dostuff init`           | Create `.dostuff/` in the current directory    |
| `dostuff config`         | Show resolved config paths and values          |
| `dostuff doctor`         | Health check (config, data dir, user ID)       |
| `dostuff session-list`   | List all past sessions with their working dirs |

---

## TUI controls

| Key / command | Action                              |
| ------------- | ----------------------------------- |
| `Enter`       | Submit message                      |
| `Ctrl+J`      | Insert a newline (multi-line input) |
| `/exit`       | Save memories and quit              |
| `Ctrl+Q`      | Graceful quit with save             |
| `/help`       | Show available in-TUI commands      |

**Visual elements:**

- **User messages** — blue background
- **Agent messages** — neutral
- **Tool calls and results** — teal background, truncated at 600 chars
- **Confirmations** — yellow background (respond with `y` or `n`)
- **Errors** — red background
- **Status bar** — `cwd • session-id • new/resumed • ↑total ↓total`
- **Loading** — `⏳ agent is thinking...` while waiting for the model

---

## Configuration

Config is layered, with later sources overriding earlier ones:

1. `~/.dostuff/config.yaml` — global defaults
2. `<cwd>/.dostuff/config.yaml` — per-project overrides
3. `~/.dostuff/.env` / `<cwd>/.dostuff/.env` — secrets
4. Environment variables — highest priority

### `config.yaml`

```yaml
# ~/.dostuff/config.yaml
data:
  global_dir: ~/.dostuff/data

mcp:
  config_path: ~/.dostuff/mcp_config.json

tracing:
  enabled: false # default OFF (no 4317 noise)
  exporter: "otlp" # otlp | console | none

model:
  name: "openai/gpt-4o-mini" # litellm format: provider/model
  api_key_env: "OPENAI_API_KEY" # name of the env var holding the key
```

See [`config.example.yaml`](./config.example.yaml) for a full template.

### `.env`

**Never put secrets in `config.yaml`.** Use `~/.dostuff/.env`:

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
OTEL_ENABLED=true OTEL_EXPORTER=console dostuff
```

---

## Models

The `name` field uses [litellm format](https://docs.litellm.ai/docs/providers): `provider/model-name`.

| Provider  | Model name                             | API key env         |
| --------- | -------------------------------------- | ------------------- |
| OpenAI    | `openai/gpt-4o-mini`                   | `OPENAI_API_KEY`    |
| Anthropic | `anthropic/claude-3-5-sonnet-20240620` | `ANTHROPIC_API_KEY` |
| Gemini    | `gemini/gemini-3.1-flash-lite`         | `GEMINI_API_KEY`    |
| Groq      | `groq/llama-3.1-70b-versatile`         | `GROQ_API_KEY`      |
| Ollama    | `ollama/llama3.1`                      | _(none)_            |

If `name` has no `/`, the `provider` field is auto-prepended.

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

MCP connections start in a background worker, so the TUI never blanks during boot. Each server is announced with a `✓` or `✗` in the message area.

To force a re-registration, delete:

```bash
rm ~/.dostuff/data/mcp_client_registrations.json
```

---

## Skills

Skills are markdown documents that teach the agent new workflows. Each is a directory with a `SKILL.md` (frontmatter + body).

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
    ├── SKILL.md        # required
    └── helpers/        # optional
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

Long-term memory survives `pip install . --upgrade` because it lives in the user's home directory, not in the package.

---

## Tracing (opt-in)

Tracing is **OFF by default** to avoid spurious `localhost:4317` errors.

```yaml
# ~/.dostuff/config.yaml
tracing:
  enabled: true
  exporter: "console" # or "otlp"
  endpoint: "localhost:4317"
```

Or via env:

```bash
OTEL_ENABLED=true OTEL_EXPORTER=console dostuff
```

---

## Troubleshooting

> [!WARNING]
> TUI says **"No LLM configured"**
>
> Set `model.name` in `~/.dostuff/config.yaml` and add the API key to `~/.dostuff/.env`. See [Quick start](#quick-start).

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
> **MarkupError** when a tool returns a long URL
>
> Fixed in the current version. Update:
>
> ```bash
> pip install --upgrade dostuff
> ```

> [!WARNING]
> **Tracing errors / `localhost:4317` retrying**
>
> Tracing is OFF by default. If you enabled it and have no collector, set:
>
> ```bash
> OTEL_ENABLED=false dostuff
> ```

> [!WARNING]
> **Session not resuming**
>
> Use `dostuff session-list` to find the session ID. Each session also stores the working directory it was started from.

> [!WARNING]
> **`pip install .` fails**
>
> Requires Python 3.10+. Upgrade pip first:
>
> ```bash
> pip install --upgrade pip
> ```

---

## Project structure

```text
dostuff/
├── cli_tui.py              # Textual App + adapter
├── cli.py                  # typer CLI entrypoint
├── config.py               # layered config (env > project > global > default)
├── agent/
│   ├── loop.py             # agent loop, tool dispatch, token tracking
│   ├── call_agent.py       # LLM call wrapper
│   └── run_tool.py         # tool execution
├── lib/
│   ├── tracing.py          # opt-in OpenTelemetry tracing
│   ├── model.py            # model resolution
│   ├── memory/             # SQLite + ChromaDB stores
│   └── mcp/                # MCP client and registries
├── helpers/
│   ├── skills/             # skill discovery
│   ├── agent/              # identity, constants, exit handlers
│   ├── mcp/                # config loading
│   └── ui/                 # emit shim
└── tools/                  # tool definitions
    ├── bash/
    ├── files/
    └── mcp/
```

---

## Project metadata

- **Author:** [varun](mailto:varunkumawatleap2@gmail.com)
- **Repository:** https://github.com/kVarunkk/DoStuff
- **Issues:** https://github.com/kVarunkk/DoStuff/issues
