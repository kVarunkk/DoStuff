## Architecture Overview

```
User runs: dostuff --session S --user U
    │
    ▼
┌─────────────────────────────────────────────┐
│  CLI (src/dostuff/cli.py)                   │
│  • typer commands (main, init, config)     │
│  • argument parsing (--session, --user)     │
│  • working dir detection                    │
└────┬────────────────────────────┬───────────┘
     │                             │
     ▼                             ▼
Config Resolver              Agent Core
(src/dostuff/config.py)     (src/dostuff/core/)
• global: ~/.dostuff/        • loads agent identity
• project: .dostuff/         • calls LLM via litellm
• hierarchy resolution       • uses memory stores
     │                             │
     ▼                             ▼
Data Paths                  Memory Stores
(data_dir from config)      (src/dostuff/memory/)
• global_dir default        • semantic Chroma
• project override         • episodic Chroma
                             • session SQLite
```

## Key Decisions

**Package layout**: `src/dostuff/` (not flat) for clean imports; `pyproject.toml` replaces `requirements.txt`.

**CLI framework**: `typer` (already in requirements, modern over `click` directly). Entry script: `dostuff = "dostuff.cli:main"`.

**Config resolution**: `Config` class reads `config.yaml`; checks `data.mode`; resolves directory paths; applies CLI overrides last.

**Data path logic**: `get_data_dir(config)` returns `.dostuff/data` when mode is `project` or `.dostuff/` detected; else `global_dir` (`~/.dostuff/data`). SQLite and Chroma use this base path with subdirectories (`sessions.db`, `chroma/semantic/`, `chroma/episodic/`).

**Skill loading**: User skills from `~/.dostuff/skills/`; global agent skills from `~/.agents/skills/`; project skills from `.dostuff/skills/`. Merged registry.

**No package-bundled skills**: The pip package does NOT include skills/ directory. Skills are discovered from user/global/project sources only. First run initializes `~/.dostuff/skills/` empty (or creates if user adds later).

**No backward compat**: No legacy detection code; clean paths. Migration is manual (re-run `dostuff init` in projects, copy old identity to `~/.dostuff/` if desired).

**Installation methods**: `pip install .`, `pipx install .` (recommended for isolated global CLI), or `uv tool install .` all supported; docs recommend `pipx`/`uv`.

## Data Flow

1. `dostuff` starts → config module loads/resolves settings.
2. Agent core initialized with `user_id`, `session_id`, `identity_path`, `data_dir`.
3. Memory stores connect to SQLite (session) and Chroma collections (semantic/episodic) using `data_dir`.
4. CLI loop continues; file tools use CWD for file operations regardless of `data_dir`.
