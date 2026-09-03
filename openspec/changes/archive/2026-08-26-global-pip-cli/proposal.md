## Why

The existing agent harness runs only via `python app.py` from a project directory. It cannot be installed globally (`pip install`), has no CLI entry point (`dostuff`), stores all data (SQLite sessions, Chroma semantic/episodic memory) tied to CWD, and requires users to clone/manage the repo manually. Converting to a Pi-style global pip-installable CLI makes it available in any working directory, with centralized config (`~/.dostuff/`) and user/session-scoped memory architecture.

## What Changes

- **New package structure**: `pyproject.toml` with `dostuff` console script entry point; `src/dostuff/` package layout.
- **New CLI entry**: `dostuff` command replaces `python app.py`; parses args via typer/click; detects working directory for file operations.
- **Global config directory**: `~/.dostuff/` created on first run; holds `config.yaml` (model, identity, data mode), `mcp.json`, `identity.md`, and user skills (`skills/`).
- **Data architecture redesign**: All three existing stores (semantic Chroma/user-scoped, episodic Chroma/user+session-scoped, session SQLite/session-scoped) moved to configurable paths (`~/.dostuff/data/` global default; optional `CWD/.dostuff/data/` project mode via `dostuff init`). Keeps `user_id` + `session_id` architecture.
- **Built-in skills**: Bundled in package (`src/dostuff/skills/`) + user extensions in `~/.dostuff/skills/`.
- **Package installation**: `pip install .`, `pipx install .` (recommended for CLI isolation), or `uv tool install .` supported; document all three in install notes.
- **No backward compatibility**: Clean break; no legacy config/file detection.
- **YAML config**: `config.yaml` preferred over JSON/Markdown for unified settings.

**BREAKING**: Previous `python app.py <session_id> <user_id>` invocation replaced by `dostuff --session <id> --user <id>`; previous CWD-local SQLite/Chroma paths become configurable (defaults to global).

## Capabilities

### New Capabilities
- `cli-entry`: Global `dostuff` command, argument parsing, working-directory-aware execution.
- `global-config`: `~/.dostuff/` hierarchy with `config.yaml`, `mcp.json`, `identity.md`, user skills.
- `data-storage`: Configurable data paths for three memory stores (semantic/episodic/session) supporting global and project modes.
- `package-install`: Pip-installable package with `pyproject.toml`, `src/` layout, built-in skills bundle.

### Modified Capabilities
- None (new capability set; existing harness is being replaced, not modified in place).

## Impact

- `requirements.txt` → `pyproject.toml` (dependencies become package metadata).
- `app.py` → `src/dostuff/cli.py` + `core/agent.py` (refactored).
- `DOSTUFF.md` → `~/.dostuff/identity.md` (global default) / `CWD/.dostuff/identity.md` (project override).
- `mcp_config.json` → `~/.dostuff/mcp.json` / `CWD/.dostuff/mcp.json`.
- `skills/` → split into package-bundled (`src/dostuff/skills/`) + user (`~/.dostuff/skills/`).
- SQLite (`agent_sessions.db`) and Chroma (`chroma_data/`) paths become `config.yaml` driven (`data.global_dir`, `data.mode`).
