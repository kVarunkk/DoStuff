## Implementation Tasks

### Phase 1: Package Foundation
- [x] Create `pyproject.toml` with `[project.scripts] dostuff = "dostuff.cli:main"`, dependencies (`typer`/`click`, `litellm`, `chroma`, `sqlite`, etc.), package name. Document `pip install`, `pipx`, `uv` in docs.
- [x] Create `src/dostuff/` directory structure (`__init__.py`, `cli.py`, `config.py`, `core/`, `memory/`).
- [x] Move `app.py` logic into `src/dostuff/core/agent.py` (refactor, not full rewrite).
- [x] Write `src/dostuff/cli.py` using typer: `main`, `init`, `config`, `doctor` subcommands.
- [x] Confirm `pip install .` creates `dostuff` entry point; test `pipx install .` and document both.
- [x] Verify `dostuff` from random directory (not repo root) — should work globally.

### Phase 1b: User ID & Session Selection (new requirements)
- [x] Implement first-run UUID generation (`~/.dostuff/user_id`); reuse by default.
- [x] Implement session list query from SQLite (session_id, working_dir, timestamp) with selectable output.

### Phase 2: Global Config & Directory
- [x] Implement `src/dostuff/config.py` with `Config` class: reads `~/.dostuff/config.yaml`, resolves `.dostuff/` override, applies CLI args.
- [x] Create default `config.yaml` template: model settings, identity file path (`identity.md`), data mode (`global` default), skills directories (`~/.dostuff/skills/`).
- [x] Implement `dostuff init`: creates `CWD/.dostuff/config.yaml` with `data.mode: project`, default identity copy.

### Phase 3: Data Architecture Redesign
- [x] Modify memory initialization to use configurable `data_dir` (from config) instead of hardcoded `chroma_data/`/`agent_sessions.db`.
- [x] Create `~/.dostuff/data/` structure on first run (global mode); create `CWD/.dostuff/data/` when project mode active.
- [x] Ensure SQLite DB (`sessions.db`) and Chroma collections (`semantic/`, `episodic/`) use `data_dir` base path.
- [x] Confirm `user_id` + `session_id` filtering remains intact across both modes.

### Phase 3b: Working Directory in Session (new requirement)
- [x] Add `working_dir` column to SQLite session DB; save CWD at session create/update.
- [x] Implement resume logic: if selected session's working_dir differs from current CWD, change CWD automatically.

### Phase 4: Skills & Packaging
- [x] Add user skills directory (`~/.dostuff/skills/`) to registry loader.
- [x] Add global agent skills directory (`~/.agents/skills/`) to registry loader.
- [x] Add project skills directory (`.dostuff/skills/`) when `data.mode` is `project`.
- [x] Remove old `requirements.txt` dependency tracking; rely solely on `pyproject.toml`.

### Phase 5: Validation & Exit
- [x] Test `dostuff` from random directory (not repo root) — should work globally.
- [ ] Test `dostuff --session S --user U` resolves session/user correctly.
- [x] Test `dostuff init` then run — confirms `.dostuff/` created with isolated data.
- [x] Confirm no backward-compat code remains; clean break documented.

# Refinements applied after 25/25
- [x] Bash tool with HITL guardrail
- [x] Resume subcommand removed; --session handles resume
- [x] .env override + CWD restore on resume
- [x] Session meta preserved (no DB overwrite on resume)
