# Feature: Global pip-installable CLI (`dostuff`)

**Session:** `package-structure-sesh`
**Change:** `global-pip-cli` (`openspec/changes/global-pip-cli/`)
**Status:** Planning complete. Implementation not started.

---

## 1. Accepted Specification

### What it does
Convert the current local `python app.py` agent harness into a globally installable pip CLI tool (`dostuff`), with centralized config (`~/.dostuff/`), configurable data stores (global vs project mode), auto user ID, session list selection, and multi-path skill loading.

### Install methods
- `pip install .` — basic
- `pipx install .` — **recommended** (isolated venv, clean global CLI)
- `uv tool install .` — fast alternative

### CLI commands
| Command | Description |
|---------|-------------|
| `dostuff` | Run agent (default or selected session) |
| `dostuff --session S --user U` | Resume specific session |
| `dostuff --session S` | Resume session (user auto-loaded) |
| `dostuff init` | Initialize project config in CWD |
| `dostuff config` | Display/edit config |
| `dostuff doctor` | Health checks |
| `dostuff skills` | Manage skills |

### Data directory structure (`~/.dostuff/`)
```
~/.dostuff/
├── config.yaml       # model, identity, data mode, skills dirs
├── mcp.json          # MCP server config
├── identity.md       # agent identity/default persona
├── user_id           # auto-generated UUID (first run)
├── skills/           # user global skills
│   └── *.md
├── data/
│   ├── sessions.db   # SQLite session history
│   └── chroma/
│       ├── semantic/ # Chroma semantic memory (user_id-scoped)
│       └── episodic/ # Chroma episodic memory (user_id + session_id-scoped)
└── .dostuff/         # optional project override
    ├── config.yaml
    ├── mcp.json
    ├── identity.md
    ├── skills/
    └── data/
```

### Config hierarchy
CLI args → project config (`.dostuff/config.yaml`) → global config (`~/.dostuff/config.yaml`)

### Config schema (`config.yaml`)
```yaml
model:
  name: "gpt-4o"
  # ... litellm params
identity:
  file: "~/.dostuff/identity.md"
data:
  mode: global      # global | project | auto
  global_dir: "~/.dostuff/data"
skills:
  dirs:
    - "~/.dostuff/skills"
    - "~/.agents/skills"
    - ".dostuff/skills"  # project (when mode=project)
mcp:
  config_path: "~/.dostuff/mcp.json"
```

### Skill load order (no package-bundled skills)
1. `~/.dostuff/skills/` — user global skills
2. `~/.agents/skills/` — global agent skills (npx convention)
3. `.dostuff/skills/` — project-specific skills (when data.mode=project)

### User ID behavior
- First run: auto-generate UUID → save to `~/.dostuff/user_id`
- Future runs: reuse from `~/.dostuff/user_id` unless `--user` provided

### Session selection behavior
- No `--session`: query SQLite for past sessions (session_id, working_dir, timestamp)
- Present selectable list → user picks, or default to most recent
- Session store includes `working_dir` column

### Working directory behavior
- SQLite session store records `working_dir` at session create/update
- On resume: if session's `working_dir` differs from current CWD → change CWD automatically

### Three stores architecture
| Store | Technology | Scope |
|-------|-----------|-------|
| Semantic memory | Chroma | `user_id`-scoped |
| Episodic memory | Chroma | `user_id` + `session_id`-scoped |
| Session history | SQLite | `session_id`-scoped |

### No backward compatibility
Clean break from `python app.py` invocation. No legacy config detection.

---

## 2. Approved Implementation Plan

See `openspec/changes/global-pip-cli/tasks.md` for full task list.

### Phase 1: Package Foundation
- [ ] Create `pyproject.toml` with `[project.scripts] dostuff = "dostuff.cli:main"`, dependencies, package name. Document pip/pipx/uv in docs.
- [ ] Create `src/dostuff/` directory structure (`__init__.py`, `cli.py`, `config.py`, `core/`, `memory/`).
- [ ] Move `app.py` logic into `src/dostuff/core/agent.py` (refactor, not full rewrite).
- [ ] Write `src/dostuff/cli.py` using typer: `main`, `init`, `config`, `doctor` subcommands.
- [ ] Confirm `pip install .` creates `dostuff` entry point; test `pipx install .` and document both.
- [ ] Verify `dostuff` from random directory (not repo root) — should work globally.

### Phase 1b: User ID & Session Selection
- [ ] Implement first-run UUID generation (`~/.dostuff/user_id`); reuse by default.
- [ ] Implement session list query from SQLite (session_id, working_dir, timestamp) with selectable output.

### Phase 2: Global Config & Directory
- [ ] Implement `src/dostuff/config.py` with `Config` class: reads `~/.dostuff/config.yaml`, resolves `.dostuff/` override, applies CLI args.
- [ ] Create default `config.yaml` template.
- [ ] Implement `dostuff init`: creates `CWD/.dostuff/config.yaml` with `data.mode: project`.

### Phase 3: Data Architecture Redesign
- [ ] Modify memory initialization to use configurable `data_dir` (from config) instead of hardcoded `chroma_data/`/`agent_sessions.db`.
- [ ] Create `~/.dostuff/data/` structure on first run (global mode); create `CWD/.dostuff/data/` when project mode active.
- [ ] Ensure SQLite DB (`sessions.db`) and Chroma collections (`semantic/`, `episodic/`) use `data_dir` base path.
- [ ] Confirm `user_id` + `session_id` filtering remains intact across both modes.

### Phase 3b: Working Directory in Session
- [ ] Add `working_dir` column to SQLite session DB; save CWD at session create/update.
- [ ] Implement resume logic: if selected session's working_dir differs from current CWD, change CWD automatically.

### Phase 4: Skills & Packaging
- [ ] Add user skills directory (`~/.dostuff/skills/`) to registry loader.
- [ ] Add global agent skills directory (`~/.agents/skills/`) to registry loader.
- [ ] Add project skills directory (`.dostuff/skills/`) when `data.mode` is `project`.
- [ ] Remove old `requirements.txt` dependency tracking; rely solely on `pyproject.toml`.

### Phase 5: Validation & Exit
- [ ] Test `dostuff` from random directory (not repo root).
- [ ] Test `dostuff --session S --user U` resolves session/user correctly.
- [ ] Test `dostuff init` then run — confirms `.dostuff/` created with isolated data.
- [ ] Confirm no backward-compat code remains.

### Current TODO
**Phase 1: Package Foundation** — next to implement.

---

## 3. Decisions That Changed the Original Approach

| Decision | Before | After |
|----------|--------|-------|
| Install method | `python app.py` from CWD | Global pip CLI (`dostuff`) |
| Config format | `DOSTUFF.md` + JSON | `config.yaml` (YAML) |
| Data paths | Hardcoded CWD | Configurable via `~/.dostuff/data/` (global) or `CWD/.dostuff/data/` (project) |
| User ID | Required as CLI arg | Auto-generated UUID on first run, persisted to `~/.dostuff/user_id` |
| Session selection | Only via `--session` | Pi-style list from SQLite + selectable or default to latest |
| Session working_dir | Not stored | Stored in SQLite; resume changes CWD if different |
| Skills bundled | `skills/` dir in repo | **No package-bundled skills**. All skills from `~/.dostuff/skills/`, `~/.agents/skills/`, `.dostuff/skills/` |
| Backward compat | Needed | Clean break — no legacy detection |
| Data storage research | — | Claude Code (flat files), Hermes (bounded markdown), OpenHands (file-based), Aider (git-based) all reviewed |
| Install tools | pip only | pip / pipx (recommended) / uv |

### Other agents compared
- **Claude Code**: flat files + JSONL session logs, no DB
- **Hermes**: bounded `MEMORY.md`/`USER.md` + SQLite trajectory
- **OpenHands**: `~/.openhands/microagents/` + workspace repo microagents + event stream
- **Aider**: git-integrated, stateless session logs

---

## 4. Commands Run and Latest Results

| Command | Result |
|---------|--------|
| `openspec store list --json` | No registered stores. Using nearest local root. |
| `openspec new change "global-pip-cli"` | Created at `openspec/changes/global-pip-cli/` |
| `openspec status --change "global-pip-cli" --json` | `isComplete: true`, all artifacts done |
| `openspec instructions <artifact> --change "global-pip-cli" --json` | Instructions retrieved for all 4 artifacts |
| `openspec list --json` | Store not yet named; context retrieved |

**No implementation commands run yet.** Planning only.

---

## 5. Known Risks and Open Questions

### Risks
1. **`chdir` behavior**: Automatically changing CWD on session resume could be surprising/confusing if scripts rely on starting directory. Consider a confirmation prompt or `--force` flag.
2. **SQLite/Chroma path across modes**: Switching from CWD-relative to absolute paths (via `~/.dostuff/`) means existing session data (at CWD paths) won't auto-migrate. Manual migration needed.
3. **Windows paths**: `~/.dostuff/` expands differently on Windows (`C:\Users\<user>\.dostuff\`). Config module must handle `os.path.expanduser()` correctly.
4. **Data isolation confusion**: Users may not understand global vs project mode. `dostuff doctor` and `dostuff config` should clearly show active mode and data location.

### Open Questions
1. **Should `chdir` on resume prompt or auto-change?** Currently spec says "or prompt". Decide: always auto, always prompt, or config flag?
2. **Migration path for existing users**: Should we provide a `dostuff migrate` command to copy CWD data to `~/.dostuff/`? Or document manual steps?
3. **`skills/` repo contents**: The 14 existing skills (skill-creator, database-query-strategy-adviser, etc.) are NOT going into the package. Should they be:
   - (a) Left in the repo only — users clone/repo-reference them?
   - (b) Copied to `~/.dostuff/skills/` on first run?
   - (c) Referenced from original repo path?
4. **`openspec-propose` artifacts location**: Created at `openspec/changes/global-pip-cli/` in the agent-1 repo. Should there be a separate `dostuff-openspec` repo (like Pi's separate store)?
5. **MCP config migration**: Current `mcp_config.json` → `~/.dostuff/mcp.json`. Should we support loading from old path as fallback during transition?

---

*Last updated: 2025-08-25 (session: package-structure-sesh)*

## Post-Implementation Refinements
- Bash tool (dostuff/tools/bash/) with HITL guardrail (ConfirmationRequired)
- Resume subcommand removed; --session handles resume + CWD restore
- .env override confirmed; session meta preserved on resume
- MCP cleanup noise suppressed; run_code removed
