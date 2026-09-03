## Purpose

Provides a globally installable CLI entry point `dostuff` that replaces direct `python app.py` invocation, with working-directory-aware execution and argument parsing.

## ADDED Requirements

### Requirement: CLI entry point installed
The system SHALL install a `dostuff` console script via `pyproject.toml` `[project.scripts]` that calls into the package CLI module.

#### Scenario: Pip install activates command
- **WHEN** user runs `pip install .` (or `pip install dostuff` from published package)
- **THEN** the `dostuff` command is available globally in any directory

### Requirement: CLI detects working directory
The system SHALL detect the current working directory (`os.getcwd()` or equivalent) and use it for file-tool operations and optional project config resolution.

#### Scenario: Run from project root
- **WHEN** user runs `dostuff` inside a directory containing `.dostuff/`
- **THEN** the agent treats that directory as the working directory for file operations

### Requirement: Auto-generate user ID on first run
The system SHALL generate a UUID `user_id` on first run, persist it to `~/.dostuff/user_id`, and reuse it by default in future runs unless `--user` is provided.

#### Scenario: First run creates user
- **WHEN** `dostuff` is executed with no `--user` and no existing `~/.dostuff/user_id`
- **THEN** a UUID is generated, saved to `~/.dostuff/user_id`, and used for the session

### Requirement: Session selection from past sessions (Pi-style)
The system SHALL query the SQLite session store to list past `session_id`s (with working_dir and timestamp); when `--session` is omitted, present a selectable list or use the most recent by default.

#### Scenario: Select past session
- **WHEN** user runs `dostuff` with no `--session`
- **THEN** system shows past session list (from SQLite); user selects or defaults to latest

### Requirement: CLI supports session and user arguments
The system SHALL parse `--session` and `--user` CLI arguments (with optional defaults) and pass them to the agent core.

#### Scenario: Resume session
- **WHEN** user runs `dostuff --session SESS_ID --user USER_ID`
- **THEN** agent loads the specified session and user context from the configured data store

### Requirement: Resume session changes working directory
When resuming a session from a different working directory, the system SHALL change CWD to the stored session `working_dir` (or prompt) before continuing file operations.

#### Scenario: Resume from different directory
- **WHEN** user selects a session whose `working_dir` differs from current CWD
- **THEN** system updates CWD to that `working_dir` automatically

### Requirement: CLI provides subcommands
The system SHALL provide `dostuff init` to initialize project config in CWD, `dostuff config` to display/edit config, and `dostuff doctor` for health checks.

#### Scenario: Project initialization
- **WHEN** user runs `dostuff init` in a directory without `.dostuff/`
- **THEN** system creates `CWD/.dostuff/config.yaml` with `data.mode: project` and default settings
