## Purpose

Defines how the three agent memory/data stores (semantic, episodic, session history) are persisted, scoped to user/session IDs, and configured for global or project mode.

## ADDED Requirements

### Requirement: Three store types preserved
The system SHALL maintain three distinct storage types: (1) semantic memory in Chroma (user_id-scoped), (2) episodic memory in Chroma (user_id + session_id-scoped), (3) session history in SQLite (session_id-scoped).

#### Scenario: Semantic retrieval for user
- **WHEN** agent searches semantic memory for user `U-001`
- **THEN** only collections/documents tagged with `user_id: U-001` are returned, regardless of session

### Requirement: Session list queryable for selection
The SQLite session DB SHALL support querying session records (with `session_id`, `user_id`, `working_dir`, `timestamp`) so the CLI can present a selectable list of past sessions.

#### Scenario: List past sessions
- **WHEN** user runs `dostuff` without `--session`
- **THEN** system queries `SELECT session_id, working_dir FROM sessions` and shows results

### Requirement: Configurable data directory
The system SHALL use the path specified by `data.global_dir` (default `~/.dostuff/data`) for the global store, or `CWD/.dostuff/data` when `data.mode` is `project` or `auto` detects a local `.dostuff/` directory.

#### Scenario: Global mode uses central store
- **WHEN** `data.mode` is `global` (default)
- **THEN** SQLite and Chroma collections are stored under `~/.dostuff/data/`

### Requirement: Project mode isolates data
The system SHALL create and use `CWD/.dostuff/data/` (with its own SQLite and Chroma collections) when `data.mode` is `project`.

#### Scenario: Project isolation
- **WHEN** user runs `dostuff init` then `dostuff`
- **THEN** session DB and Chroma collections are created under `CWD/.dostuff/data/` and contain only this project's interactions

### Requirement: Session history includes working directory
The SQLite session DB SHALL store a `working_dir` column recording the CWD at session creation/update, enabling resume from any directory.

#### Scenario: Resume from different directory
- **WHEN** user selects past session S with `working_dir=/project/old/`
- **THEN** system uses that directory for file-tool context and updates CWD accordingly

### Requirement: User and session IDs remain architecture
The system SHALL continue to require and use `user_id` (for semantic and episodic) and `session_id` (for episodic and session DB) to scope and filter stored data.

#### Scenario: Same user, different session
- **WHEN** user `U-001` starts session `S-A` and later `S-B`
- **THEN** episodic memory for `S-A` is isolated from `S-B`, while semantic memory for `U-001` is shared across both
