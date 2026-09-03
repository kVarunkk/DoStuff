## Purpose

Defines the pip-installable package structure, entry point definition, built-in skills bundle, and dependency metadata for the harness.

## ADDED Requirements

### Requirement: pyproject.toml defines package
The system SHALL include `pyproject.toml` specifying the package name (`dostuff`), version, entry script (`dostuff = "dostuff.cli:main"`), and dependencies.

#### Scenario: Install from source
- **WHEN** user runs `pip install .` in repo root
- **THEN** package installs correctly with `dostuff` command available

### Requirement: Source layout
The system SHALL use `src/dostuff/` package layout rather than flat package at root.

#### Scenario: Import from installed package
- **WHEN** installed code imports `dostuff.cli`
- **THEN** import resolves through `src/dostuff/cli.py`

### Requirement: Built-in skills bundled
The system SHALL include built-in skills in `src/dostuff/skills/` distributed with the package, loaded at runtime without requiring separate download.

#### Scenario: Skill available after install
- **WHEN** user calls `dostuff` after `pip install`
- **THEN** built-in skills (like `skill-creator`) are available in the agent's skill registry

### Requirement: Skills load from user/global directories
The system SHALL search for skills in: (1) `~/.dostuff/skills/`, (2) `~/.agents/skills/` (matching npx global install convention), (3) `.dostuff/skills/` (project-specific when data.mode is project). Merge into a single registry.

#### Scenario: Skill found in user dir
- **WHEN** a user skill exists at `~/.dostuff/skills/my-skill.md`
- **THEN** agent loads it alongside global and project skills

### Requirement: Dependency list maintained
The system SHALL declare all required dependencies (`litellm`, `pydantic`, `typer`/`click`, `chromadb`, `aiosqlite`, etc.) in `pyproject.toml` instead of `requirements.txt`.

#### Scenario: Clean install
- **WHEN** user runs `pip install dostuff` in a fresh environment
- **THEN** all dependencies resolve automatically from `pyproject.toml`
