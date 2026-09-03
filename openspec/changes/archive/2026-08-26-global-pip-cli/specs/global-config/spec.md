## Purpose

Defines the `~/.dostuff/` global configuration directory structure, file formats, and resolution hierarchy (global defaults → project overrides → CLI args).

## ADDED Requirements

### Requirement: Global config directory created
The system SHALL create `~/.dostuff/` (respecting `XDG_CONFIG_HOME` if set) on first use if it does not exist.

#### Scenario: First run creates directory
- **WHEN** user runs `dostuff` for the first time
- **THEN** `~/.dostuff/config.yaml`, `~/.dostuff/identity.md`, and `~/.dostuff/mcp.json` are created with defaults

### Requirement: Config uses YAML format
The system SHALL read and write configuration from `~/.dostuff/config.yaml` (and `CWD/.dostuff/config.yaml` when `data.mode` is `project` or `auto` with project detected) using YAML format.

#### Scenario: Read global config
- **WHEN** agent starts without project config
- **THEN** it reads settings from `~/.dostuff/config.yaml` (model, identity file path, data mode, skills directories)

### Requirement: Config hierarchy resolution
The system SHALL resolve settings in order: CLI args override project config (`.dostuff/config.yaml`), which overrides global config (`~/.dostuff/config.yaml`).

#### Scenario: Project overrides model
- **WHEN** `CWD/.dostuff/config.yaml` specifies `model.name` different from `~/.dostuff/config.yaml`
- **THEN** the project value is used for the session
