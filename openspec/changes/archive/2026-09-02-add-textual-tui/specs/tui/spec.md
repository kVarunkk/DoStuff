## Purpose

Interactive terminal interface for the `dostuff` agent that provides real-time message display, session context, and structured input using the `textual` Python library.

## ADDED Requirements

### Requirement: Interactive session view
The system SHALL render agent messages in a scrollable panel within a full-screen TUI.

#### Scenario: User submits message
- **WHEN** user types input and presses Enter
- **THEN** the input is submitted to the agent loop and the response appears in the messages panel

### Requirement: Thread-safe agent integration
The system SHALL use `run_worker()` and `call_from_thread()` to synchronize async agent responses with the TUI render loop.

#### Scenario: Agent responds asynchronously
- **WHEN** agent completes processing a prompt
- **THEN** the response updates the messages component without blocking the TUI event loop
