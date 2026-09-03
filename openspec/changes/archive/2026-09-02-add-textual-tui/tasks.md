## 1. Setup

- [x] 1.1 Add `textual` to pyproject.toml and verify `pip install -e .` succeeds with the new dep
- [x] 1.2 Confirm `textual_proxy.py` runs in `.test_venv` and displays the TUI shell

## 2. Core Implementation

- [x] 2.1 Create `dostuff/cli_tui.py` with `DostuffTUI` App class (Header, Messages, Input widgets) and verify it boots
- [x] 2.2 Wire async agent loop via `run_worker()` and `call_from_thread()` and verify echo response updates Messages
- [x] 2.3 Replace the mock `_agent_loop` with the real `dostuff.core.agent.run()` and verify a real LLM response streams in

## 3. TUI Features

- [x] 3.1 Previous steps visible on resume (load session history into widget on mount)
- [x] 3.2 Tool calls visible — via adapter.emit() (real-time, not post-turn scan)
- [x] 3.3 Input clears on submit; "You: ..." shown immediately before agent responds
- [x] 3.4 Shift+Enter inserts newline; Enter submits
- [x] 3.5 /exit saves long-term memory (verify current_session_history populated, save_memories_and_exit called)

## 4. Adapter Integration

- [x] 4.1 Define adapter interface (emit/ask/input) in cli_tui module
- [x] 4.2 loop() uses adapter for print/input — not direct print/input
- [x] 4.3 HITL: adapter.ask() prompts user via input widget; returns y/n to loop()

## 5. Visual Polish (cli_tui.py only)

- [x] 5.1 Remove Header widget; set Screen background transparent; remove all custom background colors
- [x] 5.2 Input: replace `border: solid blue` with `border-top: solid white; border-bottom: solid white`; remove left/right borders
- [x] 5.3 RichLog: remove `border: solid green`
- [x] 5.4 Per-role colors via RichLog markup: User=`bright_blue`, Agent=default, Tool=`dim italic`, Confirm=`bold yellow`, Error=`bold red`, System=`dim`

## 6. Token Tracking (agent loop + cli_tui)

- [x] 6.1 Verify `agent_loop()` returns usage dict (prompt_tokens, completion_tokens, total_tokens); extract token counts from response
- [x] 6.2 Thread usage through `loop()` return value or `adapter.emit("usage", {...})`; receive in `_run_turn`
- [x] 6.3 Display per-turn usage appended to agent response line (`↑N ↓M`); human-readable format
- [x] 6.4 Maintain running session total; store on `DostuffTUI`

## 7. Status Bar (cli_tui.py, after 6)

- [x] 7.1 Add Static/Label widget below input (height auto, no border, dim color)
- [x] 7.2 Display CWD + session-id (short hash) + resumed/new indicator
- [x] 7.3 Display cumulative session tokens (↑total ↓total) alongside CWD
