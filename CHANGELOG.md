# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Mid-turn steering: `/steer <message>` sends a correction to the turn that is running — the model reads it at the next tool boundary, inside the same turn, instead of waiting for the turn to end — and it is how every plain message is routed by default (`ui.mid_turn_messages: queue` batches them after the turn instead).

### Added
- **`/steer <message>`** — a steering message, prepared like a prompt (`@mentions` expanded, pastes resolved, in order) and read by the loop at the next tool boundary. It appears in the transcript as a user message right after the tool output it was reacting to, and a final answer that arrives while a correction is still waiting does not end the turn: the model reads it and redoes or extends the answer, before the `Stop` hooks run. With no turn running it is an ordinary prompt (it starts a turn).
- **`ui.mid_turn_messages: queue | steer`** (default `steer`) — what a **plain** message does while a turn runs: steers, read at the next tool boundary, or (with `queue`) waits and runs after the turn. `/steer` steers in both modes; skills and custom commands stay queued in both (a skill is a turn's prompt, not a correction); built-ins (`/btw`, `/queue`, `#note`, …) run now and `/clear`, `/resume`, `!command` are refused while a turn runs, exactly as before.
- **The bar, the loader and `/queue`** — the bar leads with `→ N for the agent · read at the next tool boundary` above the queued group, the loader counts `· 1 steering · 2 queued`, `/queue` lists the steering messages first (marked `→`, with *Edit* / *Remove*, and "That message was already read." once the loop has taken one), and every queued prompt gained *Send now (at the next tool boundary)*.
- **`Ctrl+B` and the latency rule** — delivery waits for the tool call in flight (nothing is interrupted to deliver sooner); while a message waits behind a running command the bar says `waiting for a running command · Ctrl+B to background it`, and `Ctrl+B` returns the call to the loop at once, so the message is read while the command keeps running.
- **Leftovers** — a message that outlives its turn is never lost: on `Esc` the raw text goes back into the input ahead of the queued prompts; otherwise it runs next (`1 message arrived after the turn ended; running it next`), re-prepared from what you typed. Either way the text is written to the history exactly once.

### Changed
- `UserPromptSubmit` runs for each correction at the boundary, so a hook can block it (shown like a blocked prompt, never reaching the model), and a preparation failure is reported instead of half-delivered.
- **A delivered correction is framed for the model**: the stored user step is your text wrapped in `The user sent a new message while you were working:` above and `Address the message above, then continue the task you were working on.` below, so a mid-turn question is answered *and* the turn gets back to the original task instead of ending on the answer (Claude Code wraps mid-turn messages the same way). What you see while the turn runs, Esc-to-restore and what a leftover re-runs stay exactly what you typed; `/resume` replays the framed text.

### Documentation
- README "Mid-turn steering (`/steer`)" section (the two modes with the routing table, what you see, latency and `Ctrl+B`, leftovers), a Features bullet, a TUI-controls bullet and a row in the commands table; `/steer` in `/help` and the `/` menu.

## [0.2.9] - 2026-11-03

Side questions: `/btw <question>` asks about the conversation while a turn runs, without stopping or queuing it — one model call, no tools, answered in a panel above the input and never added to the history; `f` hands it to a background agent that can check with tools.

### Added
- **`/btw <question>`** works mid-turn and when idle. It is never queued and it does not interrupt the running turn. The model sees the conversation up to the last completed step, with the turn's own system prompt and tools, and answers in one response. No tool runs: if the model asks for one, a note under the answer says so.
- **The panel** sits above the input and leaves the conversation visible. It shows `Asking…` until the answer arrives, and `↑`/`↓`/`PgUp`/`PgDn` scroll a long answer. The footer shows the call's tokens, prompt-cache hits and cost. `Esc`, `Enter` or `Space` close it (Esc closes the panel and does **not** cancel the turn), and `c` copies the answer. Only one side question runs at a time, and a second `/btw` while one is still asking is refused. An approval prompt or `ask_user` question that opens meanwhile takes the focus and the first `Esc`.
- **Fork out (`f`)** starts a background `fork` agent with the question, the answer to check and the same view of the conversation. `/agents` shows it, and its report comes back into the conversation like any background agent's. `Esc` on the main turn does not stop it. `/clear` and `/resume` carry it to the new session; `/fork` leaves it with the original session and prints a line saying so.
- Later `/btw`s in a session see the last 20 side questions and answers. `/clear`, `/resume` and `/fork` start a fresh side thread.
- `/help` and the command menu list `/btw <question>`.

### Changed
- A side question never touches the session: its question and answer stay out of the transcript, the stored history, `/export` and compaction, and its cost is not added to the session total. Hooks do not fire for it (no `UserPromptSubmit`, `PreToolUse` or `Stop`).

### Documentation
- README "Side questions (`/btw`)" section (panel sketch, the keys, what the model sees, fork-out, `-p` has no `/btw`), a Features bullet, a TUI-controls row and a row in the hooks "Coming from Claude Code" table.

## [0.2.8] - 2026-11-03

Structured questions: the agent can stop and ask you instead of guessing — up to 4 questions per call, 2–4 options each, multi-select and free text, answered in a panel in place of the input and returned as one parseable result.

### Added
- **`ask_user` tool**: `questions` is 1–4 questions, each with 2–4 `options` (`label`, optional `description`) or no options at all for a free-text answer, an optional `header` (the tab label) and `multi_select`. The result is a fixed block — `Asked the user N questions; they answered:`, then `1. <question>` / `   → <answer>` — written and parsed by one helper, so the tool, the transcript and `/export` cannot drift apart. A malformed call is refused with an `error:` result naming the first fault (too many questions, too few or too many options, duplicate labels, duplicate questions) before anything is shown; over-long text is clamped, not refused.
- **The panel**: it renders in place of the input, one tab per question, `▌` on the question you are on and `✓` on an answered one — both in the same two-column slot, so the strip does not re-flow as you answer. `←`/`→`/`Shift+Tab` move between questions, `↑`/`↓`/a number picks, `space` toggles a multi-select question (the answer comes back in option order), and `Submit answers` is the only thing that sends: it stays dim and inert until every question has an answer. Picking an option never submits on its own.
- **`Other`** is the last row of every question: it takes the input for a typed answer (collapsed to one line). A call whose questions are *all* free text skips the panel and the input takes each answer in turn, with the last Enter as the submission.
- **Esc** cancels the questions and ends the turn, like Esc on an approval; Ctrl+Q and `/exit` cancel them first and then quit. Every cancel path still leaves exactly one tool step, so the transcript and `/resume` stay honest.
- **Headless `dostuff -p`** asks on the terminal when stdin and stderr are both a TTY: each question and its numbered options go to stderr, and it reads one line per question (`2`, `1,3`, free text; an empty line is `(no answer)`). With piped stdin or redirected stderr it answers "No user is available in this run" and the model states an assumption instead of failing.
- The answers render as one `⎿` block, a row per question → answer, live and on `/resume`; nothing is truncated, because reading back what you answered is the point. Ctrl+O has nothing to expand there.
- A waiting question sets the window title to `waiting` and fires the usual approval notification, and the call label in the transcript is the first question plus `+N more`.

### Changed
- The system prompt gains an `## Asking the user` section: ask when the answer changes what you build and the repo can't tell you, never to confirm a plan or to ask permission, everything you need in one call, and leave `options` out for free text.
- Plan mode now tells the model to ask what it needs *before* writing the plan.
- `ask_user` never asks permission (it is a read tool) and is never timed out or dispatched beside another call, so a question can't race a streaming result or time out while you think.
- Subagents never get it: it is in the restricted set, and `map_tools` now re-checks the restriction *after* mapping Claude Code's names, so an agent file listing `AskUserQuestion` no longer slips past it.
- `/tools` says which tool talks to you.

### Fixed
- A cancelled batch no longer skips `PostToolUse` for a call that returned on its own: Esc on a question resolves it synchronously, so the hook sees the cancel text it was meant to see.
- The sequential pass checks the cancel flag before starting a call, so an `ask_user` cancelled by the batch just finished no longer leaves an unawaited coroutine.

### Documentation
- README "Asking the user" section (panel sketch, the keys, multi-select and `Other`, Esc, the transcript block and the exact text the model receives, headless), plus a Features bullet, the tools-table row and the `AskUserQuestion` → `ask_user` mapping with `multiSelect` → `multi_select`.

## [0.2.7] - 2026-11-02

Polish for the TUI's conversation view: the transcript no longer shifts when a selector takes the input's place, the jump-to-bottom button is a centered pill, and plan mode stops mistaking dotted attribute paths for `.dostuff` settings edits.

### Fixed
- The TUI no longer bounces the conversation when a selector takes the input's place or gives it back (approvals, `/model`, `/resume`, `/jobs`, `/queue`, Ctrl+R). Textual pins an anchored scrollbox using `container_size`, which it only writes back after arranging, so the first frame after the height change used the old viewport height and drew the transcript a few rows off before snapping into place. `#messages` is now a `Transcript`, which hands the compositor the size it is laying out.
- The jump-to-bottom button is a small centered pill instead of a full-width bar, so it no longer
  bands the conversation with its background, and only the pill itself is clickable.
- Plan mode no longer treats `cli_tui.DostuffTUI` (or any other dotted attribute) as a `.dostuff`
  path: only a `.dostuff` that stands alone as a path counts as a settings change.

## [0.2.6] - 2026-11-01

Subagents rebuilt: typed agents defined in Markdown files (Claude Code's `.claude/agents/` work too), parallel and background agents you can message and stop, forks that share the conversation, per-agent models, stored transcripts and an `/agents` viewer. Also: a `/tools` command, argument hints for slash commands, and a faster TUI start.

### Added
- **`agent(description, prompt, subagent_type, model, run_in_background, isolation)`** tool replaces `delegate_to_subagent`. Every run is owned by a session registry with a 6-hex id; the result ends with `agent_id: <id> (send_message to continue)`. Several `agent` calls in one response run in parallel, at most `agents.max_concurrent` (default 4) at a time, with the rest queued.
- **Agent types** from Markdown files with YAML frontmatter (`name`, `description`, `tools`, `model`, `max_iterations`, body = system prompt), searched in `.dostuff/agents/` and `.claude/agents/` in the project, then the same two under `~`. The first `name` found wins. `agents.read_claude_agents: false` skips the `.claude` folders. Claude Code tool names (`Read`, `Edit`, `Bash`, `mcp__*`, …) are mapped, and unknown or restricted ones are dropped with a warning. Files are re-read when they change. Built-in types: `general-purpose` (every subagent tool), `explore` and `plan` (read-only, and shell commands that aren't read-only are denied).
- **`fork`** type: a subagent that starts from a copy of the current conversation with the main agent's system message byte-identical, so the prompt cache is reused. It always runs on the main model.
- **Per-agent models**: the call's `model=`, then the type's `model:`, then the main model. Aliases come from `agents.models`. `sonnet`/`opus`/`haiku` fall back to the main model with a warning. `agents.explore_model` sets the built-in explore model. Cost, provider keys and the context window follow the agent's model.
- **Background agents**: `run_in_background=true` returns at once, and a `<background-agent …>` notice with the report arrives when the agent ends. In the TUI that is a `⚙ agent "…" finished after 2m 10s` line plus a notification, and an idle TUI resumes the main agent for it (`agents.wake_on_end`, sharing the 3-turn limit with jobs). Running agents show as `⚙ 2 agents` / `⚠ 1 agent waiting` on the status line, carry over on `/clear` and `/resume`, and quitting asks before stopping them.
- **Ctrl+B** also moves running or queued foreground agents to the background. Their call returns at once and Esc no longer stops them.
- **`send_message(to, message)`**: queues a message for a running agent (read before its next step), or continues an ended agent on its full history with the same id (a new worktree if its old one was removed). **`stop_agent(agent_id)`**: stops an agent, kills its jobs and closes its worktree.
- **`create_agent_type`** tool: the main agent can write a new agent type file (`.dostuff/agents/<name>.md`, or `~` with `scope="user"`) when asked. It always asks, with the whole file shown, and the type is usable at once.
- **Transcripts**: each agent's steps and compactions are stored in the session DB (schema v4, `agent_runs` / `agent_steps`), never in the main history. After a restart or `/resume` the session's agents come back as records (`interrupted` if they were running) and can be continued with `send_message`.
- **`/agents`** lists this session's agents (status, runtime, type, tokens, cost). Picking one opens its live transcript in place of the conversation, with Answer approval, Stop it and Show its type. `/agents types` lists the types, `/agents types <name>` shows one type's full config and system prompt, and `/agents reload` re-reads the files.
- Approval prompts from subagents are titled `[agent: <description> · <type>]` and queue one at a time. A background agent's prompt waits while you're typing or a picker is open.
- **Hooks**: new `SubagentStart` event (matcher = agent type; its output is added to the agent's prompt; it can't block). `PreToolUse`, `PostToolUse` and `SubagentStop` payloads inside an agent carry `agent_id` and `agent_type`, and `SubagentStop`'s matcher matches the agent type.
- **Headless**: `dostuff -p` waits for background agents and runs a turn on their notices (at most 3, then `⚙ Not waiting for more agents…` and exit 0). It stops every running agent at the end and on Ctrl+C. `stream-json` prints `{"type": "agent", …}` events when an agent starts and ends, and text mode prints background agents' start and end lines on stderr.
- **`agents:` config block**: `max_iterations` (default: `loop.max_iterations`), `max_concurrent`, `wake_on_end`, `read_claude_agents`, `explore_model`, `models`.
- **`/tools`** lists every tool offered to the main agent, with when it asks in `ask` mode, whether subagents get it, and a one-line description, then each MCP server's tools.
- Slash commands that take arguments show a dim hint after the name (`/rename <title>`, `/permissions [ask | allowlist | auto | plan]`). Tab from the `/` menu adds a space so the hint shows.
- Every shortcut now reports what it did in the input's border hint: Esc with nothing to cancel, Ctrl+C with nothing selected (instead of Textual's quit toast), Ctrl+O and Ctrl+B.

### Changed
- The main system prompt lists the agent types and explains when to use explore / plan / general-purpose / fork, parallel calls, background agents, `send_message` follow-ups and not polling.
- A subagent with no `tools:` gets every subagent tool, so `general-purpose` can write files.
- Skills are deduplicated by name (the first one found wins, project before home), and a directory reached twice is scanned once.

### Removed
- `delegate_to_subagent`. Old sessions that used it still load, replay and render.

### Fixed
- The TUI no longer stays blank for seconds at startup when an agent file with a `tools:` line exists (parsing it imported litellm).
- The project-hooks trust prompt and tool approvals no longer make the transcript jump when answered, and the first turn shows one loader instead of two.

### Documentation
- README "Subagents" section (tools, types and search order, frontmatter, Claude Code tool mapping, models, background agents, Ctrl+B, headless, `/agents`, transcripts, hooks, config). `SubagentStart` added to the Hooks tables, `/tools` and `/agents` added to the command table, and the `agents:` block documented in `templates/config.example.yaml`.

## [0.2.5] - 2026-11-01

Background shell jobs: start long-running commands, read their output, stop them, and move a running command to the background with Ctrl+B. Also: config files edited by hand apply live, and queued prompts can be viewed, edited, run next or removed.

### Added
- **Background jobs**: `bash_command(run_in_background=true, background_id=…)` starts a job and returns its id, pid and log paths. Ids are case-folded `[a-z0-9][a-z0-9._-]{0,31}`, otherwise `bg-<n>`; at most 8 running background jobs per owner. A job that exits within 200 ms is reported as `exited immediately with code N` with its first stderr lines.
- **`bash_output(background_id, filter, wait_for, timeout)`** returns new stdout / stderr since the last read; `wait_for` waits (1–110 s) for a matching line or the job's exit instead of sleeping. **`kill_shell(background_id)`** kills the process tree and returns the last 20 lines per stream. Both are read tools, available to subagents that have `bash_command`.
- **Completion notices**: a background job that ends on its own, or is stopped by the user, is reported to its owner as a `<background-job …>` message before the next model call, and shown in the TUI as `⚙ job "test" exited (1) after 42s` (with a terminal notification when the window is in the background). When the TUI is idle it resumes the agent for it (`jobs.wake_on_end`, default on, at most 3 turns in a row without a user message).
- **Ctrl+B** moves the running turn's shell calls (main agent and subagents) to the background; the call returns the output so far and the job id. `!command` is never moved.
- **`/jobs`** lists the session's background jobs; `/jobs <id>` shows the last 40 lines of each stream and the log paths; `/jobs kill <id>` stops one. Running jobs also show as `⚙ 2 shells` on the status line, after the `ui.status_bar` items; clicking it, or ↓ in an empty input, opens a picker of the jobs. A job's entry shows its last 8 lines of output (stderr in red) and its log path, with "Refresh output", "Stop it" and "Back to the list".
- **Waiting on background jobs**: in the TUI the agent tells you what's running and ends its turn for a long job, then is resumed when the job ends; it still uses `wait_for` for short waits such as a "ready" line. Headless `-p` waits with `wait_for`, since its jobs stop when the run ends.
- **Prompt queue**: queued prompts are listed above the input (up to 4 rows, then `… N more`) instead of a count. `/queue`, or a click on the list, opens a picker to edit a queued prompt (it moves into the input), run it next or remove it, or to clear the queue; `/queue clear` drops them all.
- **Lifecycle**: jobs are owner-scoped. A subagent's jobs are killed when it ends, headless `-p` kills every job at the end and on Ctrl+C, and quitting the TUI confirms before stopping them. Main-agent jobs keep running across `/clear` and `/resume` and move to the new conversation, which gets their end notices and can read or stop them. Logs live in `~/.dostuff/jobs/<session>/` and are pruned after 7 days; at startup the TUI warns about jobs an earlier session never marked ended.

### Changed
- Every shell command runs as a job writing to log files, so `cmd &` returns as soon as the shell exits (the 2 s drain grace and the "Output capture stopped" note are gone). Timeout and Esc still kill the whole process tree.
- **Config edits on disk apply live**: the TUI checks the user and project `config.yaml` and the MCP config files every second, idle or mid-turn, and applies a change without a prompt or restart (status bar, iterations, paste collapse, hooks, and MCP servers connecting, reconnecting or disconnecting). Hooks and permissions also re-read a changed file before each dispatch or tool call, in `-p` too, so a hook removed or a `permissions` rule edited by hand takes effect from the next call. YAML that doesn't parse keeps the loaded settings and the error is shown once.
- Picker options (approvals, `/model`, `/resume`, the jobs list) can be clicked.
- Esc on a running turn no longer starts the next queued prompt; the queue moves back into the input.
- The `jobs` status-bar item is retired in favour of the `⚙ N shells` item, which shows regardless of what `ui.status_bar` lists; `jobs` is still accepted without a warning.
- The system prompt describes `run_in_background`, `bash_output`, `kill_shell` and `<background-job>` messages instead of `nohup` / `&` advice.

### Fixed
- Esc on a running `!command` showed `Shell error: Result is not set.` instead of `Cancelled (process killed)`.

### Documentation
- README "Background jobs" section (tools, notices, Ctrl+B, `/jobs`, the jobs picker, logs, lifecycle table, hard-crash caveat) and the prompt queue, plus `bash_output` / `kill_shell` in the tools table and `jobs.wake_on_end` in `templates/config.example.yaml`.

## [0.2.4] - 2026-10-30

Hooks: run your own commands on agent events, with Claude Code's contract.

### Added
- **Hooks** (`hooks:` in `config.yaml`, user and project): shell commands on `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop`, `SubagentStop` and `SessionStart`. Each hook gets a JSON envelope on stdin plus `DOSTUFF_HOOK_*` env vars and runs through the configured shell in the project root (the worktree for a worktree subagent). Exit 0 stdout is context or a JSON decision (both Claude Code spellings accepted), exit 2 blocks with stderr as the reason, any other code is a non-blocking error. `matcher` fully matches the tool name, `if` searches the shell command. Timeouts are clamped to 1–300 s and run outside the tool timeout; a hook is killed with its process tree on timeout or Esc. A Claude Code PreToolUse script works unchanged once its matcher is renamed.
- **PreToolUse** can deny, ask, allow or rewrite a call's arguments. Outcomes merge strictest-wins (`deny` > `ask` > `allow`), and a rewrite is re-sanitized and faces the same prompt / plan-mode gate as a model call. Precedence: plan deny > hook deny > settings ask > hook ask > hook allow > permission layer, so a hook can never loosen plan mode or a settings prompt. `fail_closed: true` turns a crashed, timed-out or misconfigured hook into a deny.
- **PostToolUse** appends each hook's output to the tool result as `<hook name="…">…</hook>`; **UserPromptSubmit** can block a prompt or add context for that turn; **Stop** / **SubagentStop** can keep the agent going (at most `hooks.stop_block_limit` times per turn); **SessionStart** (`startup` / `resume` / `clear` / `compact`) adds session context that is never stored.
- **Project hook trust**: project hooks run only with `project_trust: allow` (read from the user file), or after a yes to a prompt listing every project command and its file. The answer is stored in `~/.dostuff/trusted_hooks.json`, keyed by project root and a hash of the project entries and `.dostuff/hooks/`; editing either asks again. Headless takes `--trust-project-hooks`, skips untrusted project hooks with a warning, and refuses to start (exit 2) when an untrusted one is `fail_closed`.
- **`/hooks`** lists every hook per event in run order with its trust mark, matcher, timeout, command and origin file; `/hooks r` reloads. Hook decisions show as `⚓ PreToolUse bash_command → deny (reason)` lines in the TUI, on headless stderr and as `{"type": "hook"}` in stream-json.
- `dostuff doctor` checks hooks: load warnings, programs missing from PATH, untrusted project hooks, project scripts outside `.dostuff/hooks/`. `read_settings` and `edit_config` know the `hooks` key.

### Security
- A `bash_command` naming `.dostuff` in its command or `cwd` is a settings change: it asks in every mode, `auto` included, with no "always", and a hook `allow` can't silence it. This closes an auto-mode path that could add a user hook without a prompt. Subagent worktree paths stay exempt.

### Documentation
- README Hooks section (events, config example with `.dostuff/hooks/` scripts, stdin / env / exit-code / JSON contract, merge rules, trust model, Claude Code name table); `templates/config.example.yaml` documents every event.

### CI
- macOS added to the test matrix alongside Ubuntu and Windows.

## [0.2.3] - 2026-10-29

OpenCode models now use their real context window instead of an assumed 128k.

### Fixed
- **OpenCode models no longer report an assumed 128k context window.** The gateways publish no `context_length` in `/v1/models` and litellm doesn't map their ids, so every OpenCode Zen / Go model fell back to 128k — `opencode-go/deepseek-v4.1-flash` is 1M, `opencode/minimax-m3` is 512k, `opencode/qwen3.8-max` is 262k. `helpers/models/opencode.py` now carries a per-gateway `_CONTEXT_WINDOWS` table taken from models.dev (the same id can have a different window on each gateway) plus a `context_window()` lookup, and `get_model_token_limit()` resolves `loop.context_window` → table → litellm on the routed name → litellm on the bare id → 128k. The "Unknown context window … assuming 128,000 tokens" notice now only appears when all of those miss, once per model. Auto-compaction thresholds, the status-bar percentage, `/context` and `/compact` budgets follow from it, so OpenCode sessions no longer compact at an eighth of their real window.

### Documentation
- The `loop.context_window` comment in `README.md` and `templates/config.example.yaml` no longer claims litellm is the only source.

## [0.2.2] - 2026-10-28

MCP OAuth sign-ins now survive restarts, and `keyring` is a core dependency.

### Changed
- `keyring` is a core dependency; the `dostuff[keyring]` extra is removed. Without a usable keyring backend, tokens are still kept in memory only, never in a plain file.

### Fixed
- **MCP OAuth no longer signs in on every start.** Tokens over ~1 KB are split across several keyring entries (Windows Credential Manager rejected the whole JSON at its 2560-byte cap, so the save failed silently). HTTP/SSE transports attach the current token per request through an `httpx2.Auth`, refresh it shortly before expiry, and on a 401 refresh once and retry; `reauthenticate()` only opens the browser when the refresh token is rejected. A one-time notice says when a token can only be kept in memory.
- `helpers/mcp/token_store.py` passes mypy again (it failed CI on every environment).

### Documentation
- README lists every install extra (`memory`, `search`, `full`, `dev`) in one table, with a pipx example.

## [0.2.1] - 2026-10-27

OpenCode Zen / Go models, plan mode, custom commands, `# note` memory, `/context`, `/doctor`, `/copy`, richer headless output and terminal title / notifications.

### Added
- **OpenCode Zen / Go models** (`opencode/<id>`, `opencode-go/<id>`, key in `OPENCODE_API_KEY`). litellm has no OpenCode provider, so `helpers/models/opencode.py` routes each model to litellm's anthropic (Claude, some Qwen / MiniMax), `openai/responses/` (GPT / Grok / Muse), gemini or openai chat route, with the gateway as `api_base`. Sends `User-Agent: dostuff/<version>` and `x-opencode-session`. In the setup wizard, `dostuff models add` and `/model`, with model lists from `/v1/models`. Context window and cost use the routed model; prompt-cache breakpoints apply to OpenCode Claude models.
- **Plan mode** (`permissions.mode: plan`, `/permissions plan`, `--permission-mode plan`): reads and read-only shell commands run; file writes, other shell commands and settings changes are denied with a "Plan mode: not run …" result, even if approved with "always" earlier. Web / MCP calls ask as in `ask`. A `## Plan mode` section is added to the system prompt while it's on.
- **Shift+Tab** cycles the session's permission mode ask → auto → plan → allowlist; the input border flashes the new mode and the status bar shows it.
- **`dostuff -c` / `--continue`**: resume the latest session in the current directory (also with `-p`).
- **Headless flags**: `--max-turns N` (model calls; exit 1 when hit), `--allowed-tools` (comma-separated / repeated; `permissions.allow` syntax or a bare tool name), `--output-format text|json|stream-json` (stream-json prints `init`, `assistant`, `tool_call`, `tool_result`, `usage`, `error` / `system`, `result` events, one per line).
- **Custom commands**: `.dostuff/commands/<name>.md` (project) and `~/.dostuff/commands/<name>.md` (user) become `/<name> [args]`, with `$ARGUMENTS` substitution and an optional `description:` frontmatter. Built-ins win a clash, then project, user commands, then skills. Works with `-p "/<name> ..."`.
- **`# note`**: a one-line prompt starting with `#` appends `- note` to this project's or the global AGENTS.md (or an existing CLAUDE.md); the system prompt is rebuilt.
- **`/memory [project|global]`** edits AGENTS.md in the `/config` editor.
- **`/context`**: estimated tokens per part of the next request with a bar, free space and the size of the last request.
- **`/doctor`** in the TUI (checks shared with `dostuff doctor`).
- **`/copy`** copies the model's last answer to the clipboard.
- **Esc Esc** (empty input, no turn running) takes back the last prompt and every step after it; the prompt goes back in the input.
- **Ctrl+R**: filterable picker over this directory's prompt history.
- **Auto-compact threshold**: `loop.auto_compact_threshold` (percent, default 50; `off` disables auto-compaction, overflow still compacts); `/compact auto <N|off>` per session; `/status` shows it.
- **Terminal title** shows the run status (`⏳` working, `✋` approval waiting, `✓` done) and is restored on exit; `ui.terminal_title: false` turns it off.
- **Notifications** (opt-in): `ui.notifications: off | bell | osc9` and `ui.notify_after_seconds` (30), when a turn finishes or an approval waits; `/notifications` per session.

### Changed
- litellm's bundled model price list is used (`LITELLM_LOCAL_MODEL_COST_MAP=True` unless already set) instead of fetching it from GitHub on import, so there are no "model cost map fetch … timed out" warnings or retry delay on slow networks. Prices for new models update with litellm upgrades.
- Ctrl+Backspace deletes the previous word on Windows (where it arrives as `^H`); Ctrl+Delete deletes the next word.
- Gemini thinking depends on the routed model, not the provider name.
- `dostuff sessions` also lists sessions running in this repo's git worktrees (`.dostuff/worktrees/`), tagged `[worktree <name>]`, so a session forked into a worktree no longer disappears from the list. `dostuff resume` with no ID still picks the latest session in the current directory only.
- TUI: select-to-copy no longer pops a large toast; a dim "copied N characters" label shows on the right of the input box's top border for 2 s.

### Fixed
- Tests: the `env` fixture in `tests/test_cli_commands.py` left the cwd in a temp dir, which broke later `python -m dostuff` tests in a full run.

### Documentation
- README: fixed the clone URL and documented the 0.2.0 and 0.2.1 commands.

## [0.2.0] - 2026-10-25

The v0.1.6 package review fixes (permissions, sandboxing, agent loop, sessions), plus web tools, settings tools, `@` mentions, cost tracking, session fork, git worktrees, skills as commands and many TUI improvements.

### Security
- **Permission layer** (`agent/permissions.py`): approval is decided before dispatch, never by the model. `_confirmed` / `overwrite` are gone from tool signatures and dropped if a model sends them. Modes `permissions.mode: ask | allowlist | auto`; prompt accepts `y` (once), `a` (always this session), anything else denies; a denial is returned to the model as the tool result.
- **Shell allow-list** replaces the bypassable `DANGEROUS_PATTERNS` denylist: a command runs without asking only if every segment is read-only and it has no redirection/substitution.
- **File sandbox**: every file tool rejects paths outside the project root (no more silent re-rooting of `/etc/passwd`); `.env*` and `~/.dostuff/` can't be read.
- **Tracing** defaults to off in `config.example.yaml`; span content is redacted unless `tracing.capture_content: true`.
- **Memory provenance**: facts carry `source` (`user_message` / `assistant`); the agent is instructed never to save facts seen only in tool output. `/memories` lists saved facts.
- **MCP OAuth**: random `state` verified, RFC 8707 `resource` sent, 120 s callback timeout, `?error=` handled, port 8085 always released, refresh uses the registered `client_id`. Optional keyring token storage (`dostuff[keyring]`).

### Added
- **Memory tools** `save_memory` (key/value fact; overlapping facts are updated via the ADD/UPDATE/NOOP resolver) and `save_episode` (DECISION / PREFERENCE / CORRECTION / EVENT / LEARNING): the agent saves memories during the session instead of an LLM extraction pass at exit. Saves don't ask for approval, are shown in the TUI, and aren't available to subagents.
- Tools: `edit_file` (exact replacement, unique-match check, `replace_all`), `grep`, `glob` (pure Python, skip `.git`/venvs/`node_modules`/`build`/`dist`); `read_file` gains `offset`/`limit` and line numbers.
- `<environment>` block in the system prompt every turn (cwd, OS, shell, date, git branch/status); restored tool-usage guidance.
- Config keys: `loop.context_window`, `loop.max_tool_output_chars` (default 30 000), `memory.episodic_scope` (`project`), `models[].api_key_env`; MCP server `connect_timeout`.
- `dostuff init --force`; `--session <unknown id>` asks before starting a new session.
- **First-run setup**: launching `dostuff` with no global config and no `MODEL` asks for provider, model and API key (hidden; saved to `~/.dostuff/.env`), writes `~/.dostuff/config.yaml` and offers a test request. `dostuff setup` re-runs it (old config kept as `config.yaml.bak`). No TTY → one-line hint only.
- **Headless mode** `dostuff -p "<prompt>"`: one turn, answer on stdout, tool activity on stderr (`-q` quiets). `--json`, piped stdin appended (`-p -` reads the prompt from stdin), `--session`, `--permission-mode`. Approvals asked on a TTY, else denied. Ctrl+C exits 130.
- **Sessions**: `dostuff resume [<id>]` (latest session in this directory), `dostuff sessions [--all] [-n N]`, `sessions rename`, `sessions rm`, `sessions prune [--older-than 30d] [--dry-run]`; ids accept a unique prefix. Session titles (schema v2). `dostuff export [<id>] [-o file|-]` / `/export` write Markdown.
- **Models**: `/model` picker (saved models, then each provider's catalogue, filterable); picking an unsaved model adds it to the global config. `/model <name> [--active]`; `dostuff models add <provider/model> [--active] [--project] [--api-key-env VAR]` and `dostuff models list`. Model lists come from litellm's catalogue, OpenRouter's public API and the local Ollama server; config edits keep comments.
- **TUI commands**: `/` command menu (filter, Tab completes, Enter runs); `/resume` picker; `/rename`; `/status`; `/permissions [mode]` (session-only, shown in the status bar); `/undo [force]` restores files from per-turn checkpoints of `write_file` / `edit_file` / `delete_file` (incl. subagents; not shell changes).
- **`!command`** runs a shell command directly from the input (no approval); command and output are saved so the model sees them next turn.
- **`todo_write` tool**: visible task list for multi-step work, rendered as a checklist.
- **Prompt caching** for Claude models (anthropic, and Claude on bedrock / vertex_ai / openrouter); cached tokens shown in the `tokens` status-bar item. `loop.prompt_caching: false` turns it off.
- `dostuff mcp list / add / remove` (stdio or `--url`; `--project` writes the project file).
- `bash_command` `timeout` parameter (default 60 s, max 110 s). Config keys `loop.max_retries` (default 6) and `loop.retry_max_delay` (default 60 s).
- Tests for schemas, permissions, subagents, file/search tools, loop robustness, DB migration, config, memory, MCP end-to-end, CLI commands and the TUI; GitHub Actions CI (pytest + mypy).
- **`web_fetch` tool**: GET a URL; HTML becomes Markdown-ish text (headings, lists, links, `<pre>`; scripts/styles dropped), JSON/XML/text as-is, binary refused; `max_chars` (default 20 000), downloads stop at 5 MB. 4xx/5xx HTML pages return only the header and a note.
- **`web_search` tool**, no API key: SearXNG if `search.searxng_url` is set, else the `ddgs` package (`dostuff[search]`), else DuckDuckGo's HTML page. Title / URL / snippet per result, `max_results` 1-20.
- New permission kind `network` for `web_fetch` / `web_search`: asks in `ask` / `allowlist`; "always" and `permissions.allow` work per domain (`web_fetch:docs.python.org`) or for every search (`web_search`). Both are in the subagents' default tools.
- **Settings tools** (`tools/settings.py`): `read_settings`, `edit_config` (exact-text edit of `config.yaml`, keeps comments, refuses invalid YAML, warns on unknown keys), `set_mcp_server` / `remove_mcp_server`; `scope` is `user` or `project`. Changes apply live (`Config.reload()`, status bar, iteration cap, MCP servers reconnected). New permission kind `settings` asks in every mode with no "always"; writes inside a `.dostuff` folder are treated the same. Not available to subagents.
- **`@path` mentions**: `@src/app.py` attaches the file, `@src` a folder listing (`@"a b.txt"` for spaces); non-paths stay plain text, `.env*` and binaries skipped. `@` opens a completion menu in the TUI; also works in `dostuff -p`.
- **Cost per turn and per session**, priced with litellm's table (cached tokens included); stored in `sessions.cost_usd` (schema v3), shown in `/status`, the turn summary and the optional `cost` status-bar item; `cost_usd` in `dostuff -p --json`.
- **TUI**: Shift+Enter (also Ctrl+Enter, Alt+Enter, Ctrl+J, `\` + Enter) inserts a newline; Up/Down prompt history saved per directory; Ctrl+O expands/collapses truncated tool results (click toggles one); select text to copy (OSC 52, plus `clip.exe` on Windows) while the input keeps focus.
- **Memory commands** `/episodes` and `/forget <key or id>`.
- **Configurable status bar** (`ui.status_bar`): it fits the terminal by shortening the cwd/model and dropping the least important items, and redraws on resize.
- `dostuff --version` (`-V`); `python -m dostuff` (and `python -m dostuff.cli`) runs the CLI.
- **`/fork [title]`** copies the current session into a new session ID and switches to it (the original is unchanged; `/resume <old id>` returns to it); `dostuff resume [id] --fork` from the shell. Copies steps, working steps and the session row (notes, tokens, cost); not undo checkpoints or todos.
- **Git worktrees** (`dostuff/lib/worktree.py`): `/worktree [name]` moves the session into `.dostuff/worktrees/<name>` on branch `dostuff/<name>`; `/worktree exit` removes it when unchanged, otherwise offers keep/remove. On quit an unchanged worktree is removed and a changed one is kept (path and branch printed). `/fork --worktree`, `dostuff --worktree` / `-w` / `--worktree-name`, `dostuff resume --worktree`; `/status` shows the worktree. Untracked local config (`.env`, `.dostuff/config.yaml`, ...) is copied in and doesn't count as a change.
- **Subagent isolation**: `delegate_to_subagent(isolation="worktree")` runs the subagent in its own worktree; unchanged → removed, changed → kept with the path, changed files and review/merge/remove commands. Tools resolve paths against a per-task project root (`helpers/tools/project_root.py`) instead of chdir-ing. The system prompt recommends it for write-capable subagents.
- **Big pastes collapse**: a paste over `ui.paste_collapse_lines` (default 10) or 2 000 chars shows as `[Pasted text #1 +243 lines]` in the input and the conversation and is expanded on submit. Over `loop.paste_file_lines` (default 200) or `loop.max_tool_output_chars` it is written to `.dostuff/pastes/<session>-<n>.txt` (with its own `.gitignore`) and the model gets a stub with a 20-line preview to page through with `read_file` / `grep`. Files older than `loop.paste_retention_days` (default 7) are removed at startup; 0 removes them on exit. `!` commands and approval notes get the text inline.
- **`/config [mcp] [project]`**: edit `config.yaml` or the MCP config (user / project) in a full-height editor in the TUI (YAML / JSON highlighting when tree-sitter is installed, line numbers). `Ctrl+S` validates, saves and applies without a restart; an invalid file stays open with the error. `Esc` closes (twice with unsaved changes). A missing `config.yaml` starts as the commented-out template.
- **Skills as slash commands**: every discovered skill is `/<name> [args]` (in the `/` menu marked `(skill)`, ghost text and `/help`); it runs a turn with the SKILL.md body plus the args. Built-in commands win a clash, and the first skill found wins among skills. Works in `dostuff -p "/<skill> ..."` too.
- **Slash-command ghost text**: the rest of the command, then its flag (`/model ` → `--active`, `/fork ` → `--worktree`), shown dim after the cursor; `Tab` or `Right` accepts it.
- **Turn summary** after each turn: `✻ Worked for 1m 5s · $0.01 · done 12:59 PM` (`Cancelled after …` on ESC; cost only when priced).
- **`↓ Jump to bottom`** bar when the conversation is scrolled up; it hides when you scroll back down or send a message.

### Changed
- **TUI runs turns, `/compact`, `/clear` and exit cleanup as async workers** on Textual's loop (no threads); the timer uses `set_interval`. Reasoning text streams into one widget per model call; streamed answers aren't duplicated; all tool calls of a resumed step are rendered.
- **Agent loop**: compaction is checked after every model call; retries only rate-limit/timeout/5xx/connection errors, with jittered exponential backoff that honours `Retry-After` (context overflow compacts then retries once; auth/bad-request fail fast); an interrupted stream is marked in the UI; an empty response gets one nudge; malformed tool-call JSON becomes a tool error; read-only tool calls run in parallel, mutating ones in order; tool output over the cap keeps head + tail; on ESC, finished tool results are kept.
- `consume_stream` keeps provider tool-call ids and `finish_reason`. `call_agent` uses `drop_params=True` and sends `stream_options` / Gemini thinking params only when they apply.
- Compaction summarises in chunks (each request < 50 % of the window) and always makes progress. Unknown models default to a 128k window with a one-time warning.
- **Session DB**: working history is stored as rows in `working_steps` (one INSERT per step; compaction flags rows). Existing DBs are migrated on first open (`PRAGMA user_version` 1); the old column is emptied, not dropped. One connection per store.
- **Memory**: episodic recall is per user (and per project directory by default) instead of per session; per-fact JSON guards; UPDATE with an unknown key becomes ADD; tz-aware timestamps; one shared Chroma client; Chroma calls run in threads.
- **MCP**: tools are keyed `server__tool`; servers connect concurrently with per-server timeouts; each connection lives in its own task (clean shutdown, no cross-task errors); `search_mcp_tools` only returns connected servers' tools and falls back to token-overlap search; tool calls re-authenticate once on HTTP 401.
- **Config**: `.env` read from `~/.dostuff/.env` then `./.dostuff/.env` (project overrides user; real env wins), loaded once; `get_config()` singleton; model read at call time; `model.api_key_env` honoured.
- Version comes from package metadata (`importlib.metadata`), used by the banner and `dostuff doctor`.
- Default identity is neutral ("DoStuff, a coding agent").
- Packaging: `mcp>=2.0`, `httpx2` declared, `setuptools>=77`; extras `keyring`, `dev`. Templates ship in the package (`dostuff/templates/`).
- **`/clear` starts a new session**; the old one stays resumable. Sessions are only created with their first step (no empty sessions per launch).
- **Approval prompts are a selector** (arrows / number / `y` `a` `n`); several calls in one response are approved together ("allow all", "review each", "no to all"). Denying takes an optional note that is passed to the model.
- **TUI look**: no message backgrounds, Claude Code-style prefixes (`>`, `●`, `⎿`), grey band on user messages, terminal background shows through, auto-scroll follows streaming output, Gemini reasoning rendered as formatted text.
- Project MCP config is merged over the global one per server; `mcp.config_path` is honoured; a broken JSON file is skipped with a message.
- `dostuff doctor` shows ✓/✗ checks with fixes and exits 1 on failure; `config` / `doctor` no longer create files.
- README quick start is `pip install dostuff` → `dostuff`.
- **chromadb is optional** (`pip install "dostuff[memory]"`, included in `full`). Without it the memory tools aren't offered, recall is skipped, and `search_mcp_tools` uses keyword search.
- **Faster startup** (~9 s → under 1 s to the first frame): litellm, chromadb and mcp load in a thread after the first frame.
- Chroma's embedding model downloads at startup with a message instead of silently during the first turn; `dostuff doctor` reports it.
- **Project instructions are combined**: global `~/.dostuff/AGENTS.md` (or CLAUDE.md), then one file per folder from the outermost parent down to the cwd.
- Memory recall leaves out facts below 0.25 similarity; the memory resolver skips the model call when nothing related exists; `save_episode` skips near-duplicates (≥ 0.85). The unused `context_window` episode field is gone.
- System prompt: a "The web" section (search before guessing URLs / packages / config values, never present placeholders as real) and guidance for adding MCP servers.
- Tool result previews are capped by characters (8 × 200) as well as lines.
- grep/glob skip `.dostuff/worktrees/`. Permissions: the "`.dostuff` in a write path → settings" rule only looks below the project root, and `.dostuff/worktrees/` in a shell command no longer counts as touching settings (`..`, nested `.dostuff` / `.env` still forbidden).
- `pyproject.toml` Homepage points at the repo root; CI also tests Python 3.14 and installs `.[dev,memory]`.
- Tool-call lines are one line, `name(key=value, …)` (values cut to 60 chars, the line to 200), instead of the full arguments (e.g. a whole `write_file` content).
- The loader shows the turn's elapsed time (`45s`, `1m 5s`, `1h 2m`, ticking once a second) and the queued-prompt count; ESC shows `Cancelling...`. The `timer` status-bar item is gone (ignored in older configs) and the default `ui.status_bar` is `[cwd, model, mode, context]`; `help`, `session`, `tokens` and `cost` can be added.
- `/model` says that a plain switch lasts for this session and `--active` makes it the default for new sessions.
- The `/` and `@` menu is drawn on its own layer over the conversation, so opening or closing it no longer repaints (and flashes) the screen.

### Fixed
- Subagents: keyword call into `loop`, `tool_names` enforced, `delete_file`/`delegate_to_subagent` never granted, valid seed message, parent adapter/cancel state passed through, `loop.subagent_max_iterations`.
- Tool schemas generated with pydantic from the real signatures (`dict` → `object`, arrays have `items`, nullable unions).
- Subagents are no longer killed by the 120 s tool timeout, and return partial results (a wrap-up or a step digest) when they stop early.
- Long-running shell commands froze the TUI on Windows: commands now run detached with no stdin, and the whole process tree is killed on timeout or ESC.
- Unrelated past episodes were recalled in every session; recall now needs a minimum similarity.
- Exit no longer waits on memory extraction.
- **MCP OAuth with confidential clients** (Supabase 422 at the token endpoint): the registration's `client_secret` and auth method are saved and sent (`client_secret_post` or HTTP Basic); cached registrations missing a needed secret are redone. Scope follows the MCP spec (`WWW-Authenticate`, else the resource's `scopes_supported`). OAuth errors include the server's message.
- `python -m dostuff.cli ...` exited 0 with no output.
- Importing the tool definitions failed on Python < 3.12: `todo_write`'s schema now uses `typing_extensions.TypedDict`.
- TUI: the input could stay unfocused after focus was cleared on slow machines (Windows CI); it is now refocused one message later instead of after a repaint.
- Slash-command ghost text looked like typed text under the `ansi-dark` theme; it is grey now.
- Settings writes turned a CRLF `config.yaml` into LF.

### Removed
- `learn_from_session.py` (dead, referenced a non-existent tool), `extract_text.py`, exit-time memory extraction (`save_memories_and_exit.py`, `extract_semantic_memories.py`, `extract_episodic_memory.py`, `format_transcript.py`) and `memory.exit_timeout_s`, `InMemorySessionStore`, `SQLiteSessionStore.save()`, `WORKSPACE_ROOT` / `SKILLS_ROOT` / `system_prompt_removed`, `random_file.txt`, `build/`, `dostuff_test.egg-info/`.

## [0.1.6] - 2026-09-21

### Added
- **Configurable max iterations:** `loop.max_iterations` (default `MAX_ITERATIONS` = 100) via `config.yaml`; `cli_tui` passes to `loop`; no hardcoded fallback.

## [0.1.5] - 2026-09-05

### Fixed
- **DB migration `last_input_tokens`:** `session_store.py` `_init_db` now runs `ALTER TABLE sessions ADD COLUMN last_input_tokens INTEGER DEFAULT 0`; fixes `OperationalError: no such column` on fresh install / resume with older DB.

## [0.1.4] - 2026-09-12

### Added

- **Streaming agent response** (`stream=True`): `_consume_stream()` accumulates `delta.content`/`delta.tool_calls`; adapter `emit("partial")`; live widget updates; `stream_options={"include_usage": True}`.
- **Usage tracking in stream mode**: `stream_options={"include_usage": True}` added; final usage-only chunk captured (`choices=[]`, `usage` non-null) after `continue` check removed.
- **Prompt queue during work**: `self._prompt_queue` + `_process_prompt_queue()`; queued prompts execute sequentially.
- **Live timer (`threading.Thread`)**: Updates status bar `⏳ {elapsed}s`; loader shows `Working...` (no freeze text, no timer embedded).
- **Loader persistence / removal**: `_freeze_final_time()` removed; loader stays mounted; messages mount `before=loader_widget`.
- **Manual `/compact` command**: `_compact_session()` loads DB truth, compacts with `0.15 * token_limit` budget (capped 20K), 60s timeout, loader, DB save (`compaction_notes` + `working_history`), and adapter sync.
- **Console trace disabled:** `ConsoleSpanExporter` removed from `tracing.py`; OTLP file trace (`~/.dostuff/data/traces/{sid}.jsonl`) planned for Jaeger.
- **Structured compaction summary:** `compact_context` prompt now requires sections (Goal, Constraints, Progress, Decisions, Next Steps, Critical Context, Turn Context); existing `compaction_notes` passed to LLM for merge/update.
- **ESC cancel confirmed done:** cancellation event wired through adapter + loop; stream interruption verified.
- **Context window percentage**: status bar shows `%` of context window filled from turn-level `prompt_tokens` / `token_limit`; persisted in DB via `last_input_tokens` and restored on resume.
- **Auto compaction**: threshold set to 50% of context window; uses turn-level prompt tokens; compacts before iteration loop.
- **Context window percentage**: status bar shows `%` of context window filled from turn-level `prompt_tokens` / `token_limit`; persisted in DB via `last_input_tokens` and restored on resume.
- **Auto compaction**: threshold set to 50% of context window; uses turn-level prompt tokens; compacts before iteration loop.
- **Token persistence**: DB `prompt_tokens`/`completion_tokens`/`total_tokens` saved per turn; `last_input_tokens` column added for context percentage.
- **Session last-used time**: `sessions.last_used` updated on every `save_session_meta`; shown in `session-list`.

### Fixed

- **Tool call accumulation in stream**: Captured `fn.name` from `fn.name` (not `arguments` fallback); `tc_key` uses `index` (primary) or `id` (fallback); instance-bound `lambda` for `model_dump()`.
- **Rate limit / usage omission**: `stream_options` added; universal provider limitation confirmed (stream chunks omit usage except final chunk).
- **Token counts hidden when 0**: `token_str` conditional; non-streamed turns show real usage, streamed turns don't until final chunk.
- **`GeneratorExit` / `BaseExceptionGroup` suppression**: Added to `_consume_stream()` for `mcp` library anyio generator cleanup during stream end.
- **MCP `streamable_http_client` crash (`RuntimeError: cancel scope`)**: Suppressed in stream loop; cleanup handled.
- **Package sync**: Source files (`cli_tui.py`, `loop.py`, `call_agent.py`, `config.py`, `extract_semantic_memories.py`, `extract_episodic_memory.py`) copied to installed `dostuff/` package after each edit.
- **`.env` lazy loading** (`Config.load()`), `--user` removal, loader persistence, live timer, queue, `ctrl+q` freeze, `call_from_thread` fix, `_fmt_tokens` consolidation already in 0.1.3; stream completes 0.1.3 work.

### Changed

- **Status bar formatting**: `📁 {cwd} • sess: {sid} • ↑{p} ↓{c}{loader}`; "new"/"resumed" keywords removed; loader no longer freezes final time.
- **Loader behavior**: `_clear_loader()` added then removed (caused abrupt app close when called from `run_worker` thread without `call_from_thread`).
- **`stream=True` architecture verified**: `await acompletion(..., stream=True)` (not raw coroutine); `types.SimpleNamespace` used (no `FakeResponse`); `_update_streaming_widget()` replaces and accumulates cumulative text.

### Removed

- `--user` flag removed (already in 0.1.3).
- `_freeze_final_time()` removed (loader no longer freezes; kept visible).

## [0.1.3] - 2026-09-05

### Added

- **`/exit` and `ctrl+q` unified**: Both now use `run_worker(action_quit())` — same fast path. Cleanup runs in a background thread with a "Saving session..." message displayed.
- **`[TOOL CALL]` / `[TOOL RESULT]` labels in history**: Historical messages now match live display format — tool calls show `[TOOL CALL] name(args)` and results show `[TOOL RESULT] result`, both with correct CSS colors (`.msg-tool` / `.msg-tool-result`).

### Changed

- **`cli.py` restructured**: No longer loads `Config()` before chdir. `session_id` is computed first, then session meta is loaded and `os.chdir()` switches to the resumed session's cwd, then `DostuffTUI` is launched. `Config()` is initialized inside the TUI only (after correct cwd).
- **`--user` flag removed**: Single-user only. `user_id` always comes from `Config().get_user_id()` (auto-creates if missing). No override per session.
- **`.env` lazy loading**: `load_dotenv()` moved from module-level to `Config.load()` — now reloads after `os.chdir()` when resuming a session from a different directory. Resolved the "d1 env applies to d2 session" issue.
- **`action_quit()` refactored**: Spawns a `cleanup_worker` (threaded) that calls `_cleanup_and_save_session()`, then schedules `self.exit()`. No blocking on the main event loop.

### Fixed

- **`TuiAdapter.emit("error")` not styled**: `"error"` event type was missing from `type_to_msg` map — fell through to default `"agent"` styling (no red). Added `"error": "error"` mapping; retry errors now show red background.
- **`ctrl+q` frozen TUI**: `ctrl+q` binding tried to `await action_quit()` directly on the event loop, blocking all input. Now uses `run_worker()` (same as `/exit`).
- **Session messages printed on mouse move**: `loop.py` had `print(msg)` fallback when adapter not set. Removed — TUI noise eliminated.
- **Loader inserted after messages**: Messages appended to bottom of stream, appearing _after_ the loader. Fixed — loader stays at bottom; all new messages mount with `before=loader_widget`.
- **`call_from_thread` in same-thread context**: Cleanup worker now uses `thread=True` — `call_from_thread` for status messages works correctly without "same thread" RuntimeError.
- **Dead code removed**: `~/.dostuff/project/.env` path removed from `.env` lookup order (never used).

### Removed

- **`--user` flag**: Removed from `cli.py` and `DostuffTUI.__init__`. Single-user only.

## [0.1.2] - 2026-09-04

### Added

- Initial PyPI release (`dostuff` 0.1.2).
- Textual TUI with ASCII banner.
- Persistent session storage (SQLite).
- MCP integration (configurable servers).
- Semantic memory (ChromaDB).
- Multi-model support via litellm.
- `dostuff init`, `dostuff config`, `dostuff doctor`, `dostuff session-list` subcommands.
- Session resume (`--session <id>`).
- 3 retries on model call failures.
- ESC cancel binding (priority=True).
- Loader widget in conversation stream.
- Tool call / tool result colored messages.

## [0.1.1] - 2026-09-04

### Added

- Python 3.11+ compatibility fixes (f-string `.get()` syntax).
- `get_session_meta_sync()` added to `SQLiteSessionStore`.
- `README.md` updated for 3.11+.
- `FUTURE.md` created with session resume + 4 pending features.

### Fixed

- Python 3.10 stale chroma → wipe `~/.dostuff/data/chroma`.
- Fixed `dostuff` 1.0.0 PyPI conflict (another publisher owns it; pip version sorting picked 1.0.0 over 0.1.1).

## [0.1.0] - 2026-09-03

### Added

- First package build (`dostuff`).
- `pyproject.toml`: `name="dostuff"`, `requires-python=">=3.11"`, `chromadb>=0.4.0,<1.0.0`.
- `cli_tui.py` skeleton with Textual TUI framework.
- `dostuff/config.py`: `Config()` with `.env` loading at import time.