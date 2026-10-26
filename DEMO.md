# DoStuff 0.2.0 — Demo Script

A step-by-step run of show for presenting the 0.2.0 features. Plan on **25–30 minutes** for the full version, or follow only the parts marked ⭐ for a **10-minute** version.

Each step lists **Do** (what to type) and **Show** (what to point out to the audience).

---

## 0. Before the demo (do this off-screen)

1. **Fresh environment**, so first-run setup still triggers:
   ```powershell
   python -m venv demo-venv
   demo-venv\Scripts\activate
   pip install "dostuff[full]"        # includes memory (chromadb) + search
   ```
   Move your global config aside so setup runs again: rename `~/.dostuff/config.yaml` → `config.yaml.real` (put it back afterwards).
   If you'd rather skip the setup wizard on stage, leave the config in place and skip step 1.

2. **Demo project.** Create a small throwaway git repo with a few files:
   ```powershell
   mkdir C:\demo\shop; cd C:\demo\shop; git init
   ```
   Add something like `app.py` (a small Flask/CLI app with an obvious bug), `utils.py`, and a `README.md`. Commit it, so worktrees and `/undo` have a clean base.

3. **A demo skill** at `C:\demo\shop\.dostuff\skills\review\SKILL.md`:
   ```markdown
   ---
   name: review
   description: Review a file for bugs and suggest fixes
   ---
   Review the given file(s). List bugs first, then style issues. Be brief.
   ```

4. **A big paste** ready on your clipboard: ~300 lines of a log file or stack trace.

5. Terminal: dark theme, large font, wide window (the status bar shrinks gracefully but looks best wide). Pre-warm the Chroma embedding model by launching once beforehand, so the first-run download doesn't happen on stage.

---

## 1. Install & first run ⭐ (2 min)

**Do**
```powershell
dostuff --version
dostuff
```
Go through the setup wizard: pick provider → model → paste API key (hidden input) → accept the test request.

**Show**
- `dostuff --version` → `0.2.0`.
- Setup wizard asks only what it needs; the key is saved to `~/.dostuff/.env`, config to `~/.dostuff/config.yaml`.
- **Startup speed**: first frame in under a second (heavy libraries load in the background).
- Mention `dostuff setup` re-runs the wizard (old config kept as `config.yaml.bak`) and `dostuff doctor` shows ✓/✗ health checks.

---

## 2. Tour of the TUI ⭐ (3 min)

**Do**
1. Type `/` and pause. Type `mo` to filter, then `Tab`.
2. Type `/fork ` and pause on the dim ghost text (`--worktree`), then clear the line.
3. Type `/status` and press Enter.

**Show**
- The **`/` command menu**: filter as you type, `Tab` completes, drawn on its own layer so the screen doesn't flash.
- **Ghost text** completes the command name and then its flag; `Tab` or `→` accepts it.
- `/status`: session, directory, model, permission mode, tokens, context %, MCP servers.
- **Status bar** at the bottom (`cwd • model • mode • context`). Resize the window narrower to show items shortening/dropping instead of being cut off.

---

## 3. A real coding turn ⭐ (4 min)

**Do**
```
@app.py there's a bug in the total calculation — find it and fix it. Use a todo list.
```
Approve the edit with the selector (choose **allow once**).

**Show**
- **`@` mentions**: typing `@` opens a file completion menu; the file gets attached with line numbers.
- **`todo_write`**: the checklist the agent keeps while it works.
- Tool calls are **one compact line** (`edit_file(path=app.py, …)`), results under `⎿`.
- `Ctrl+O` expands / collapses truncated tool output.
- **Coloured diff** for `edit_file`.
- **Approval selector**: arrows / number / `y` `a` `n`; several calls in one response can be approved together.
- The loader shows a **live elapsed timer**; afterwards the **turn summary**: `✻ Worked for 42s · $0.01 · done 3:14 PM`.

Then:

**Do** `/undo`

**Show** the file is restored from the per-turn checkpoint. Run the fix again (or `Up` arrow to recall the prompt — **prompt history** is saved per directory) so later steps have it.

---

## 4. Permissions & safety ⭐ (3 min)

**Do**
1. `/permissions` — show the current mode, then `/permissions allowlist`.
2. Ask: `list the files here and show git status` → runs without asking (read-only commands are on the allow-list).
3. Ask: `delete utils.py` → it asks. Choose **deny** and type a note: "keep it, we still use it".
4. Ask: `read C:\Windows\System32\drivers\etc\hosts` (or `.env`).

**Show**
- Modes `ask | allowlist | auto`, changed per session and shown in the status bar.
- Approval is decided by the harness **before** the tool runs, never by the model.
- A denial **with a note** goes back to the model, which adapts.
- **File sandbox**: paths outside the project and `.env*` / `~/.dostuff/` are refused.

---

## 5. Shell, pastes and multi-line input (2 min)

**Do**
1. `!git log --oneline -5` → runs immediately; then ask `summarise those commits`.
2. Type a line, press **Shift+Enter**, type a second line (multi-line input).
3. Paste the 300-line log and type: `what's the root cause of this error?`

**Show**
- `!command` runs straight away, and the model sees the command and output next turn.
- The paste collapses to `[Pasted text #1 +300 lines]`. Because it's over 200 lines it's saved to `.dostuff/pastes/` and the model **pages through it with `read_file` / `grep`** instead of filling the context.

---

## 6. Web tools (2 min)

**Do**
```
what's new in the latest Textual release? check the web.
```
Approve the network request; pick **always** for the domain.

**Show**
- `web_search` works **without an API key** (SearXNG → `ddgs` → DuckDuckGo HTML).
- `web_fetch` turns HTML into readable Markdown-ish text.
- New permission kind **`network`**; "always" is remembered **per domain**.

---

## 7. Models & cost (2 min)

**Do**
1. `/model` → scroll the picker, open "More <provider> models…", type to filter, pick one.
2. Ask a quick question with the new model.
3. `/status` again.

**Show**
- Saved models first, then each provider's live catalogue (litellm, OpenRouter, local Ollama).
- A plain switch lasts this session; `--active` makes it the default.
- **Cost per turn and per session** (in `/status` and the turn summary). Mention **prompt caching** for Claude models — cached tokens appear in the `tokens` status-bar item.

---

## 8. Settings without leaving the TUI (2 min)

**Do**
1. `/config` → pick **config.yaml (user)**. Add `cost` and `tokens` to `ui.status_bar`. `Ctrl+S`.
2. Ask the agent: `turn on the session item in my status bar`.

**Show**
- Full-height editor with YAML highlighting and line numbers; **`Ctrl+S` validates and applies live** — the status bar changes immediately, no restart. (Optionally introduce a YAML error first to show it stays open with the error.)
- The agent can change settings itself with the **settings tools** (`edit_config`, `set_mcp_server`), and these **always ask** — a `settings` permission with no "always".
- `/config mcp project` edits MCP servers the same way.

---

## 9. Skills as commands ⭐ (1 min)

**Do**
```
/review @utils.py
```

**Show**
- Every skill is a slash command, marked `(skill)` in the `/` menu and in `/help`.
- Also works headless: `dostuff -p "/review @utils.py"`.

---

## 10. Sessions, fork & worktrees ⭐ (4 min)

**Do**
1. `/rename bugfix demo`
2. `/fork try-refactor --worktree`
3. Ask: `refactor utils.py into a class` (let it edit).
4. `/status` → shows the worktree path and branch `dostuff/…`.
5. `/worktree exit` → choose **keep**.
6. `/resume` → pick the original "bugfix demo" session.

**Show**
- `/fork` copies the conversation into a new session; the original is untouched.
- `--worktree` puts the fork in `.dostuff/worktrees/<name>` on its own branch, so experiments never touch your working tree.
- Leaving an unchanged worktree removes it; a changed one asks keep/remove.
- `/resume` picker lists sessions for this directory, with titles.

**Bonus — isolated subagents:**
```
use a subagent in an isolated worktree to add type hints to utils.py
```
When it finishes, the result lists the worktree path, the changed files and the commands to review / merge / remove.

---

## 11. Memory (2 min)

**Do**
1. `I prefer pytest over unittest, and always use type hints.`
2. `/memories` and `/episodes`
3. `/clear` (new session), then ask: `write a test for utils.py`.
4. `/forget <key>` for one fact.

**Show**
- The agent saves facts **during** the session (`save_memory`, `save_episode`) — shown in the TUI, no approval needed.
- Facts carry their **source**; nothing seen only in tool output gets saved.
- The new session recalls the preference and writes a pytest test with type hints.

---

## 12. Headless & scripting ⭐ (2 min)

Exit the TUI (`/exit` or `Ctrl+Q`), then in the shell:

```powershell
dostuff -p "what does @app.py do? one paragraph"
git diff | dostuff -p "write a commit message for this diff"
dostuff -p "count the python files here" --json
dostuff sessions
dostuff resume --fork
dostuff export -o session.md
```

**Show**
- One-shot answers on stdout, tool activity on stderr (`-q` silences it) — easy to pipe.
- Piped stdin is appended to the prompt.
- `--json` includes `cost_usd`.
- Session management from the shell: `sessions`, `resume`, `rename`, `rm`, `prune --older-than 30d --dry-run`, `export`.
- Also: `dostuff mcp add/list/remove` and `dostuff models add/list`.

---

## Wrap-up talking points

- **Safer**: permission layer decided by the harness, shell allow-list, file sandbox, redacted tracing.
- **More capable**: web search/fetch, `edit_file` / `grep` / `glob`, todo lists, isolated subagents.
- **Nicer to use**: fast startup, `/` menu with ghost text, `@` mentions, paste collapsing, live config editor, turn summaries with cost.
- **Session power tools**: fork, git worktrees, resume, export, undo.
- **Install**: `pip install dostuff` → `dostuff`.

## After the demo

- Restore `~/.dostuff/config.yaml.real` → `config.yaml`.
- Remove leftover worktrees: `git worktree list`, then `git worktree remove <path>`.
- Delete the demo repo and `demo-venv` if you don't need them.

## If something goes wrong on stage

| Problem | Fix |
| --- | --- |
| Shift+Enter submits instead of adding a newline | Use `\` then Enter, or `Ctrl+J` |
| Turn stuck / too slow | `Esc` cancels; finished tool results are kept |
| Web search returns nothing | Install the search extra: `pip install "dostuff[search]"` |
| Memory commands missing | chromadb isn't installed: `pip install "dostuff[memory]"` |
| Anything else | `dostuff doctor` |
