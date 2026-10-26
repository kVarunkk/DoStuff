import typer

app = typer.Typer()


def _lookup_session_dir(session_id: str) -> tuple[bool, str | None]:
    """(exists, working_dir) for a stored session; (False, None) if there is no DB or no row."""
    import sqlite3
    from pathlib import Path
    db_path = Path.home() / ".dostuff" / "data" / "sessions.db"
    if not db_path.is_file():
        return False, None
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute("SELECT working_dir FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return False, None
    return (row is not None), (row[0] if row else None)


PERMISSION_MODES = ("ask", "allowlist", "auto")


def _print_version(value: bool) -> None:
    if value:
        from dostuff import __version__
        typer.echo(f"dostuff {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    session: str = typer.Option(None, "--session", help="Resume a session by ID (default: start a new one)"),
    prompt: str = typer.Option(None, "--print", "-p", help="Run this prompt without the TUI, print the answer and exit ('-' reads it from stdin)"),
    output_json: bool = typer.Option(False, "--json", help="With -p: print a JSON object (result, session_id, usage)"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="With -p: no progress on stderr, only the answer"),
    permission_mode: str = typer.Option(None, "--permission-mode", help="With -p: ask | allowlist | auto (default: config)"),
    worktree: bool = typer.Option(False, "--worktree", "-w", help="Run the session in a new git worktree (.dostuff/worktrees/<name>, branch dostuff/<name>)"),
    worktree_name: str = typer.Option(None, "--worktree-name", help="Name for --worktree (implies it; default: a random one)"),
    version: bool = typer.Option(False, "--version", "-V", callback=_print_version, is_eager=True, help="Print the version and exit"),
):
    if ctx.invoked_subcommand is not None:
        return  # subcommand (init/setup/config/doctor/resume/sessions/...) — don't start agent
    if prompt is None and (output_json or quiet or permission_mode):
        typer.echo("--json, --quiet and --permission-mode only apply with -p/--print.", err=True)
        raise typer.Exit(code=2)
    if prompt is not None:
        if worktree or worktree_name:
            typer.echo("--worktree only applies to the TUI.", err=True)
            raise typer.Exit(code=2)
        _run_print_mode(session, prompt, output_json, quiet, permission_mode)
        return
    _launch(session, worktree=worktree_name or ("" if worktree else None))


def _enter_session_dir(session: str, interactive: bool) -> None:
    """Switch to a stored session's cwd BEFORE Config loads; an unknown ID asks (TUI) or fails (-p)."""
    import os
    exists, working_dir = _lookup_session_dir(session)
    if not exists:
        typer.echo(f"No saved session with ID '{session}'.", err=True)
        if not interactive or not typer.confirm("Start a new session with this ID?", default=False):
            raise typer.Exit(code=1)
    elif working_dir and os.path.isdir(working_dir):
        os.chdir(working_dir)


def _run_print_mode(session: str | None, prompt: str, output_json: bool, quiet: bool, permission_mode: str | None) -> None:
    """`dostuff -p`: one headless turn. Text piped to stdin is appended to the prompt."""
    import asyncio
    import sys
    if permission_mode is not None and permission_mode not in PERMISSION_MODES:
        typer.echo(f"--permission-mode must be one of: {', '.join(PERMISSION_MODES)}", err=True)
        raise typer.Exit(code=2)
    stdin_is_tty = sys.stdin is not None and sys.stdin.isatty()
    piped = "" if stdin_is_tty or sys.stdin is None else sys.stdin.read()
    if prompt == "-":
        prompt, piped = piped, ""
    if piped.strip():
        prompt = f"{prompt}\n\n<stdin>\n{piped.rstrip()}\n</stdin>" if prompt.strip() else piped
    if not prompt.strip():
        typer.echo("Empty prompt.", err=True)
        raise typer.Exit(code=2)
    if session:
        _enter_session_dir(session, interactive=False)
    from dostuff.headless import run_headless
    try:
        code = asyncio.run(run_headless(
            prompt, session_id=session, permission_mode=permission_mode,  # type: ignore[arg-type]
            output_json=output_json, quiet=quiet,
            # Approval prompts need a terminal on both ends; piped stdin is already consumed
            interactive=stdin_is_tty and sys.stderr.isatty(),
        ))
    except KeyboardInterrupt:
        typer.echo("Interrupted.", err=True)
        raise typer.Exit(code=130)
    raise typer.Exit(code=code)


def _enter_new_worktree(name: str) -> None:
    """Create a git worktree (random name if empty) and chdir into it, before Config loads."""
    import os
    from dostuff.lib.worktree import WorktreeError, create_worktree, detect_worktree
    current = detect_worktree()
    if current:
        typer.echo(f"Already in worktree '{current.name}' ({current.path}).", err=True)
        raise typer.Exit(code=1)
    try:
        wt = create_worktree(name or None)
    except WorktreeError as e:
        typer.echo(f"Worktree failed: {e}", err=True)
        raise typer.Exit(code=1)
    os.chdir(wt.path)
    typer.echo(f"Created worktree {wt.path} on branch {wt.branch}.")


def _launch(session: str | None, worktree: str | None = None) -> None:
    """Start the TUI: resume `session` if given, else a new session; `worktree` (a name, or "" for a
    random one) runs it in a new git worktree."""
    import uuid
    from dostuff.setup_wizard import can_prompt, needs_setup, run_setup
    session_id = session or uuid.uuid4().hex[:16]

    if session:
        _enter_session_dir(session, interactive=True)
    if worktree is not None:
        _enter_new_worktree(worktree)

    if needs_setup():
        if can_prompt():
            run_setup(first_run=True)
        else:
            typer.echo("No ~/.dostuff/config.yaml; using defaults. Run `dostuff setup` to configure.", err=True)

    typer.echo(f"Starting TUI for session: {session_id}")
    from dostuff.cli_tui import DostuffTUI
    tui_app = DostuffTUI(session_id=session_id)
    tui_app.run()
    if tui_app.exit_note:
        typer.echo(tui_app.exit_note)


@app.command()
def init(force: bool = typer.Option(False, "--force", help="Overwrite an existing .dostuff/config.yaml")):
    import pathlib
    cwd = pathlib.Path.cwd()
    proj_dir = cwd / ".dostuff"
    config_path = proj_dir / "config.yaml"
    if config_path.exists() and not force:
        typer.echo(f"{config_path} already exists. Re-run with --force to overwrite it.")
        raise typer.Exit(code=1)
    proj_dir.mkdir(exist_ok=True)
    config_template = """# Project-specific config (overrides ~/.dostuff/config.yaml)
# Uncomment any section below to override the global config.
# Full reference: https://github.com/kVarunkk/DoStuff/blob/package-structure/dostuff/templates/config.example.yaml

# mcp:
#   config_path: "mcp_config.json"  # project-specific MCP servers

# models (multi-model support; active selects which to use; env MODEL overrides):
# models:
#   - name: "gemini/gemini-3.1-flash-lite-preview"
#     provider: "gemini"
#     active: true
#   - name: "openrouter/openrouter/free"
#     provider: "openrouter"
#     active: false

# permissions:
#   mode: "ask"        # ask | allowlist | auto
#   allow: []          # shell patterns auto-approved in allowlist mode, e.g. "npm test"

# tracing:
#   enabled: false
#   exporter: "otlp"
"""
    config_path.write_text(config_template)
    (proj_dir / "skills").mkdir(exist_ok=True)
    typer.echo(f"Initialized project at {cwd}")


@app.command()
def setup():
    """Pick a model, save its API key, and write ~/.dostuff/config.yaml."""
    from dostuff.setup_wizard import run_setup
    if not run_setup():
        raise typer.Exit(code=1)


def _read_user_id(cfg) -> str:
    """The stored user ID without creating one (read-only commands shouldn't write files)."""
    return cfg.user_id_path.read_text().strip() if cfg.user_id_path.exists() else "(not created yet)"


@app.command()
def config():
    from dostuff.config import get_config
    cfg = get_config()
    typer.echo(f"Global config: {cfg.global_path}")
    typer.echo(f"Data dir: {cfg.data_dir}")
    typer.echo(f"User ID: {_read_user_id(cfg)}")
    typer.echo(f"Model: {cfg.model_name}")
    typer.echo(f"Permission mode: {cfg.permission_mode}")


@app.command()
def doctor():
    """Check config, API key, MCP and data dir; print a fix for each problem."""
    from pathlib import Path
    from dostuff import __version__
    from dostuff.config import get_config
    from dostuff.helpers.mcp.load_mcp_config import mcp_config_sources, read_mcp_file
    from dostuff.setup_wizard import mark, missing_api_keys

    cfg = get_config()
    problems = 0

    def check(ok: bool, label: str, fix: str = "") -> None:
        nonlocal problems
        typer.secho(f"  {mark(ok)} {label}", fg=typer.colors.GREEN if ok else typer.colors.RED)
        if not ok:
            problems += 1
            if fix:
                typer.echo(f"      -> {fix}")

    def info(label: str) -> None:
        typer.echo(f"  - {label}")

    typer.echo(f"dostuff {__version__}")
    check(cfg.global_path.exists(), f"Global config: {cfg.global_path}", "run `dostuff setup`")
    proj = Path.cwd() / ".dostuff" / "config.yaml"
    info(f"Project config: {proj}" + ("" if proj.exists() else " (none; `dostuff init` creates one)"))
    for env_path in (cfg.global_path.parent / ".env", Path.cwd() / ".dostuff" / ".env"):
        info(f".env: {env_path}" + ("" if env_path.exists() else " (none)"))
    model = cfg.model_name
    missing = missing_api_keys(model)
    check(not missing, f"Model: {model}" + (f" (missing {', '.join(missing)})" if missing else ""),
          f"run `dostuff setup`, or add {missing[0]}=... to {cfg.global_path.parent / '.env'}" if missing else "")
    servers: dict = {}
    for source in mcp_config_sources():
        try:
            servers.update(read_mcp_file(source))
        except Exception as e:
            check(False, f"MCP config could not be read: {source}: {e}", "fix the JSON in that file")
    info(f"MCP servers: {len(servers)} (`dostuff mcp list`)" if servers else "MCP servers: none configured (optional; `dostuff mcp add`)")
    info(f"Data dir: {cfg.data_dir}" + ("" if cfg.data_dir.exists() else " (created on first run)"))
    from dostuff.lib.memory.chroma import MEMORY_INSTALL_HINT, chroma_available, embedding_model_missing
    if not chroma_available():
        info(f"Long-term memory: off, chromadb not installed (optional; `{MEMORY_INSTALL_HINT}`)")
    else:
        info("Memory embedding model: " + ("not downloaded yet (~80 MB; the TUI fetches it on first start)"
                                           if embedding_model_missing() else "downloaded"))
    info(f"Permission mode: {cfg.permission_mode}")
    if problems:
        typer.echo(f"\n{problems} problem(s) found.")
        raise typer.Exit(code=1)
    typer.echo("\nAll good.")




# ── Sessions ───────────────────────────────────────────────────────────────────

def _with_store(fn):
    """Run `await fn(store)` against the session DB and close it."""
    import asyncio
    from dostuff.config import get_config
    from dostuff.lib.memory.session_store import SQLiteSessionStore

    async def _run():
        store = SQLiteSessionStore(db_path=str(get_config().get_data_dir() / "sessions.db"))
        try:
            return await fn(store)
        finally:
            await store.close()
    return asyncio.run(_run())


def _list_sessions(working_dir: str | None, include_empty: bool = False) -> list[dict]:
    return _with_store(lambda store: store.list_sessions(working_dir=working_dir, include_empty=include_empty))


def _fmt_last_used(last: str) -> str:
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(last).replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(last)


def _resolve_session(id_or_prefix: str) -> dict:
    """The session whose ID is (or uniquely starts with) id_or_prefix; exits with a message otherwise."""
    rows = _list_sessions(None, include_empty=True)
    exact = [r for r in rows if r["session_id"] == id_or_prefix]
    matches = exact or [r for r in rows if r["session_id"].startswith(id_or_prefix)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        typer.echo(f"No session matches '{id_or_prefix}'. `dostuff sessions --all` lists them.", err=True)
    else:
        typer.echo(f"'{id_or_prefix}' matches {len(matches)} sessions; give more characters.", err=True)
    raise typer.Exit(code=1)


def _latest_here() -> dict:
    import os
    rows = _list_sessions(os.getcwd())
    if not rows:
        typer.echo(f"No sessions in {os.getcwd()}. Run `dostuff` to start one, or `dostuff sessions --all` to see others.", err=True)
        raise typer.Exit(code=1)
    return rows[0]


@app.command()
def resume(
    session_id: str = typer.Argument(None, help="Session ID or unique prefix (default: the latest one in this directory)"),
    fork: bool = typer.Option(False, "--fork", help="Resume a copy of the session; the original is left unchanged"),
    worktree: bool = typer.Option(False, "--worktree", "-w", help="Move the session into a new git worktree (with --fork: give the fork its own checkout)"),
):
    """Resume the most recent session started in the current directory."""
    target = _resolve_session(session_id)["session_id"] if session_id else _latest_here()["session_id"]
    if fork:
        import uuid
        forked = uuid.uuid4().hex[:16]
        _with_store(lambda store: store.fork_session(target, forked))
        typer.echo(f"Forked {target} into {forked}.")
        target = forked
    _launch(target, worktree="" if worktree else None)


sessions_app = typer.Typer(help="List, rename and delete past sessions.")
app.add_typer(sessions_app, name="sessions")


def _worktree_of(working_dir: str, repo_dir: str) -> str | None:
    """Name of the worktree under repo_dir/.dostuff/worktrees that working_dir is in, if any."""
    from pathlib import Path
    from dostuff.lib.memory.session_store import _norm_dir
    from dostuff.lib.worktree import WORKTREES_DIR
    if not working_dir:
        return None
    try:
        rel = Path(_norm_dir(working_dir)).relative_to(_norm_dir(str(Path(repo_dir) / WORKTREES_DIR)))
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def _sessions_here(cwd: str) -> list[dict]:
    """Sessions in cwd plus those in its worktrees (tagged with `worktree`), newest first."""
    from dostuff.lib.memory.session_store import _norm_dir
    rows = []
    for row in _list_sessions(None):
        if _norm_dir(row["working_dir"]) == _norm_dir(cwd):
            rows.append(row)
        elif (name := _worktree_of(row["working_dir"], cwd)) is not None:
            rows.append({**row, "worktree": name})
    return rows


def _print_sessions(all_dirs: bool, limit: int) -> None:
    import os
    import shutil
    rows = _list_sessions(None) if all_dirs else _sessions_here(os.getcwd())
    if not rows:
        typer.echo("No past sessions." if all_dirs else f"No sessions in {os.getcwd()}. `dostuff sessions --all` lists every directory.")
        return
    shown = rows[:limit] if limit > 0 else rows
    if not all_dirs:
        typer.echo(f"Sessions in {os.getcwd()}:\n")
    width = shutil.get_terminal_size((120, 20)).columns
    header = f"{'SESSION':<16}  {'LAST USED':<16}  {'STEPS':>5}  "
    title_width = max(width - len(header) - 1, 20)
    typer.echo(header + ("DIRECTORY / TITLE" if all_dirs else "TITLE"))
    for row in shown:
        title = row["title"] or "(no prompt)"
        if row.get("worktree"):
            title = f"[worktree {row['worktree']}] {title}"
        if len(title) > title_width:
            title = title[: title_width - 3] + "..."
        typer.echo(f"{row['session_id']:<16}  {_fmt_last_used(row['last_used']):<16}  {row['steps']:>5}  " + (row["working_dir"] if all_dirs else title))
        if all_dirs:
            typer.echo(" " * len(header) + title)
    if len(rows) > len(shown):
        typer.echo(f"\n...and {len(rows) - len(shown)} more (use --limit 0 to show all).")
    typer.echo("\nResume with `dostuff resume` (latest here) or `dostuff resume <id>`.")


@sessions_app.callback(invoke_without_command=True)
def sessions(
    ctx: typer.Context,
    all_dirs: bool = typer.Option(False, "--all", "-a", help="Show sessions from every directory"),
    limit: int = typer.Option(20, "--limit", "-n", help="Show at most this many (0 = no limit)"),
):
    """List past sessions in the current directory, newest first."""
    if ctx.invoked_subcommand is None:
        _print_sessions(all_dirs, limit)


def _delete_sessions(session_ids: list[str]) -> None:
    """Delete sessions from the DB, plus their /undo checkpoints."""
    import shutil
    from dostuff.config import get_config

    async def _run(store):
        for sid in session_ids:
            await store.delete_session(sid)
    _with_store(_run)
    checkpoints = get_config().data_dir / "checkpoints"
    for sid in session_ids:
        shutil.rmtree(checkpoints / sid, ignore_errors=True)


@sessions_app.command("rm")
def sessions_rm(
    session_ids: list[str] = typer.Argument(..., help="Session IDs (or unique prefixes) to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation"),
):
    """Delete sessions and their /undo checkpoints."""
    rows = [_resolve_session(sid) for sid in session_ids]
    for row in rows:
        typer.echo(f"  {row['session_id']}  {_fmt_last_used(row['last_used'])}  {row['steps']} steps  {row['title'] or '(no prompt)'}")
    if not yes and not typer.confirm(f"Delete {len(rows)} session(s)? This can't be undone.", default=False):
        raise typer.Exit(code=1)
    _delete_sessions([r["session_id"] for r in rows])
    typer.echo(f"Deleted {len(rows)} session(s).")


def _parse_age(text: str):
    """'30d', '12h', '2w', '45m' -> timedelta."""
    import re
    from datetime import timedelta
    m = re.fullmatch(r"\s*(\d+)\s*([mhdw])\s*", text.lower())
    if not m:
        raise typer.BadParameter("use a number and a unit: 45m, 12h, 30d or 2w")
    n, unit = int(m.group(1)), m.group(2)
    return {"m": timedelta(minutes=n), "h": timedelta(hours=n), "d": timedelta(days=n), "w": timedelta(weeks=n)}[unit]


@sessions_app.command("prune")
def sessions_prune(
    older_than: str = typer.Option(None, "--older-than", help="Also delete sessions not used for this long, e.g. 30d, 12h, 2w"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only show what would be deleted"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation"),
):
    """Delete empty sessions (no messages), and with --older-than, old ones too."""
    from datetime import datetime
    age = _parse_age(older_than) if older_than else None
    cutoff = datetime.now() - age if age else None

    def stale(row: dict) -> bool:
        if cutoff is None:
            return False
        try:
            return datetime.fromisoformat(row["last_used"]) < cutoff
        except ValueError:
            return False

    rows = [r for r in _list_sessions(None, include_empty=True) if not r["steps"] or stale(r)]
    if not rows:
        typer.echo("Nothing to prune.")
        return
    empty = sum(1 for r in rows if not r["steps"])
    typer.echo(f"{len(rows)} session(s) to delete: {empty} empty" + (f", {len(rows) - empty} not used for {older_than}" if cutoff else "") + ".")
    if dry_run:
        for row in rows:
            typer.echo(f"  {row['session_id']}  {_fmt_last_used(row['last_used'])}  {row['steps']:>4} steps  {row['working_dir']}")
        return
    if not yes and not typer.confirm("Delete them? This can't be undone.", default=False):
        raise typer.Exit(code=1)
    _delete_sessions([r["session_id"] for r in rows])
    typer.echo(f"Deleted {len(rows)} session(s).")


@sessions_app.command("rename")
def sessions_rename(
    session_id: str = typer.Argument(..., help="Session ID or unique prefix"),
    title: list[str] = typer.Argument(None, help="New title (empty: go back to the first prompt)"),
):
    """Give a session a title, shown by `dostuff sessions`."""
    row = _resolve_session(session_id)
    text = " ".join(title or [])
    _with_store(lambda store: store.set_title(row["session_id"], text))
    typer.echo(f"Renamed {row['session_id']} to: {text}" if text else f"Cleared the title of {row['session_id']}.")


@app.command("session-list", hidden=True)
def session_list():
    """Deprecated alias for `dostuff sessions --all`."""
    _print_sessions(all_dirs=True, limit=0)


@app.command()
def export(
    session_id: str = typer.Argument(None, help="Session ID or unique prefix (default: the latest one in this directory)"),
    output: str = typer.Option(None, "--output", "-o", help="File to write (default: dostuff-session-<id>.md here; '-' for stdout)"),
):
    """Export a session as Markdown: prompts, answers, tool calls and results."""
    from pathlib import Path
    from dostuff.helpers.memory.export_session import session_to_markdown
    row = _resolve_session(session_id) if session_id else _latest_here()
    sid = row["session_id"]

    async def _load(store):
        return await store.load(sid), await store.get_session_meta(sid)
    steps, meta = _with_store(_load)
    markdown = session_to_markdown(steps, meta, sid)
    if output == "-":
        typer.echo(markdown, nl=False)
        return
    path = Path(output or f"dostuff-session-{sid}.md")
    path.write_text(markdown, encoding="utf-8")
    typer.echo(f"Exported {len(steps)} step(s) to {path}")


# ── MCP servers ────────────────────────────────────────────────────────────────

mcp_app = typer.Typer(help="List, add and remove MCP servers (mcp_config.json).", no_args_is_help=True)
app.add_typer(mcp_app, name="mcp")


def _mcp_file(project: bool):
    from dostuff.helpers.mcp.load_mcp_config import global_mcp_config_path, project_mcp_config_path
    return project_mcp_config_path() if project else global_mcp_config_path()


@mcp_app.command("list")
def mcp_list():
    """Show configured servers and the file each one comes from."""
    from dostuff.helpers.mcp.load_mcp_config import mcp_config_sources, read_mcp_file
    merged: dict = {}
    for source in mcp_config_sources():
        try:
            for name, cfg in read_mcp_file(source).items():
                merged[name] = (cfg, source)
        except Exception as e:
            typer.secho(f"Could not read {source}: {e}", fg=typer.colors.RED, err=True)
    if not merged:
        typer.echo("No MCP servers configured. Add one with `dostuff mcp add`.")
        return
    for name, (cfg, source) in merged.items():
        cfg = cfg if isinstance(cfg, dict) else {}
        transport = cfg.get("transport") or ("http" if cfg.get("url") else "stdio")
        target = cfg.get("url") or " ".join([str(cfg.get("command") or "")] + [str(a) for a in cfg.get("args") or []])
        typer.echo(f"{name:<20} {transport:<6} {target}")
        typer.echo(f"{'':<20} {'':<6} from {source}")


def _parse_pairs(items: list[str], sep: str, what: str) -> dict:
    pairs = {}
    for item in items:
        key, found, value = item.partition(sep)
        if not found or not key.strip():
            raise typer.BadParameter(f"{what} must look like KEY{sep}VALUE, got '{item}'")
        pairs[key.strip()] = value.strip() if sep == ":" else value
    return pairs


@mcp_app.command("add")
def mcp_add(
    name: str = typer.Argument(..., help="Server name (tools are called <name>__<tool>)"),
    command: list[str] = typer.Argument(None, help="stdio: the command and its arguments; put -- before them if any start with -"),
    url: str = typer.Option(None, "--url", help="HTTP server URL (instead of a command)"),
    transport: str = typer.Option(None, "--transport", "-t", help="stdio | http | sse (default: stdio, or http with --url)"),
    env: list[str] = typer.Option(None, "--env", "-e", help="KEY=VALUE for a stdio server's environment (repeatable)"),
    header: list[str] = typer.Option(None, "--header", "-H", help="'Name: value' HTTP header (repeatable)"),
    project: bool = typer.Option(False, "--project", help="Write to ./.dostuff/mcp_config.json instead of ~/.dostuff/"),
    force: bool = typer.Option(False, "--force", help="Replace a server with the same name"),
):
    """Add a server, e.g. `dostuff mcp add github -e GITHUB_TOKEN=... -- uvx mcp-server-github`."""
    from dostuff.helpers.mcp.load_mcp_config import read_mcp_file, write_mcp_file
    from dostuff.helpers.mcp.server_config import build_server_config
    try:
        cfg = build_server_config(
            command=command, url=url, transport=transport,
            env=_parse_pairs(env, "=", "--env") if env else None,
            headers=_parse_pairs(header, ":", "--header") if header else None,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    path = _mcp_file(project)
    servers = read_mcp_file(path)
    if name in servers and not force:
        typer.echo(f"'{name}' already exists in {path}. Use --force to replace it.", err=True)
        raise typer.Exit(code=1)
    servers[name] = cfg
    write_mcp_file(path, servers)
    typer.echo(f"Added '{name}' to {path}. It connects the next time dostuff starts.")


@mcp_app.command("remove")
def mcp_remove(
    name: str = typer.Argument(..., help="Server name"),
    project: bool = typer.Option(False, "--project", help="Remove from ./.dostuff/mcp_config.json instead of ~/.dostuff/"),
):
    """Remove a server from mcp_config.json."""
    from dostuff.helpers.mcp.load_mcp_config import read_mcp_file, write_mcp_file
    path = _mcp_file(project)
    servers = read_mcp_file(path)
    if name not in servers:
        other = "~/.dostuff/mcp_config.json" if project else "./.dostuff/mcp_config.json (--project)"
        typer.echo(f"No server '{name}' in {path}. Is it in {other}? `dostuff mcp list` shows where each one is.", err=True)
        raise typer.Exit(code=1)
    del servers[name]
    write_mcp_file(path, servers)
    typer.echo(f"Removed '{name}' from {path}.")


# ── Models ─────────────────────────────────────────────────────────────────────

models_app = typer.Typer(help="List and add models (the `models:` list in config.yaml).", no_args_is_help=True)
app.add_typer(models_app, name="models")


def _yaml_models(path) -> list:
    """The `models:` list of one config file ([] if the file, the key or the YAML is missing)."""
    import yaml
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, yaml.YAMLError):
        return []
    models = data.get("models") if isinstance(data, dict) else None
    return models if isinstance(models, list) else []


@models_app.command("list")
def models_list():
    """Show the configured models; * marks the one in use."""
    import os
    from pathlib import Path
    from dostuff.config import get_config
    cfg = get_config()
    proj = Path.cwd() / ".dostuff" / "config.yaml"
    models = cfg.configured_models()
    if not models:
        typer.echo(f"No models configured; using {cfg.model_name}. Add one with `dostuff models add <provider/model>`.")
        return
    current = cfg.model_name
    for m in models:
        mark = "*" if m["name"] in (current, current.split("/", 1)[-1]) else " "
        typer.echo(f"{mark} {m['name']:<40} {m['provider']:<12} {'default' if m['active'] else ''}")
    typer.echo(f"\nfrom {proj if _yaml_models(proj) else cfg.global_path}")
    import os
    if os.environ.get("MODEL"):
        typer.echo(f"MODEL={os.environ['MODEL']} is set and overrides the default.")


@models_app.command("add")
def models_add(
    name: str = typer.Argument(None, help="litellm model name, provider/model (e.g. anthropic/claude-sonnet-5-5); omit to pick from a list"),
    provider: str = typer.Option(None, "--provider", "-P", help="Provider (default: the part before the first /)"),
    api_key_env: str = typer.Option(None, "--api-key-env", help="Read the API key from this env var instead of <PROVIDER>_API_KEY"),
    active: bool = typer.Option(False, "--active", help="Make it the default model (also works on a model already listed)"),
    project: bool = typer.Option(False, "--project", help="Write to ./.dostuff/config.yaml instead of ~/.dostuff/"),
):
    """Add a model to config.yaml, e.g. `dostuff models add anthropic/claude-sonnet-5-5 --active`; no name: pick one."""
    import os
    from pathlib import Path
    from dostuff.config import load_env_files
    from dostuff.helpers.models.add_model_entry import ModelConfigError, add_model_entry
    from dostuff.setup_wizard import PROVIDERS, can_prompt, choose_model, choose_provider, missing_api_keys, read_template

    if name is None:
        if not can_prompt():
            raise typer.BadParameter("give a model name (provider/model); the picker needs a terminal")
        if provider and provider not in PROVIDERS:
            raise typer.BadParameter(f"no model list for '{provider}'; give the model name instead")
        provider = provider or choose_provider()
        name = choose_model(provider, PROVIDERS[provider][0])
    name = name.strip()
    if not name or any(c.isspace() for c in name):
        raise typer.BadParameter("the model name can't be empty or contain spaces")
    if "/" not in name:
        if not provider:
            raise typer.BadParameter("use provider/model (e.g. openai/gpt-4o), or pass --provider")
        name = f"{provider}/{name}"
    provider = provider or name.split("/", 1)[0]
    global_path = Path.home() / ".dostuff" / "config.yaml"
    proj_path = Path.cwd() / ".dostuff" / "config.yaml"
    path = proj_path if project else global_path
    try:
        is_new = add_model_entry(path, name, provider, api_key_env or "", active,
                                 template="" if project else read_template("config.example.yaml"))
    except ModelConfigError as e:
        hint = " Use --active to make it the default." if "already in" in str(e) else ""
        typer.echo(f"{e}.{hint}", err=True)
        raise typer.Exit(code=1)
    now_default = any(isinstance(m, dict) and m.get("name") == name and m.get("active") is True for m in _yaml_models(path))
    typer.echo((f"Added '{name}' to {path}." if is_new else f"'{name}' was already in {path}.")
               + (" It is now the default model." if now_default else " Switch to it with /model in the TUI."))

    if project and _yaml_models(global_path):
        typer.echo("Note: in this directory the project `models:` list replaces the one in ~/.dostuff/config.yaml.")
    elif not project and _yaml_models(proj_path):
        typer.echo(f"Note: {proj_path} has its own `models:` list, which replaces this one in this directory.")
    load_env_files()
    if now_default and os.environ.get("MODEL"):
        typer.echo(f"Note: MODEL={os.environ['MODEL']} is set and overrides the default.")
    missing = ([api_key_env] if not os.environ.get(api_key_env) else []) if api_key_env else missing_api_keys(name)
    if missing:
        typer.echo(f"Its API key isn't set: add {missing[0]}=... to {global_path.parent / '.env'}.")


# `python -m dostuff.cli ...`: without this it defines the app and exits silently with status 0
if __name__ == "__main__":
    app()
