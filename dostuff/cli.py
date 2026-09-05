import typer
import sys

app = typer.Typer()

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context = typer.Option(None),
    session: str = typer.Option(None, "--session", help="Session ID (default: new UUID)"),
):
    if ctx.invoked_subcommand is not None:
        return  # subcommand (init/config/doctor/session_list/resume) — don't start agent

    # Compute session_id and switch cwd before TUI init (Config loaded inside TUI)
    import sqlite3, uuid
    from pathlib import Path
    session_id = session or (sys.argv[0] + "_" + uuid.uuid4().hex[:8])
    if session is None and session_id.startswith(sys.argv[0]):
        session_id = uuid.uuid4().hex[:16]

    # Switch to session's saved cwd BEFORE TUI starts
    try:
        _home_db = str(Path.home() / ".dostuff" / "data" / "sessions.db")
        conn = sqlite3.connect(_home_db)
        cur = conn.execute("SELECT working_dir FROM sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            import os
            os.chdir(row[0])
    except Exception:
        pass

    # User ID and data dir handled by Config().get_user_id() inside TUI
    typer.echo(f"Starting TUI for session: {session_id}")
    from dostuff.cli_tui import DostuffTUI
    tui_app = DostuffTUI(session_id=session_id)
    tui_app.run()

@app.command()
def init():
    import pathlib
    cwd = pathlib.Path.cwd()
    proj_dir = cwd / ".dostuff"
    proj_dir.mkdir(exist_ok=True)
    config_template = """# Project-specific config (overrides ~/.dostuff/config.yaml)
# Uncomment any section below to override the global config.
# Full reference: https://github.com/kVarunkk/DoStuff/blob/package-structure/config.example.yaml

# mcp:
#   config_path: "mcp_config.json"  # project-specific MCP servers

# model:
#   name: "openai/gpt-4o-mini"  # litellm format
#   api_key_env: "OPENAI_API_KEY"  # env var with key (never put key in YAML)

# tracing:
#   enabled: false
#   exporter: "otlp"
"""
    (proj_dir / "config.yaml").write_text(config_template)
    (proj_dir / "skills").mkdir(exist_ok=True)
    typer.echo(f"Initialized project at {cwd}")

@app.command()
def config():
    from dostuff.config import Config
    cfg = Config()
    typer.echo(f"Global config: {cfg.global_path}")
    typer.echo(f"Data dir: {cfg.get_data_dir()}")
    typer.echo(f"User ID: {cfg.get_user_id()}")

@app.command()
def doctor():
    typer.echo("Health check: dostuff package installed.")
    from dostuff.config import Config
    cfg = Config()
    typer.echo(f"  Global config: {cfg.global_path} (exists={cfg.global_path.exists()})")
    cwd = __import__('pathlib').Path.cwd()
    proj = cwd / ".dostuff" / "config.yaml"
    typer.echo(f"  Project config: {proj} (exists={proj.exists()})")
    typer.echo(f"  User ID file: {cfg.user_id_path} (exists={cfg.user_id_path.exists()})")
    typer.echo(f"  Data dir: {cfg.get_data_dir()} (exists={cfg.get_data_dir().exists()})")

@app.command()
def session_list():
    import asyncio
    from dostuff.memory import SQLiteSessionStore
    from dostuff.config import Config

    async def _run():
        cfg = Config()
        store = SQLiteSessionStore(db_path=str(cfg.get_data_dir() / "sessions.db"))
        rows = await store.list()
        if not rows:
            typer.echo("No past sessions.")
            return
        typer.echo(f"{'SESSION':<36} | {'WORKING_DIR':<60}")
        typer.echo("-" * 100)
        for sid, wd in rows:
            typer.echo(f"{sid:<36} | {str(wd):<60}")
    asyncio.run(_run())
