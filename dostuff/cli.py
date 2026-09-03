import typer
import sys

app = typer.Typer()

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context = typer.Option(None),
    session: str = typer.Option(None, "--session", help="Session ID (default: new UUID)"),
    user: str = typer.Option(None, "--user", help="User ID (default: from ~/.dostuff/user_id)"),
):
    if ctx.invoked_subcommand is not None:
        return  # subcommand (init/config/doctor/session_list/resume) — don't start agent
    from dostuff.config import Config

    config = Config()
    user_id = user or config.get_user_id()
    session_id = session or (sys.argv[0] + "_" + __import__("uuid").uuid4().hex[:8])
    if session is None and session_id.startswith(sys.argv[0]):
        session_id = __import__("uuid").uuid4().hex[:16]
    typer.echo(f"USER: {user_id}  SESSION: {session_id}")
    typer.echo(f"DATA_DIR: {config.get_data_dir()}")
    typer.echo("Starting TUI...")
    typer.echo("STARTING DoStuff...")
    # Real TUI — default experience (not skeleton)
    from dostuff.cli_tui import DostuffTUI
    tui_app = DostuffTUI(session_id=session_id, user_id=user_id)
    tui_app.run()

@app.command()
def init():
    import pathlib
    cwd = pathlib.Path.cwd()
    proj_dir = cwd / ".dostuff"
    proj_dir.mkdir(exist_ok=True)
    
    # Create project config.yaml with template
    config_template = """# Project-specific config (overrides ~/.dostuff/config.yaml)
# Uncomment any section below to override the global config.
# Full reference: https://github.com/<you>/dostuff/blob/main/config.example.yaml

# data:
#   mode: "project"  # use project data dir instead of global

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
    
    # Create directories
    (proj_dir / "skills").mkdir(exist_ok=True)
    
    typer.echo(f"Initialized project at {cwd}")

@app.command()
def config():
    from dostuff.config import Config

    cfg = Config()
    typer.echo(f"Global config: {cfg.global_path}")
    # typer.echo(f"Data mode: {cfg.raw.get('data',{}).get('mode','global')}")
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


