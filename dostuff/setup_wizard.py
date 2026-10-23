"""First-run setup: pick a model, store its API key, and write ~/.dostuff/config.yaml.

Runs automatically from `dostuff` when no global config exists (interactive terminals only),
and on demand as `dostuff setup`.
"""
import os
import re
import sys
from importlib import resources
from pathlib import Path

import typer

from dostuff.config import load_env_files, reset_config

# provider -> (default model, API key env var or None when no key is needed)
PROVIDERS: dict[str, tuple[str, str | None]] = {
    "openai": ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
    "anthropic": ("anthropic/claude-sonnet-5-5", "ANTHROPIC_API_KEY"),
    "gemini": ("gemini/gemini-3.1-flash-lite", "GEMINI_API_KEY"),
    "groq": ("groq/llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "mistral": ("mistral/mistral-large-latest", "MISTRAL_API_KEY"),
    "openrouter": ("openrouter/openrouter/free", "OPENROUTER_API_KEY"),
    "ollama": ("ollama/llama3.2", None),
}
CHOOSE_MODEL_PAGE = 15  # models listed at once by choose_model


def dostuff_home() -> Path:
    return Path.home() / ".dostuff"


def read_template(name: str) -> str:
    """A file shipped in dostuff/templates/."""
    return (resources.files("dostuff") / "templates" / name).read_text(encoding="utf-8")


def needs_setup() -> bool:
    """True when there is no global config and no MODEL from the environment or a .env file."""
    load_env_files()
    return not (dostuff_home() / "config.yaml").exists() and not os.environ.get("MODEL")


def can_prompt() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def mark(ok: bool) -> str:
    """✓ / ✗, or OK / X when stdout can't encode them (e.g. redirected output on Windows)."""
    symbol = "✓" if ok else "✗"
    try:
        symbol.encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        return "OK" if ok else "X"
    return symbol


def missing_api_keys(model: str) -> list[str]:
    """Env vars litellm needs for `model` that are not set; [] when none are missing or unknown."""
    try:
        import litellm
        env = litellm.validate_environment(model=model)
    except Exception:
        return []
    if env.get("keys_in_environment"):
        return []
    return list(env.get("missing_keys") or [])


def render_config(model: str, provider: str) -> str:
    """The config template with its `models` list replaced by the chosen model."""
    block = f'models:\n  - name: "{model}"\n    provider: "{provider}"\n    active: true\n\n'
    text = read_template("config.example.yaml")
    # `models:` up to the next top-level key; indented lines and blank lines belong to the block
    rendered, count = re.subn(r"(?ms)^models:\n.*?(?=^\S)", lambda _: block, text, count=1)
    return rendered if count else text.rstrip("\n") + "\n\n" + block


def set_env_var(path: Path, key: str, value: str) -> None:
    """Set KEY=value in a .env file, replacing an existing (or commented-out) line for KEY."""
    if re.search(r"[\s#'\"]", value):
        value = '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    line = f"{key}={value}"
    text = path.read_text(encoding="utf-8") if path.exists() else read_template("env.example")
    pattern = re.compile(rf"^[ \t]*#?[ \t]*{re.escape(key)}[ \t]*=.*$", re.M)
    if pattern.search(text):
        text = pattern.sub(lambda _: line, text, count=1)
    else:
        text = text.rstrip("\n") + "\n" + line + "\n"
    write_private(path, text)


def write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)


def choose_model(provider: str, default: str) -> str:
    """Numbered list of the provider's models; a number picks, other text filters or is used as typed."""
    from dostuff.helpers.models.list_models import ModelListError, provider_models
    if provider in ("ollama", "openrouter"):
        typer.echo(f"Fetching {provider} models...")
    try:
        models = provider_models(provider)
    except ModelListError as e:
        typer.echo(f"No model list: {e}.")
        models = []
    if not models:
        return typer.prompt("Model", default=default).strip()
    shown = models
    while True:
        page = shown[:CHOOSE_MODEL_PAGE]
        typer.echo(f"\n{provider} models" + (f" ({len(shown)} matching)" if shown is not models else f" ({len(models)} known)") + ":")
        for i, name in enumerate(page, 1):
            typer.echo(f"  {i:>2}. {name}")
        if len(shown) > len(page):
            typer.echo(f"  ...and {len(shown) - len(page)} more; type part of a name to narrow the list.")
        answer = typer.prompt("Model (number, name, or text to filter)", default=default).strip()
        if answer.isdigit() and 1 <= int(answer) <= len(page):
            return page[int(answer) - 1]
        full = answer if answer.startswith(provider + "/") else f"{provider}/{answer}"
        if answer == default or full in models:
            return answer if answer == default else full
        matches = [m for m in models if answer.lower() in m.lower()]
        if matches:
            shown = matches
            continue
        if typer.confirm(f"'{full}' isn't in the list. Use it anyway?", default=True):
            return full


def choose_provider() -> str:
    names = list(PROVIDERS)
    typer.echo("Which provider do you want to use?")
    for i, name in enumerate(names, 1):
        typer.echo(f"  {i}. {name}")
    while True:
        answer = typer.prompt("Provider", default="1").strip().lower()
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        if answer in PROVIDERS:
            return answer
        typer.echo(f"Enter a number from 1 to {len(names)} or a provider name.")


def _check_connection(model: str) -> None:
    typer.echo("Checking the model responds...")
    try:
        import litellm
        litellm.completion(model=model, messages=[{"role": "user", "content": "ping"}], max_tokens=1, timeout=30)
    except Exception as e:
        reason = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        typer.secho(f"{mark(False)} {reason[:300]}", fg=typer.colors.RED)
        typer.echo("  Fix the key in ~/.dostuff/.env or re-run `dostuff setup`.")
        return
    typer.secho(f"{mark(True)} Model responded.", fg=typer.colors.GREEN)


def run_setup(first_run: bool = False) -> bool:
    """Interactive setup. Returns False if the user cancelled before the config was written."""
    home = dostuff_home()
    config_path = home / "config.yaml"
    env_path = home / ".env"
    try:
        if first_run:
            typer.secho("Welcome to DoStuff! Let's pick a model (Ctrl+C to skip).\n", bold=True)
        elif config_path.exists():
            if not typer.confirm(f"{config_path} exists. Replace it? (a backup is kept)", default=False):
                typer.echo("Setup cancelled; nothing changed.")
                return False

        provider = choose_provider()
        default_model, key_var = PROVIDERS[provider]
        model = choose_model(provider, default_model)
        if "/" not in model:
            model = f"{provider}/{model}"

        if key_var is None:
            typer.echo(f"{provider} runs locally and needs no API key.")
        elif os.environ.get(key_var):
            typer.echo(f"Using {key_var} from your environment.")
        else:
            key = typer.prompt(
                f"{key_var} (input hidden, Enter to skip)", default="", show_default=False, hide_input=True
            ).strip()
            if key:
                set_env_var(env_path, key_var, key)
                os.environ[key_var] = key
                typer.echo(f"Saved {key_var} to {env_path}")
            else:
                typer.echo(f"Skipped. Add {key_var}=... to {env_path} before chatting.")
    except (typer.Abort, KeyboardInterrupt, EOFError):
        typer.echo("\nSetup skipped. Run `dostuff setup` anytime.")
        return False

    if config_path.exists():
        config_path.replace(config_path.with_name("config.yaml.bak"))
        typer.echo(f"Backed up the old config to {config_path.with_name('config.yaml.bak')}")
    home.mkdir(parents=True, exist_ok=True)
    config_path.write_text(render_config(model, provider), encoding="utf-8")
    if not env_path.exists():
        write_private(env_path, read_template("env.example"))
    reset_config()
    typer.secho(f"{mark(True)} Wrote {config_path}", fg=typer.colors.GREEN)

    if key_var is None or os.environ.get(key_var):
        try:
            if typer.confirm("Send a one-token test request?", default=True):
                _check_connection(model)
        except (typer.Abort, KeyboardInterrupt, EOFError):
            typer.echo()

    typer.echo(
        f"\nEdit {config_path} for permissions, MCP servers and limits. "
        "Run `dostuff setup` to switch models, `dostuff doctor` to check your install.\n"
    )
    return True
