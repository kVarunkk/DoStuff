from pathlib import Path
from typing import Any
import yaml
import os
from dotenv import dotenv_values
from dostuff.helpers.agent.constants import MAX_ITERATIONS
from dostuff.helpers.hooks.load_hooks import HookEntry, load_merged_hooks

SUBAGENT_MAX_ITERATIONS = 25
MAX_TOOL_OUTPUT_CHARS = 30_000
MAX_RETRIES = 6
RETRY_MAX_DELAY_S = 60.0
KNOWN_PROVIDERS = ("gemini", "anthropic", "openai", "groq", "mistral", "openrouter", "ollama", "opencode-go", "opencode")
STATUS_BAR_ITEMS = ("help", "cwd", "session", "model", "mode", "tokens", "cost", "context")
DEFAULT_STATUS_BAR = ("cwd", "model", "mode", "context")
# Former items, accepted without a warning: "timer" moved to the loader message, "jobs" to an item that is always shown
RETIRED_STATUS_BAR_ITEMS = ("timer", "jobs")
PASTE_COLLAPSE_LINES = 10
PASTE_FILE_LINES = 200
PASTE_RETENTION_DAYS = 7
AUTO_COMPACT_THRESHOLD = 50  # percent of the context window
NOTIFY_KINDS = ("off", "bell", "osc9")
NOTIFY_AFTER_SECONDS = 30.0


def parse_compact_threshold(value: Any) -> int | None:
    """`loop.auto_compact_threshold`: a percent 1-100, or off/false/0 (None = no auto-compaction).
    Raises ValueError for anything else."""
    text = str(value).strip().lower().rstrip("%")
    if text in ("off", "false", "no", "none", "0"):
        return None
    percent = int(float(text))
    if not 1 <= percent <= 100:
        raise ValueError(f"auto_compact_threshold must be 1-100 or off, got {value!r}")
    return percent


def _block(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """A config block as a dict — tolerates a missing key, `key:` with no value, or a non-dict."""
    value = raw.get(key)
    return value if isinstance(value, dict) else {}


def load_env_files(cwd: Path | None = None) -> None:
    """Load `~/.dostuff/.env` then `./.dostuff/.env`; project values override user values.

    Variables already set in the real environment always win. This is the only place
    `.env` files are read.
    """
    merged: dict[str, str] = {}
    for env_path in (Path.home() / ".dostuff" / ".env", (cwd or Path.cwd()) / ".dostuff" / ".env"):
        if env_path.is_file():
            merged.update({k: v for k, v in dotenv_values(env_path).items() if v is not None})
    for key, value in merged.items():
        os.environ.setdefault(key, value)


def project_config_path() -> Path:
    """`./.dostuff/config.yaml` of the project the session started in (not a subagent's worktree)."""
    return Path.cwd() / ".dostuff" / "config.yaml"


class Config:
    def __init__(self, path: Path | None = None):
        self.global_path = Path.home() / ".dostuff" / "config.yaml"
        if path is None:
            path = self.global_path
        self.path = path
        self.raw: dict[str, Any] = {}
        self.load()
        self.data_dir = Path.home() / ".dostuff" / "data"
        self.user_id_path = Path.home() / ".dostuff" / "user_id"
        self.mcp_path = Path(_block(self.raw, "mcp").get("config_path", str(Path.home() / ".dostuff" / "mcp_config.json")))
        # Model selection: YAML supports a `models` list (first entry with active: true) or a
        # legacy `model` block. litellm wants "provider/model".
        active_entry: dict[str, Any] = {}
        raw_models = self.raw.get("models") or []
        if isinstance(raw_models, list):
            active_entry = next((e for e in raw_models if isinstance(e, dict) and e.get("active") is True), {})
        legacy = _block(self.raw, "model")
        self._yaml_model_name = str(active_entry.get("name") or legacy.get("name") or "")
        self._yaml_provider = str(active_entry.get("provider") or legacy.get("provider") or "")
        # API key: explicit `api_key_env` (models entry or model block) is read at call time
        self.api_key_env_explicit = str(active_entry.get("api_key_env") or legacy.get("api_key_env") or "")
        # Set by /model for the rest of the process; wins over env MODEL and YAML
        self._session_model: dict[str, str] | None = None
        # Set by /compact auto <N|off>: a 1-tuple so "off" (None) differs from "not set"
        self.session_compact_threshold: tuple[int | None] | None = None
        # Tracing config (env vars override config.yaml)
        t = _block(self.raw, "tracing")
        self.tracing_enabled = (
            os.environ.get("OTEL_ENABLED", str(t.get("enabled", False))).lower() in ("1", "true", "yes")
        )
        self.tracing_endpoint = os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            t.get("endpoint", "localhost:4317"),
        )
        self.tracing_service = os.environ.get(
            "OTEL_SERVICE_NAME",
            t.get("service_name", "dostuff"),
        )
        # Span attributes carrying user input / tool args / tool output are redacted unless enabled
        self.tracing_capture_content = bool(t.get("capture_content", False))
        self.tracing_exporter = os.environ.get(
            "OTEL_EXPORTER",
            t.get("exporter", "otlp"),  # otlp | none
        )
        loop_cfg = _block(self.raw, "loop")
        self.max_iterations = int(loop_cfg.get("max_iterations") or MAX_ITERATIONS)
        self.subagent_max_iterations = int(loop_cfg.get("subagent_max_iterations") or SUBAGENT_MAX_ITERATIONS)
        # Overrides litellm's model info; useful for models litellm doesn't know
        self.context_window = int(loop_cfg.get("context_window") or 0)
        self.max_tool_output_chars = int(loop_cfg.get("max_tool_output_chars") or MAX_TOOL_OUTPUT_CHARS)
        # Big TUI pastes go to .dostuff/pastes/ and the model gets a path + preview (helpers/agent/pastes.py)
        file_lines = loop_cfg.get("paste_file_lines")
        self.paste_file_lines = max(int(file_lines), 0) if file_lines is not None else PASTE_FILE_LINES
        days = loop_cfg.get("paste_retention_days")
        self.paste_retention_days = max(float(days), 0.0) if days is not None else PASTE_RETENTION_DAYS
        # Model-call retries on rate limits / timeouts / 5xx: exponential backoff (or Retry-After), capped per wait
        retries = loop_cfg.get("max_retries")
        self.max_retries = max(int(retries), 0) if retries is not None else MAX_RETRIES
        self.retry_max_delay = float(loop_cfg.get("retry_max_delay") or RETRY_MAX_DELAY_S)
        # Auto-compaction when a model call's input passes this percent of the context window (None = off)
        threshold = loop_cfg.get("auto_compact_threshold")
        try:
            self.auto_compact_threshold = AUTO_COMPACT_THRESHOLD if threshold is None else parse_compact_threshold(threshold)
        except ValueError:
            self.auto_compact_threshold = AUTO_COMPACT_THRESHOLD
        # Cache breakpoints on Claude models (see lib/model.uses_explicit_prompt_caching)
        self.prompt_caching = str(loop_cfg.get("prompt_caching", True)).lower() not in ("0", "false", "no")
        mem = _block(self.raw, "memory")
        scope = str(mem.get("episodic_scope") or "project").lower()
        self.episodic_scope = scope if scope in ("project", "user") else "project"
        # Permission layer (dostuff/agent/permissions.py): mode ask | allowlist | auto | plan
        perms = _block(self.raw, "permissions")
        mode = str(perms.get("mode", "ask")).lower()
        self.permission_mode = mode if mode in ("ask", "allowlist", "auto", "plan") else "ask"
        self.permission_allow = [str(p) for p in (perms.get("allow") or [])]
        # web_search backend: a SearXNG instance if set, else ddgs / DuckDuckGo (tools/web_search.py)
        self.searxng_url = str(_block(self.raw, "search").get("searxng_url") or "").strip()
        ui = _block(self.raw, "ui")
        # Run status in the terminal tab/window title (helpers/ui/terminal_signals.py)
        self.terminal_title = str(ui.get("terminal_title", True)).lower() not in ("0", "false", "no", "off")
        # Bell / OSC 9 when a long turn finishes or an approval waits; off unless the user turns it on
        raw_notify = ui.get("notifications", "off")
        # YAML reads on/off as booleans: `notifications: on` means the bell
        notify = "bell" if raw_notify is True else str(raw_notify).lower()
        self.notifications = notify if notify in NOTIFY_KINDS else "off"
        after = ui.get("notify_after_seconds")
        self.notify_after_seconds = max(float(after), 0.0) if after is not None else NOTIFY_AFTER_SECONDS
        collapse = ui.get("paste_collapse_lines")
        self.paste_collapse_lines = max(int(collapse), 0) if collapse is not None else PASTE_COLLAPSE_LINES
        # Start a turn when a background job ends while the TUI is idle (background shell spec D15)
        self.jobs_wake_on_end = str(_block(self.raw, "jobs").get("wake_on_end", True)).lower() not in ("0", "false", "no", "off")
        # TUI status bar: which items show, in this order; unknown names are reported once at startup
        items = _block(self.raw, "ui").get("status_bar")
        names = [str(i).strip().lower() for i in items] if isinstance(items, list) else list(DEFAULT_STATUS_BAR)
        self.status_bar_items = [n for n in names if n in STATUS_BAR_ITEMS]
        self.status_bar_unknown = [n for n in names if n not in (*STATUS_BAR_ITEMS, *RETIRED_STATUS_BAR_ITEMS)]

    def reload(self) -> None:
        """Re-read config.yaml in place (after a settings change); /model and /compact auto choices for this session stay."""
        session_model, key_env = self._session_model, self.api_key_env_explicit
        session_compact = self.session_compact_threshold
        self.__init__(self.path)  # type: ignore[misc]
        if session_model:
            self._session_model, self.api_key_env_explicit = session_model, key_env
        self.session_compact_threshold = session_compact

    @property
    def compact_threshold(self) -> int | None:
        """Auto-compaction threshold in percent (None = off): /compact auto for this session, else config."""
        if self.session_compact_threshold is not None:
            return self.session_compact_threshold[0]
        return self.auto_compact_threshold

    def configured_models(self) -> list[dict[str, Any]]:
        """The `models:` entries from config.yaml (name, provider, active, api_key_env)."""
        raw_models = self.raw.get("models") or []
        if not isinstance(raw_models, list):
            return []
        return [
            {"name": str(e["name"]), "provider": str(e.get("provider") or ""), "active": e.get("active") is True,
             "api_key_env": str(e.get("api_key_env") or "")}
            for e in raw_models if isinstance(e, dict) and e.get("name")
        ]

    def use_model(self, name: str, provider: str = "", api_key_env: str = "") -> None:
        """Switch the model for the rest of this process (/model); config.yaml is not changed."""
        self._session_model = {"name": name, "provider": provider}
        self.api_key_env_explicit = api_key_env

    def _raw_model_name(self) -> str:
        if self._session_model:
            return self._session_model["name"]
        return os.environ.get("MODEL") or self._yaml_model_name

    @property
    def model_name(self) -> str:
        """Active litellm model ("provider/model"): /model, else env MODEL, else YAML. Read at call time."""
        name = self._raw_model_name() or "openai/gpt-4o-mini"
        if "/" in name:
            return name
        return f"{self.provider}/{name}"

    @property
    def provider(self) -> str:
        """Provider: env MODEL_PROVIDER, the entry's `provider`, the model prefix, or a known name prefix."""
        if self._session_model and self._session_model["provider"]:
            return self._session_model["provider"]
        explicit = None if self._session_model else (os.environ.get("MODEL_PROVIDER") or self._yaml_provider)
        if explicit:
            return explicit
        name = self._raw_model_name()
        if "/" in name:
            return name.split("/")[0] or "openai"
        return next((p for p in KNOWN_PROVIDERS if name.startswith(p)), "openai")

    @property
    def api_key_env(self) -> str:
        if self.provider.startswith("opencode"):
            return self.api_key_env_explicit or "OPENCODE_API_KEY"  # Zen and Go share one key
        return self.api_key_env_explicit or self.provider.upper() + "_API_KEY"

    def api_key(self) -> str | None:
        """The key from `api_key_env` when one is configured explicitly; else let litellm find it."""
        if not self.api_key_env_explicit:
            return None
        return os.environ.get(self.api_key_env_explicit) or None

    def _files_signature(self) -> tuple[tuple[int, int] | None, ...]:
        """(mtime, size) of the user and project config.yaml; None for a missing file."""
        signature: list[tuple[int, int] | None] = []
        for path in (self.path, project_config_path()):
            try:
                st = path.stat()
                signature.append((st.st_mtime_ns, st.st_size))
            except OSError:
                signature.append(None)
        return tuple(signature)

    def reload_if_changed(self) -> bool:
        """Reload when a config.yaml changed on disk since the last load: edit_file, a shell command or an
        editor, not only edit_config. A file that doesn't parse (half-written, or a bad edit) keeps the
        loaded config and is retried on the next check. True when it reloaded."""
        if self._files_signature() == self._loaded_signature:
            return False
        for path in (self.path, project_config_path()):
            try:
                if path.exists():
                    yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as e:
                self.load_error = f"{path}: {e}"
                return False
        self.reload()
        return True

    def load(self):
        # Before reading, so a write that lands mid-read is seen by the next reload_if_changed()
        self._loaded_signature = self._files_signature()
        # Why the last reload_if_changed() kept the loaded config ("" = the files on disk are in use)
        self.load_error = ""
        load_env_files()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self.raw = yaml.safe_load(f)
                if not isinstance(self.raw, dict):
                    self.raw = {}
        else:
            self.raw = {}
        # Project override
        proj = project_config_path()
        # Hooks are loaded per file, not from the merged raw: a project may add hooks but never
        # replace the user's, and every entry keeps its provenance for the trust gate (hooks spec D10)
        loaded = load_merged_hooks(self.path, proj)
        self.hooks: dict[str, list[HookEntry]] = loaded.hooks
        self.hooks_project_trust = loaded.project_trust
        self.hooks_stop_block_limit = loaded.stop_block_limit
        self.hooks_warnings = loaded.warnings
        if proj.exists():
            with open(proj, encoding="utf-8") as f:
                proj_raw = yaml.safe_load(f)
                if not isinstance(proj_raw, dict):
                    proj_raw = {}
            self.raw = self._merge(self.raw, proj_raw)
        # CLI/env overrides handled at call site

    def _merge(self, base, override):
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                base[k] = self._merge(base[k], v)
            else:
                base[k] = v
        return base

    def get_user_id(self):
        if self.user_id_path.exists():
            return self.user_id_path.read_text().strip()
        uid = __import__("uuid").uuid4().hex[:16]
        self.user_id_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_id_path.write_text(uid)
        return uid

    def get_data_dir(self, mode_override=None):
        # Global-only data storage (user/session determined globally)
        dir_path = Path.home() / ".dostuff" / "data"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path


_config: Config | None = None


def get_config() -> Config:
    """Lazily created process-wide Config (read after cli.py has switched cwd)."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Drop the cached Config (tests, or after a cwd change)."""
    global _config
    _config = None
