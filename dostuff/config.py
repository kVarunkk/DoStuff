from pathlib import Path
import yaml
import os
from dotenv import load_dotenv

# Module-level .env loading removed — env files are now loaded lazily in Config.load()
# so they can be re-read after a cwd change (e.g. resuming a session from a different directory).

class Config:
    def __init__(self, path: Path | None = None):
        self.global_path = Path.home() / ".dostuff" / "config.yaml"
        if path is None:
            path = self.global_path
        self.path = path
        self.raw = {}
        self.load()
        self.data_dir = Path((self.raw.get("data", {}) or {}).get("global_dir", str(Path.home() / ".dostuff" / "data")))
        self.user_id_path = Path.home() / ".dostuff" / "user_id"
        self.mcp_path = Path((self.raw.get("mcp", {}) or {}).get("config_path", str(Path.home() / ".dostuff" / "mcp_config.json")))
        # Model / provider selection (env overrides YAML for secrets)
        # Litellm format: model.name can be "provider/model" (e.g. gemini/gemini-3.1-flash-lite)
        m = self.raw.get("model", {}) or {}
        _name_raw = m.get("name", "")
        if "/" in _name_raw:
            self.provider = os.environ.get("MODEL_PROVIDER") or _name_raw.split("/")[0] or "openai"
            self.model_name = os.environ.get("MODEL") or _name_raw or "openai/gpt-4o-mini"
        else:
            self.provider = os.environ.get("MODEL_PROVIDER") or m.get("provider", "openai")
            self.model_name = os.environ.get("MODEL") or _name_raw or "gpt-4o-mini"
        # API key loaded from .env (not YAML) — see setup instructions
        self.api_key_env = m.get("api_key_env", self.provider.upper() + "_API_KEY")
        # Tracing config (env vars override config.yaml)
        t = self.raw.get("tracing", {}) or {}
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
        self.tracing_exporter = os.environ.get(
            "OTEL_EXPORTER",
            t.get("exporter", "otlp"),  # otlp | console | none
        )

    def load(self):
        # Lazy .env loading — supports session resume from different cwd
        _dostuff_home = Path.home() / ".dostuff"
        for _env_path in [_dostuff_home / ".env", Path.cwd() / ".dostuff" / ".env"]:
            if _env_path.exists():
                load_dotenv(_env_path)
                break
        if self.path.exists():
            with open(self.path) as f:
                self.raw = yaml.safe_load(f)
                if not isinstance(self.raw, dict):
                    self.raw = {}
        else:
            self.raw = {}
        # Project override
        cwd = Path.cwd()
        proj = cwd / ".dostuff" / "config.yaml"
        if proj.exists():
            with open(proj) as f:
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
