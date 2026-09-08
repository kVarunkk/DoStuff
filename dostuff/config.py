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
        # Model selection: YAML supports `models` list or legacy `model` dict.
        # Env MODEL overrides YAML; pick first entry with active: true.
        active_provider = ""
        raw_models = self.raw.get("models", [])
        active_model_name = None
        if isinstance(raw_models, list):
            for entry in raw_models:
                if isinstance(entry, dict) and entry.get("active") is True:
                    active_model_name = entry.get("name", "")
                    active_provider = entry.get("provider", "")
                    break
        # Fallback: legacy single model block
        if not active_model_name:
            m = self.raw.get("model", {}) or {}
            active_model_name = m.get("name", "") if isinstance(m, dict) else ""
        # Env override
        env_model = os.environ.get("MODEL", "")
        final_model = env_model or active_model_name or "openai/gpt-4o-mini"
        # Derive provider/model (entry-level provider overrides bare names)
        if "/" in final_model:
            self.provider = active_provider or final_model.split("/")[0] or "openai"
            self.model_name = final_model
        else:
            # Derive provider from entry-level provider, or infer from name prefix
            _name = final_model or active_model_name
            prov_candidates = [active_provider] if active_provider else []
            # Try provider from entry dict explicitly
            if isinstance(raw_models, list) and active_model_name:
                for entry in raw_models:
                    if isinstance(entry, dict) and entry.get("name") == active_model_name:
                        prov_candidates.insert(0, entry.get("provider", ""))
                        break
            # Infer known prefixes
            for prefix in ("gemini", "anthropic", "openai", "groq", "mistral", "openrouter", "ollama"):
                if _name.startswith(prefix) or f"/{prefix}" in _name or f"{prefix}/" in _name:
                    prov_candidates.append(prefix)
                    break
            derived_provider = next((p for p in prov_candidates if p), "openai")
            self.provider = derived_provider
            self.model_name = f"{derived_provider}/{_name}" if not ("/" in _name) else f"{derived_provider}/{_name}"
        # API key loaded from .env (not YAML) — see setup instructions
        self.api_key_env = (self.raw.get("model", {}) or {}).get("api_key_env", self.provider.upper() + "_API_KEY")
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
