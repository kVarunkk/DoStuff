import os
from pathlib import Path
from dotenv import load_dotenv
from dostuff.config import Config

# Load env secrets first
for p in [Path.home() / ".dostuff" / ".env", Path(".dostuff") / ".env"]:
    if p.exists():
        load_dotenv(p)

_cfg = Config()

# Backward-compat: env MODEL wins over config; config wins over empty default
MODEL: str = os.getenv("MODEL") or _cfg.model_name or ""
# Litellm requires provider/name format (e.g. openai/gpt-4o-mini, gemini/gemini-3.1-flash-lite).
# If config.name lacks a / (old format), construct from provider + name.
if "/" not in MODEL and MODEL:
    provider_for_model = os.getenv("MODEL_PROVIDER") or (_cfg.provider or "openai")
    MODEL = f"{provider_for_model}/{MODEL}"

# Provider / API key routing for litellm
PROVIDER: str = os.getenv("MODEL_PROVIDER") or _cfg.provider or "openai"
