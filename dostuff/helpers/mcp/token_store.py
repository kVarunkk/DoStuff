"""Optional OAuth token persistence in the OS keyring (`pip install dostuff[keyring]`).

Without `keyring`, tokens live in memory only; they are never written to a plaintext file.
"""
import json
from typing import Any

SERVICE = "dostuff-mcp"

try:
    import keyring as _keyring
except ImportError:  # optional dependency
    _keyring = None


def keyring_available() -> bool:
    return _keyring is not None


def load_token(server_url: str) -> dict[str, Any] | None:
    if _keyring is None:
        return None
    try:
        raw = _keyring.get_password(SERVICE, server_url)
        data = json.loads(raw) if raw else None
    except Exception:
        return None
    return data if isinstance(data, dict) and data.get("access_token") else None


def save_token(server_url: str, token: dict[str, Any]) -> None:
    if _keyring is None:
        return
    try:
        _keyring.set_password(SERVICE, server_url, json.dumps(token))
    except Exception:
        pass  # no usable keyring backend; keep the in-memory token


def delete_token(server_url: str) -> None:
    if _keyring is None:
        return
    try:
        _keyring.delete_password(SERVICE, server_url)
    except Exception:
        pass
