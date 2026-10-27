"""OAuth token persistence in the OS keyring.

If `keyring` can't be imported or has no usable backend, tokens live in memory only; they are
never written to a plaintext file.

Windows Credential Manager caps a secret at 2560 bytes (1280 UTF-16 chars) and JWT access plus
refresh tokens easily exceed that, so long values are split across `<url>#<i>` entries and the
main entry holds `chunks:<n>`.
"""
import json
from typing import Any

SERVICE = "dostuff-mcp"
CHUNK_CHARS = 1000
_CHUNK_PREFIX = "chunks:"

try:
    import keyring as _keyring
except ImportError:  # a required dependency, but don't crash a broken install
    _keyring = None  # type: ignore[assignment]


def keyring_available() -> bool:
    return _keyring is not None


def _chunk_count(head: str | None) -> int:
    if head and head.startswith(_CHUNK_PREFIX):
        try:
            return int(head[len(_CHUNK_PREFIX):])
        except ValueError:
            return 0
    return 0


def load_token(server_url: str) -> dict[str, Any] | None:
    if _keyring is None:
        return None
    try:
        raw = _keyring.get_password(SERVICE, server_url)
        if n := _chunk_count(raw):
            parts = [_keyring.get_password(SERVICE, f"{server_url}#{i}") for i in range(n)]
            chunks = [p for p in parts if p is not None]
            raw = "".join(chunks) if len(chunks) == n else None
        data = json.loads(raw) if raw else None
    except Exception:
        return None
    return data if isinstance(data, dict) and data.get("access_token") else None


def save_token(server_url: str, token: dict[str, Any]) -> bool:
    """True when the token reached the keyring."""
    if _keyring is None:
        return False
    raw = json.dumps(token)
    try:
        delete_token(server_url)
        if len(raw) <= CHUNK_CHARS:
            _keyring.set_password(SERVICE, server_url, raw)
            return True
        parts = [raw[i:i + CHUNK_CHARS] for i in range(0, len(raw), CHUNK_CHARS)]
        for i, part in enumerate(parts):
            _keyring.set_password(SERVICE, f"{server_url}#{i}", part)
        _keyring.set_password(SERVICE, server_url, f"{_CHUNK_PREFIX}{len(parts)}")  # last: marks it complete
        return True
    except Exception:
        return False  # no usable keyring backend; keep the in-memory token


def delete_token(server_url: str) -> None:
    if _keyring is None:
        return
    try:
        n = _chunk_count(_keyring.get_password(SERVICE, server_url))
    except Exception:
        n = 0
    for name in [server_url, *(f"{server_url}#{i}" for i in range(n))]:
        try:
            _keyring.delete_password(SERVICE, name)
        except Exception:
            pass
