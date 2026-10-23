"""User-facing message shim — routes to the TUI adapter when one is set."""
from __future__ import annotations
from typing import Any, Optional

# Set by cli_tui.py at startup; None = no UI (messages are dropped)
_active_adapter: Optional[Any] = None


def set_adapter(adapter: Any) -> None:
    """Register the active TUI adapter. Pass None to detach."""
    global _active_adapter
    _active_adapter = adapter


def emit(message: str, msg_type: str = "system") -> None:
    """Show a user-facing message in the TUI, if one is running. Safe from any thread."""
    if _active_adapter is not None and hasattr(_active_adapter, "emit"):
        try:
            _active_adapter.emit(msg_type, message)
        except Exception:
            pass
