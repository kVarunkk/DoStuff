"""User-facing print shim — routes to TUI adapter if set, else stdout."""
from __future__ import annotations
import sys
from typing import Any, Optional

# Set by cli_tui.py at startup; None = plain terminal mode
_active_adapter: Optional[Any] = None


def set_adapter(adapter: Any) -> None:
    """Register the active TUI adapter. Pass None to fall back to stdout."""
    global _active_adapter
    _active_adapter = adapter


def emit(message: str, msg_type: str = "system") -> None:
    """Print a user-facing message. Routes to TUI if available, else stdout."""
    if _active_adapter is not None and hasattr(_active_adapter, "emit"):
        try:
            _active_adapter.emit(msg_type, message)
            return
        except Exception:
            pass  # fall through to stdout
    # print(message, file=sys.stderr if msg_type == "error" else sys.stdout)
