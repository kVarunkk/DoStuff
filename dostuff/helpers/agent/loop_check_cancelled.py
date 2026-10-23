from typing import Any

from dostuff.helpers.agent.append_step import append_step


def is_cancelled(adapter: Any) -> bool:
    """True if the user pressed ESC (adapter cancel flag or event)."""
    if adapter is None or not hasattr(adapter, "_cancel_event"):
        return False
    return bool(adapter._cancelled or adapter._cancel_event.is_set())


async def loop_check_cancelled(steps_history, working_history, current_session_history, session_id, store, turn_type, adapter = None) -> bool:
    if is_cancelled(adapter):
        adapter._cancelled = False
        adapter._cancel_event.clear()
        cancel_step = {"role": "assistant", "content": "[Cancelled by user before responding]"}
        await append_step(cancel_step, steps_history, working_history, current_session_history, session_id, store, turn_type)
        return True
    return False
