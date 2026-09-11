from dostuff.helpers.agent.append_step import append_step

async def loop_check_cancelled(steps_history, working_history, current_session_history, session_id, store, turn_type, adapter = None) -> bool:
    if adapter and hasattr(adapter, "_cancel_event"):
        if adapter._cancel_event.is_set() or adapter._cancelled:
            adapter._cancelled = False
            adapter._cancel_event.clear()
            cancel_step = {"role": "assistant", "content": "[Cancelled by user before responding]"}
            await append_step(cancel_step, steps_history, working_history, current_session_history, session_id, store, turn_type)
            # await loop_emit("system", "Turn cancelled by user (ESC).", adapter=adapter)
            return True
    return False