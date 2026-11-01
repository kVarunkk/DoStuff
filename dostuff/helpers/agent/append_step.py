from dostuff.lib.memory.session_store import  SessionStore
from typing import Any, Literal

async def append_step(step: dict, steps_history: list, working_history: list, current_session_history: list, session_id: str, store: SessionStore | Any | None, turn_type: Literal['interactive_loop', 'subagent_loop']) -> None:
    """Add a step to the working history and store it. A subagent's `store` is its transcript
    (`AgentTranscript`, subagents spec D14): its steps never reach the session's own history."""
    working_history.append(step)

    if turn_type == 'interactive_loop':
        steps_history.append(step)
        current_session_history.append(step)
    if store is not None:
        await store.append(session_id, step)
