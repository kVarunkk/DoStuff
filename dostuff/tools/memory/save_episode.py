import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dostuff.config import get_config
from dostuff.helpers.ui.emit import emit
from dostuff.lib.memory.stores import get_episodic_store
from dostuff.lib.memory.types import EpisodeType


async def save_episode(
    event_type: EpisodeType,
    anchor_event: str,
    summary: str,
    session_id: str,
    tags: list[str] | None = None,
) -> str:
    """Records a significant moment from this session as an episodic memory, recalled in
    future sessions in this project when relevant.

    Use this right after something worth remembering next time: a technical decision, a
    correction from the user, or a failure and how it was resolved. Skip routine steps,
    greetings, plain Q&A and bare requests ("user asked for X"): save the outcome or lesson,
    not the fact that a task was asked for.

    Args:
        event_type: DECISION (a choice made), PREFERENCE (how the user wants things done),
            CORRECTION (a mistake the user caught), EVENT (something notable happened) or
            LEARNING (an error and its fix).
        anchor_event: The core moment in 1-2 sentences, e.g. 'Switched session DB to WAL
            mode to stop "database is locked" errors'.
        summary: What happened, why, and the outcome.
        tags: Key topics or technologies involved, e.g. ['sqlite', 'wal_mode'].

    Returns:
        A confirmation with the episode id.
    """
    anchor_event, summary = anchor_event.strip(), summary.strip()
    if not anchor_event or not summary:
        raise ValueError("Both anchor_event and summary must be non-empty.")
    now_epoch = int(time.time())
    episode_id = f"ep_{session_id}_{uuid.uuid4().hex[:8]}"
    episode = {
        "id": episode_id,
        "vector_text": f"- EVENT TYPE: {event_type}\n  - ANCHOR EVENT: {anchor_event}\n  - SUMMARY: {summary}",
        "metadata": {
            "user_id": get_config().get_user_id(),
            "session_id": session_id,
            "working_dir": str(Path.cwd().resolve()),
            "created_at_epoch": now_epoch,
            "created_at_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "event_type": event_type,
            "anchor_event": anchor_event,
            "summary": summary,
            "tags": json.dumps([str(t) for t in tags or []]),
            "context_window": json.dumps([]),
        },
    }
    await get_episodic_store().add([episode])
    emit(f"Remembered episode [{event_type}]: {anchor_event}", msg_type="system")
    return f"Saved episode {episode_id}."
