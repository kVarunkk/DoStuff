import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dostuff.config import get_config
from dostuff.helpers.memory.same_episode import is_same_episode
from dostuff.helpers.ui.emit import emit
from dostuff.lib.memory.chroma import MEMORY_INSTALL_HINT
from dostuff.lib.memory.episodic_memory_store import DUPLICATE_EPISODE_SIMILARITY, SIMILAR_EPISODE_SIMILARITY
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
        tags: Key topics and technologies involved, plus other words someone might use when asking
            about this later (the component, the symptom, a plainer name), e.g. ['sqlite', 'wal_mode',
            'locking', 'concurrency', 'store'].

    Returns:
        A confirmation with the episode id, or the id of the same moment already saved.
    """
    anchor_event, summary = anchor_event.strip(), summary.strip()
    if not anchor_event or not summary:
        raise ValueError("Both anchor_event and summary must be non-empty.")
    store = get_episodic_store()
    if store is None:
        raise RuntimeError(f"Long-term memory is off (chromadb is not installed: {MEMORY_INSTALL_HINT}, or set memory.backend: sqlite).")
    user_id = get_config().get_user_id()
    working_dir = str(Path.cwd().resolve())
    vector_text = f"- EVENT TYPE: {event_type}\n  - ANCHOR EVENT: {anchor_event}\n  - SUMMARY: {summary}"
    # Each store's own band: a keyword score runs lower than an embedding's for the same rewording
    duplicate = float(getattr(store, "DUPLICATE", DUPLICATE_EPISODE_SIMILARITY))
    similar = float(getattr(store, "SIMILAR", SIMILAR_EPISODE_SIMILARITY))
    saved = await store.nearest(user_id, vector_text, working_dir)  # its own default floor is `similar`
    # A near-copy is the same moment; in the band below that, a reworded moment and a related one look alike to the
    # embedding, so the model decides (memory recall D10). Only between episodes of one type: a LEARNING next to the
    # EVENT it explains is the lesson the event lacked, and the judge took them for one moment (live eval S7)
    if saved and (
        saved["similarity"] >= duplicate
        or saved["similarity"] >= similar
        and str(saved.get("event_type") or "").upper() == str(event_type).upper()
        and await is_same_episode(saved, event_type, anchor_event, summary)
    ):
        return f"Already recorded as {saved['id']}; nothing saved."
    now_epoch = int(time.time())
    episode_id = f"ep_{session_id}_{uuid.uuid4().hex[:8]}"
    episode = {
        "id": episode_id,
        "vector_text": vector_text,
        "metadata": {
            "user_id": user_id,
            "session_id": session_id,
            "working_dir": working_dir,
            "created_at_epoch": now_epoch,
            "created_at_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "event_type": event_type,
            "anchor_event": anchor_event,
            "summary": summary,
            "tags": json.dumps([str(t) for t in tags or []]),
        },
    }
    await store.add([episode])
    try:
        await store.prune(user_id, working_dir)
    except Exception:
        pass  # the episode is saved; an over-full project is tidied on the next save
    emit(f"Remembered episode [{event_type}]: {anchor_event}", msg_type="system")
    return f"Saved episode {episode_id}."
