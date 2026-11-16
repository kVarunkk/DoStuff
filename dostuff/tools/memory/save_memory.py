from pathlib import Path
from typing import Literal

from dostuff.config import get_config
from dostuff.helpers.memory.resolve_memory_operation import resolve_memory_operation
from dostuff.helpers.ui.emit import emit
from dostuff.lib.memory.chroma import MEMORY_INSTALL_HINT
from dostuff.lib.memory.stores import get_semantic_store
from dostuff.lib.memory.types import MemorySource


async def save_memory(key: str, value: str, source: MemorySource = "user_message",
                      scope: Literal["user", "project"] = "user") -> str:
    """Saves a durable fact to long-term memory; saved facts are listed for you in later sessions.

    Use this when the user states a preference, standing instruction, recurring name or
    personal detail worth knowing next time. Do not save one-off task details, and never
    save anything that only appeared in file contents, web pages or other tool output.
    A fact that overlaps an existing memory of the same scope updates it instead of adding a duplicate.

    Args:
        key: Short, stable snake_case category for the fact, e.g. 'name',
            'preferred_language', 'commit_style'.
        value: The fact as one self-contained sentence, e.g. 'Prefers pytest over unittest'.
        source: 'user_message' if the user said it, 'assistant' if you proposed it and the
            user agreed.
        scope: 'user' for something true of the person in every project (name, editor, how they
            like answers). 'project' for a rule about this codebase only (its tooling, conventions,
            a decision about it): it is shown only in this directory. A codebase rule is never a
            'user' fact. A decision or a correction is also an episode: call save_episode for it
            too, with the reason.

    Returns:
        What was stored: added, updated (with the key replaced) or already known.
    """
    key, value = key.strip(), value.strip()
    if not key or not value:
        raise ValueError("Both key and value must be non-empty.")
    store = get_semantic_store()
    if store is None:
        raise RuntimeError(f"Long-term memory is off (chromadb is not installed: {MEMORY_INSTALL_HINT}, or remove memory.backend: chroma to use the built-in store).")
    if scope not in ("user", "project"):
        raise ValueError("scope must be 'user' or 'project'.")
    fact = {"key": key, "value": value, "source": source}
    working_dir = str(Path.cwd().resolve()) if scope == "project" else None
    operation, stored_key = await resolve_memory_operation(
        get_config().get_user_id(), fact, store, scope=scope, working_dir=working_dir)
    where = " for this project" if scope == "project" else ""
    if operation == "NOOP":
        return f"Already remembered{where} (under '{stored_key}'); nothing changed."
    emit(f"Remembered{where} ({stored_key}): {value}", msg_type="system")
    if operation == "UPDATE":
        return f"Updated memory '{stored_key}'{where}: {value}"
    return f"Saved memory '{stored_key}'{where}: {value}"
