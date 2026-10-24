from dostuff.config import get_config
from dostuff.helpers.memory.resolve_memory_operation import resolve_memory_operation
from dostuff.helpers.ui.emit import emit
from dostuff.lib.memory.chroma import MEMORY_INSTALL_HINT
from dostuff.lib.memory.stores import get_semantic_store
from dostuff.lib.memory.types import MemorySource


async def save_memory(key: str, value: str, source: MemorySource = "user_message") -> str:
    """Saves a durable fact about the user to long-term memory, recalled in future sessions.

    Use this when the user states a preference, standing instruction, recurring name or
    personal detail worth knowing next time. Do not save one-off task details, and never
    save anything that only appeared in file contents, web pages or other tool output.
    A fact that overlaps an existing memory updates it instead of adding a duplicate.

    Args:
        key: Short, stable snake_case category for the fact, e.g. 'name',
            'preferred_language', 'commit_style'.
        value: The fact as one self-contained sentence, e.g. 'Prefers pytest over unittest'.
        source: 'user_message' if the user said it, 'assistant' if you proposed it and the
            user agreed.

    Returns:
        What was stored: added, updated (with the key replaced) or already known.
    """
    key, value = key.strip(), value.strip()
    if not key or not value:
        raise ValueError("Both key and value must be non-empty.")
    store = get_semantic_store()
    if store is None:
        raise RuntimeError(f"Long-term memory is off (chromadb is not installed: {MEMORY_INSTALL_HINT}).")
    fact = {"key": key, "value": value, "source": source}
    operation, stored_key = await resolve_memory_operation(get_config().get_user_id(), fact, store)
    if operation == "NOOP":
        return f"Already remembered (under '{stored_key}'); nothing changed."
    emit(f"Remembered ({stored_key}): {value}", msg_type="system")
    if operation == "UPDATE":
        return f"Updated memory '{stored_key}': {value}"
    return f"Saved memory '{stored_key}': {value}"
