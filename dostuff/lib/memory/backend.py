"""Which long-term memory store is in use (`memory.backend`), and whether memory is on at all. Light on import: the
system prompt and the tool list ask before the stores are loaded."""
import sqlite3
from functools import lru_cache

from dostuff.lib.memory import chroma


DEFAULT_BACKEND = "hybrid"
# memory.db's keyword index tokenizer (sqlite_memory_store builds every FTS5 table with it)
FTS5_TOKENIZE = "porter unicode61"


@lru_cache(maxsize=1)
def fts5_available() -> bool:
    """True if this Python's sqlite3 was built with FTS5 (almost every build; memory.db needs it)."""
    try:
        sqlite3.connect(":memory:").execute(f"CREATE VIRTUAL TABLE t USING fts5(a, tokenize='{FTS5_TOKENIZE}')")
        return True
    except sqlite3.Error:
        return False


def configured_backend() -> str:
    """`memory.backend` as configured: hybrid, sqlite or chroma."""
    from dostuff.config import get_config
    return str(getattr(get_config(), "memory_backend", DEFAULT_BACKEND))


def memory_backend() -> str:
    """The backend in use: `hybrid` (the default: sqlite plus an embedding index when the `memory` extra is
    installed), `sqlite` (FTS5 keyword search, built in) or `chroma` (embeddings only, needs the extra; the default
    before hybrid). hybrid and sqlite need FTS5; on a Python built without it they fall back to chroma, as before
    hybrid existed, so memory works with the extra and is off without it, instead of failing every start."""
    backend = configured_backend()
    if backend in ("sqlite", "hybrid") and not fts5_available():
        return "chroma"
    return backend


def memory_available() -> bool:
    """True if long-term memory works: sqlite and hybrid always (given FTS5), chroma with chromadb installed."""
    return memory_backend() in ("sqlite", "hybrid") or chroma.chroma_available()


def fallback_notice() -> str:
    """What to tell the user when the configured backend can't run here, or "" when it can."""
    if memory_backend() == configured_backend():
        return ""
    if chroma.chroma_available():
        return (f"Long-term memory uses the older chroma store: memory.backend is {configured_backend()}, which "
                "needs SQLite's FTS5, and this Python's sqlite3 was built without it.")
    return (f"Long-term memory is off: memory.backend is {configured_backend()}, which needs SQLite's FTS5, and this "
            f"Python's sqlite3 was built without it. Use a Python whose sqlite3 has FTS5, or run "
            f"`{chroma.MEMORY_INSTALL_HINT}` for the older chroma store.")
