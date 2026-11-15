"""Which long-term memory store is in use (`memory.backend`), and whether memory is on at all. Light on import: the
system prompt and the tool list ask before the stores are loaded."""
from dostuff.lib.memory import chroma


DEFAULT_BACKEND = "hybrid"


def memory_backend() -> str:
    """`hybrid` (the default: sqlite plus an embedding index when the `memory` extra is installed), `sqlite` (FTS5
    keyword search, built in) or `chroma` (embeddings only, needs the extra; the default before hybrid)."""
    from dostuff.config import get_config
    return str(getattr(get_config(), "memory_backend", DEFAULT_BACKEND))


def memory_available() -> bool:
    """True if long-term memory works: sqlite and hybrid always, chroma with chromadb installed."""
    return memory_backend() in ("sqlite", "hybrid") or chroma.chroma_available()
