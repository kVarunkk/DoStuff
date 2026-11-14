"""Which long-term memory store is in use (`memory.backend`), and whether memory is on at all. Light on import: the
system prompt and the tool list ask before the stores are loaded."""
from dostuff.lib.memory import chroma


def memory_backend() -> str:
    """`chroma` (embeddings, needs the `memory` extra), `sqlite` (FTS5 keyword search, built in) or `hybrid` (sqlite
    plus an embedding index when the extra is installed)."""
    from dostuff.config import get_config
    return str(getattr(get_config(), "memory_backend", "chroma"))


def memory_available() -> bool:
    """True if long-term memory works: sqlite and hybrid always, chroma with chromadb installed."""
    return memory_backend() in ("sqlite", "hybrid") or chroma.chroma_available()
