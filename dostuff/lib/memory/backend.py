"""Which long-term memory store is in use (`memory.backend`), and whether memory is on at all. Light on import: the
system prompt and the tool list ask before the stores are loaded."""
from dostuff.lib.memory import chroma


def memory_backend() -> str:
    """`chroma` (embeddings, needs the `memory` extra) or `sqlite` (FTS5 keyword search, built in)."""
    from dostuff.config import get_config
    return str(getattr(get_config(), "memory_backend", "chroma"))


def memory_available() -> bool:
    """True if long-term memory works: the sqlite backend always, the chroma one with chromadb installed."""
    return memory_backend() == "sqlite" or chroma.chroma_available()
