"""One semantic and one episodic store per data dir, shared by the TUI (recall) and the memory tools (save)."""
from functools import lru_cache

from dostuff.config import get_config
from dostuff.lib.memory.episodic_memory_store import EpisodicMemoryStore
from dostuff.lib.memory.semantic_memory_store import SemanticMemoryStore


def _persist_path() -> str:
    return str(get_config().get_data_dir() / "chroma")


@lru_cache(maxsize=None)
def _semantic(persist_path: str) -> SemanticMemoryStore:
    return SemanticMemoryStore(persist_path=persist_path)


@lru_cache(maxsize=None)
def _episodic(persist_path: str) -> EpisodicMemoryStore:
    return EpisodicMemoryStore(persist_path=persist_path)


def get_semantic_store() -> SemanticMemoryStore:
    return _semantic(_persist_path())


def get_episodic_store() -> EpisodicMemoryStore:
    return _episodic(_persist_path())
