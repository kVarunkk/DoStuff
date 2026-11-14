"""One semantic and one episodic store per data dir, shared by the TUI (recall) and the memory tools (save).

Each getter returns None when chromadb isn't installed: long-term memory is then off.
"""
from functools import lru_cache
from typing import TYPE_CHECKING

from dostuff.config import get_config
from dostuff.lib.memory import chroma
from dostuff.lib.memory.backend import memory_backend
from dostuff.lib.memory.episodic_memory_store import EpisodicMemoryStore
from dostuff.lib.memory.semantic_memory_store import SemanticMemoryStore

if TYPE_CHECKING:
    from dostuff.lib.mcp.mcp_tool_registry_store import MCPToolRegistryStore
    from dostuff.lib.memory.hybrid_memory_store import HybridEpisodeStore, HybridFactStore
    from dostuff.lib.memory.sqlite_memory_store import SQLiteEpisodeStore, SQLiteFactStore


def _persist_path() -> str:
    return str(get_config().get_data_dir() / "chroma")


@lru_cache(maxsize=None)
def _semantic(persist_path: str) -> SemanticMemoryStore:
    return SemanticMemoryStore(persist_path=persist_path)


@lru_cache(maxsize=None)
def _episodic(persist_path: str) -> EpisodicMemoryStore:
    return EpisodicMemoryStore(persist_path=persist_path)


@lru_cache(maxsize=None)
def _sqlite_facts(db_path: str) -> "SQLiteFactStore":
    from dostuff.lib.memory.sqlite_memory_store import SQLiteFactStore
    return SQLiteFactStore(db_path)


@lru_cache(maxsize=None)
def _sqlite_episodes(db_path: str) -> "SQLiteEpisodeStore":
    from dostuff.lib.memory.sqlite_memory_store import SQLiteEpisodeStore
    return SQLiteEpisodeStore(db_path)


@lru_cache(maxsize=None)
def _hybrid_facts(db_path: str, index_path: str) -> "HybridFactStore":
    from dostuff.lib.memory.hybrid_memory_store import HybridFactStore
    return HybridFactStore(db_path, index_path)


@lru_cache(maxsize=None)
def _hybrid_episodes(db_path: str, index_path: str) -> "HybridEpisodeStore":
    from dostuff.lib.memory.hybrid_memory_store import HybridEpisodeStore
    return HybridEpisodeStore(db_path, index_path)


def _index_path() -> str:
    """The hybrid backend's embedding index: its own Chroma dir, so a chroma-backend store beside it is never mixed in."""
    return str(get_config().get_data_dir() / "memory-index")


def _keywords_only() -> bool:
    """sqlite, or hybrid without chromadb (then it is sqlite: same memory.db, no embedding side)."""
    backend = memory_backend()
    return backend == "sqlite" or (backend == "hybrid" and not chroma.chroma_available())


def _sqlite_path() -> str:
    data_dir = get_config().get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / "memory.db")


def get_semantic_store() -> "SemanticMemoryStore | SQLiteFactStore | HybridFactStore | None":
    if _keywords_only():
        return _sqlite_facts(_sqlite_path())
    if memory_backend() == "hybrid":
        return _hybrid_facts(_sqlite_path(), _index_path())
    return _semantic(_persist_path()) if chroma.chroma_available() else None


def get_episodic_store() -> "EpisodicMemoryStore | SQLiteEpisodeStore | HybridEpisodeStore | None":
    if _keywords_only():
        return _sqlite_episodes(_sqlite_path())
    if memory_backend() == "hybrid":
        return _hybrid_episodes(_sqlite_path(), _index_path())
    return _episodic(_persist_path()) if chroma.chroma_available() else None


def make_tool_registry_store() -> "MCPToolRegistryStore | None":
    """Semantic index of MCP tools, or None without chromadb (search_mcp_tools then matches keywords)."""
    if not chroma.chroma_available():
        return None
    from dostuff.lib.mcp.mcp_tool_registry_store import MCPToolRegistryStore
    return MCPToolRegistryStore(persist_path=_persist_path())


def memory_collections() -> list[tuple[str, object, dict | None]]:
    """(name, collection, filter a real query uses) for a health check, skipping empty/missing ones.

    The filter matters: an unfiltered query can succeed where the filtered one callers run fails.
    Returns [] without chromadb.
    """
    if not chroma.chroma_available() or memory_backend() != "chroma":
        return []
    client = chroma.get_chroma_client(_persist_path())
    existing = {collection.name for collection in client.list_collections()}
    user_id = get_config().get_user_id()
    found: list[tuple[str, object, dict | None]] = []
    for name in ("semantic_memory", "episodic_memory", "mcp_tools_index"):
        if name not in existing:
            continue
        collection = client.get_collection(name)
        if collection.count() <= 0:
            continue
        # mcp_tools_index is keyed by server rather than user; check_collection picks a real
        # server_name from the rows, since an empty filter would test a different code path
        where = None if name == "mcp_tools_index" else {"user_id": user_id}
        found.append((name, collection, where))
    return found
