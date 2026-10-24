"""One semantic and one episodic store per data dir, shared by the TUI (recall) and the memory tools (save).

Each getter returns None when chromadb isn't installed: long-term memory is then off.
"""
from functools import lru_cache
from typing import TYPE_CHECKING

from dostuff.config import get_config
from dostuff.lib.memory import chroma
from dostuff.lib.memory.episodic_memory_store import EpisodicMemoryStore
from dostuff.lib.memory.semantic_memory_store import SemanticMemoryStore

if TYPE_CHECKING:
    from dostuff.lib.mcp.mcp_tool_registry_store import MCPToolRegistryStore


def _persist_path() -> str:
    return str(get_config().get_data_dir() / "chroma")


@lru_cache(maxsize=None)
def _semantic(persist_path: str) -> SemanticMemoryStore:
    return SemanticMemoryStore(persist_path=persist_path)


@lru_cache(maxsize=None)
def _episodic(persist_path: str) -> EpisodicMemoryStore:
    return EpisodicMemoryStore(persist_path=persist_path)


def get_semantic_store() -> SemanticMemoryStore | None:
    return _semantic(_persist_path()) if chroma.chroma_available() else None


def get_episodic_store() -> EpisodicMemoryStore | None:
    return _episodic(_persist_path()) if chroma.chroma_available() else None


def make_tool_registry_store() -> "MCPToolRegistryStore | None":
    """Semantic index of MCP tools, or None without chromadb (search_mcp_tools then matches keywords)."""
    if not chroma.chroma_available():
        return None
    from dostuff.lib.mcp.mcp_tool_registry_store import MCPToolRegistryStore
    return MCPToolRegistryStore(persist_path=_persist_path())
