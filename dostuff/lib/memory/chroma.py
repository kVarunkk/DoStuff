"""One shared Chroma PersistentClient per storage path.

chromadb is optional (`pip install "dostuff[memory]"`): without it long-term memory is off and
MCP tool search falls back to keyword matching. Import it only through get_chroma_client.
"""
import importlib.util
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

MEMORY_INSTALL_HINT = 'pip install "dostuff[memory]"'

_clients: dict[str, Any] = {}
_lock = threading.Lock()


@lru_cache(maxsize=None)
def chroma_available() -> bool:
    """True if chromadb is installed (long-term memory and semantic MCP tool search work)."""
    return importlib.util.find_spec("chromadb") is not None


def embedding_model_missing() -> bool:
    """True if Chroma's default embedding model (~80 MB) still downloads on first use."""
    try:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    except Exception:
        return False
    return not (Path(ONNXMiniLM_L6_V2.DOWNLOAD_PATH) / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME / "model.onnx").exists()


def download_embedding_model() -> None:
    """Fetch the model now (embedding one string triggers the download)."""
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    ONNXMiniLM_L6_V2()(["warm up"])


def get_chroma_client(persist_path: str) -> Any:
    import chromadb
    key = str(Path(persist_path).expanduser().resolve())
    with _lock:
        client = _clients.get(key)
        if client is None:
            client = chromadb.PersistentClient(path=key)
            _clients[key] = client
        return client
