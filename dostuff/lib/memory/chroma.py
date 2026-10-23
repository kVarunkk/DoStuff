"""One shared Chroma PersistentClient per storage path."""
import threading
from pathlib import Path
from typing import Any

import chromadb

_clients: dict[str, Any] = {}
_lock = threading.Lock()


def get_chroma_client(persist_path: str) -> Any:
    key = str(Path(persist_path).expanduser().resolve())
    with _lock:
        client = _clients.get(key)
        if client is None:
            client = chromadb.PersistentClient(path=key)
            _clients[key] = client
        return client
