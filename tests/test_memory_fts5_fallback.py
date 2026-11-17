"""hybrid and sqlite keep memory in memory.db, whose keyword index is FTS5. On a Python whose sqlite3 was built
without it they run as the chroma backend (as before hybrid existed) instead of failing every start."""
import asyncio
import importlib.util
import sqlite3

import pytest

from dostuff.config import get_config
from dostuff.lib.memory import backend, chroma, stores


class _NoFts5(sqlite3.Connection):
    """A connection that behaves like a sqlite3 built without FTS5."""

    def _refuse(self, sql: str) -> None:
        if "fts5" in sql.lower():
            raise sqlite3.OperationalError("no such module: fts5")

    def execute(self, sql, *args):  # type: ignore[override]
        self._refuse(sql)
        return super().execute(sql, *args)

    def executescript(self, sql):  # type: ignore[override]
        self._refuse(sql)
        return super().executescript(sql)


@pytest.fixture
def no_fts5(tmp_path, monkeypatch):
    """sqlite3 without FTS5, memory.backend hybrid, and a data dir of the test's own."""
    real_connect = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: real_connect(*a, **{"factory": _NoFts5, **k}))
    backend.fts5_available.cache_clear()
    cfg = get_config()
    monkeypatch.setattr(type(cfg), "get_data_dir", lambda self, *a, **k: tmp_path)
    monkeypatch.setattr(cfg, "memory_backend", "hybrid")
    yield tmp_path
    backend.fts5_available.cache_clear()


def test_the_probe_sees_a_sqlite3_without_fts5(no_fts5):
    assert backend.fts5_available() is False


def test_the_probe_sees_fts5_here():
    backend.fts5_available.cache_clear()
    assert backend.fts5_available() is True


@pytest.mark.parametrize("configured", ["hybrid", "sqlite"])
def test_without_fts5_or_chromadb_memory_is_off_instead_of_failing(no_fts5, monkeypatch, configured):
    monkeypatch.setattr(get_config(), "memory_backend", configured)
    monkeypatch.setattr(chroma, "chroma_available", lambda: False)
    assert backend.memory_backend() == "chroma"
    assert not backend.memory_available()
    # Before: both raised "no such module: fts5", which failed the TUI's startup and every `dostuff -p` turn
    assert stores.get_semantic_store() is None and stores.get_episodic_store() is None
    assert stores.chroma_import_result() is None
    notice = backend.fallback_notice()
    assert "Long-term memory is off" in notice and "FTS5" in notice and configured in notice
    assert not (no_fts5 / "memory.db").exists()


def test_without_fts5_the_memory_tools_are_not_offered(no_fts5, monkeypatch):
    from dostuff.tools.definitions import default_tool_names
    monkeypatch.setattr(chroma, "chroma_available", lambda: False)
    names = default_tool_names()
    assert not {"save_memory", "save_episode", "search_memory"} & set(names)


@pytest.mark.skipif(importlib.util.find_spec("chromadb") is None, reason="needs the memory extra")
def test_without_fts5_but_with_chromadb_memory_uses_the_chroma_store(no_fts5):
    from dostuff.lib.memory.episodic_memory_store import EpisodicMemoryStore
    from dostuff.lib.memory.semantic_memory_store import SemanticMemoryStore
    assert backend.memory_backend() == "chroma" and backend.memory_available()
    assert isinstance(stores.get_semantic_store(), SemanticMemoryStore)
    assert isinstance(stores.get_episodic_store(), EpisodicMemoryStore)
    assert "older chroma store" in backend.fallback_notice()


def test_no_notice_when_the_configured_backend_runs(tmp_path, monkeypatch):
    backend.fts5_available.cache_clear()
    for configured in ("hybrid", "sqlite", "chroma"):
        monkeypatch.setattr(get_config(), "memory_backend", configured)
        assert backend.memory_backend() == configured
        assert backend.fallback_notice() == ""


def test_headless_prints_the_fallback_notice(no_fts5, monkeypatch):
    from dostuff import headless
    monkeypatch.setattr(chroma, "chroma_available", lambda: False)
    assert any("FTS5" in n for n in headless._memory_notices())


def test_headless_prints_the_import_of_old_memories(tmp_path, monkeypatch):
    """`dostuff -p` copies the old chroma store's memories like the TUI does, and now says so on stderr."""
    from dostuff import headless
    from dostuff.lib.memory.migrate import ImportResult
    cfg = get_config()
    monkeypatch.setattr(type(cfg), "get_data_dir", lambda self, *a, **k: tmp_path)
    monkeypatch.setattr(cfg, "memory_backend", "hybrid")
    monkeypatch.setattr(stores, "chroma_import_result", lambda: ImportResult("imported", 3, 2))
    assert headless._memory_notices() == [ImportResult("imported", 3, 2).message()]
    monkeypatch.setattr(stores, "chroma_import_result", lambda: ImportResult("done"))
    assert headless._memory_notices() == []
    monkeypatch.setattr(stores, "chroma_import_result", lambda: ImportResult("failed", error="disk full"))
    assert "disk full" in headless._memory_notices()[0]


def test_doctor_names_the_missing_fts5(no_fts5, monkeypatch):
    from dostuff.helpers.ui.doctor import doctor_checks
    monkeypatch.setattr(chroma, "chroma_available", lambda: False)
    memory = [c for c in doctor_checks() if "memory" in c.label.lower()]
    assert any("FTS5" in c.label and c.ok is False for c in memory)
    assert not any("chromadb not installed" in c.label for c in memory)


def test_save_memory_refuses_as_memory_off_not_with_the_fts5_error(no_fts5, monkeypatch):
    from dostuff.tools.memory.save_memory import save_memory
    monkeypatch.setattr(chroma, "chroma_available", lambda: False)
    with pytest.raises(RuntimeError, match="Long-term memory is off"):
        asyncio.run(save_memory(key="editor", value="neovim"))
