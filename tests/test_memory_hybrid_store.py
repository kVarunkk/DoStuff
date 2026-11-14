"""The hybrid memory backend (memory.backend: hybrid): SQLite of record, searched by keywords and embeddings."""
import asyncio
import json
import time

import pytest

pytest.importorskip("chromadb")

from dostuff.lib.memory.hybrid_memory_store import HybridEpisodeStore, HybridFactStore  # noqa: E402
from dostuff.lib.memory.sqlite_memory_store import SQLiteEpisodeStore, SQLiteFactStore  # noqa: E402


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def facts(tmp_path):
    return HybridFactStore(str(tmp_path / "memory.db"), str(tmp_path / "memory-index"))


@pytest.fixture
def episodes(tmp_path):
    return HybridEpisodeStore(str(tmp_path / "memory.db"), str(tmp_path / "memory-index"))


def _ep(i, anchor, summary, event_type="DECISION", wd="/p"):
    meta = {"user_id": "u", "session_id": "s", "working_dir": wd, "event_type": event_type, "anchor_event": anchor,
            "summary": summary, "tags": json.dumps([]), "created_at_epoch": int(time.time()), "created_at_iso": ""}
    return {"id": f"e{i}", "vector_text": f"- EVENT TYPE: {event_type}\n  - ANCHOR EVENT: {anchor}\n  - SUMMARY: {summary}",
            "metadata": meta}


FACTS = [("py_errors", "Raises ValueError with a message naming the bad argument for invalid input"),
         ("keyboard", "Types on a Dvorak keyboard layout"),
         ("editor", "Uses Neovim with the LazyVim distribution")]


def _seed(facts):
    for key, value in FACTS:
        run(facts.upsert_fact("u", {"key": key, "value": value}))


def test_paraphrase_found_by_embeddings_where_keywords_find_nothing(facts):
    _seed(facts)
    prompt = "what exception do I throw on illegal values?"
    assert run(facts.keyword.query("u", prompt, top_k=3, min_similarity=SQLiteFactStore.RECALL_FLOOR)) == []
    found = run(facts.query("u", prompt, top_k=3, min_similarity=facts.RECALL_FLOOR))
    assert [f["key"] for f in found][:1] == ["py_errors"]


def test_keyword_hit_kept_and_unrelated_prompt_finds_nothing(facts):
    _seed(facts)
    assert [f["key"] for f in run(facts.query("u", "dvorak", top_k=3, min_similarity=facts.RECALL_FLOOR))][:1] == \
        ["keyboard"]
    assert run(facts.query("u", "hi", top_k=3, min_similarity=facts.RECALL_FLOOR)) == []


def test_each_fact_once_and_store_of_record_is_sqlite(facts):
    _seed(facts)
    found = run(facts.query("u", "ValueError for invalid input", top_k=5, min_similarity=0.0))
    assert len({f["key"] for f in found}) == len(found)
    assert {f["key"] for f in run(facts.list_all("u"))} == {k for k, _ in FACTS}
    assert run(facts.delete("u", "keyboard"))
    assert run(facts.embedding.query("u", "Dvorak keyboard layout", top_k=3, min_similarity=0.5)) == []
    assert "keyboard" not in {f["key"] for f in run(facts.list_all("u"))}


def test_broken_embedding_index_falls_back_to_keywords(facts, monkeypatch):
    _seed(facts)

    async def boom(*a, **k):
        raise RuntimeError("index divergence")

    monkeypatch.setattr(facts.embedding, "query", boom)
    monkeypatch.setattr(facts.embedding, "upsert_fact", boom)
    run(facts.upsert_fact("u", {"key": "shell", "value": "Uses fish as the login shell"}))
    assert [f["key"] for f in run(facts.query("u", "fish shell", top_k=3, min_similarity=facts.RECALL_FLOOR))] == \
        ["shell"]


def test_episode_union_and_reworded_duplicate(episodes):
    run(episodes.add([_ep(1, "Enabled WAL mode on the SQLite store", "Concurrent writers hit 'database is locked'"),
                      _ep(2, "Chose ruff over flake8", "flake8 kept fighting black over line length")]))
    found = run(episodes.query("u", "why do parallel writes to the db stop failing?", top_k=3, working_dir="/p"))
    assert [e["id"] for e in found][:1] == ["e1"]
    stats: dict = {}
    run(episodes.query("u", "linter choice", top_k=3, working_dir="/p", stats=stats))
    assert stats["matching"]
    same = run(episodes.nearest("u", "- EVENT TYPE: DECISION\n  - ANCHOR EVENT: Enabled WAL mode on the SQLite store\n"
                                     "  - SUMMARY: Concurrent writers hit 'database is locked'", "/p"))
    assert same and same["id"] == "e1" and same["similarity"] >= episodes.DUPLICATE
    assert run(episodes.nearest("u", "Bought a new desk lamp", "/p")) is None
    assert run(episodes.count("u", "/p")) == 2 and run(episodes.delete("u", "e2")) and run(episodes.count("u", "/p")) == 1


def test_backend_setting_selects_hybrid_and_keywords_without_chromadb(tmp_path, monkeypatch):
    from dostuff.config import get_config
    from dostuff.lib.memory import backend, chroma, stores
    cfg = get_config()
    monkeypatch.setattr(type(cfg), "get_data_dir", lambda self, *a, **k: tmp_path)
    monkeypatch.setattr(cfg, "memory_backend", "hybrid")
    assert backend.memory_available()
    assert isinstance(stores.get_semantic_store(), HybridFactStore)
    assert isinstance(stores.get_episodic_store(), HybridEpisodeStore)
    monkeypatch.setattr(chroma, "chroma_available", lambda: False)
    assert backend.memory_available()
    assert isinstance(stores.get_semantic_store(), SQLiteFactStore)
    assert isinstance(stores.get_episodic_store(), SQLiteEpisodeStore)
