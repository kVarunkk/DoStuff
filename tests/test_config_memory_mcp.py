"""Config loading, memory hardening and MCP lookup (REQ-9.2, 7.2, 1.9, 8.4, 8.5)."""
import asyncio
from types import SimpleNamespace

import pytest

from dostuff import config as config_module
from dostuff.config import Config, load_env_files
from dostuff.helpers.memory import resolve_memory_operation as rmo
from dostuff.lib.mcp.mcp_client import MCPClient, tool_key
from dostuff.tools.mcp.search_mcp_tools import keyword_search


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    (home / ".dostuff").mkdir(parents=True)
    (proj / ".dostuff").mkdir(parents=True)
    monkeypatch.setattr(config_module.Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(proj)
    for var in ("MODEL", "MODEL_PROVIDER", "DOSTUFF_A", "DOSTUFF_B"):
        monkeypatch.delenv(var, raising=False)
    return home, proj


def test_env_project_overrides_user_but_not_real_env(home, monkeypatch):
    user, proj = home
    (user / ".dostuff" / ".env").write_text("DOSTUFF_A=user\nDOSTUFF_B=user\n")
    (proj / ".dostuff" / ".env").write_text("DOSTUFF_A=project\n")
    monkeypatch.setenv("DOSTUFF_B", "real")
    load_env_files()
    import os
    assert os.environ["DOSTUFF_A"] == "project"
    assert os.environ["DOSTUFF_B"] == "real"


def test_empty_blocks_and_model_read_at_call_time(home, monkeypatch):
    user, _ = home
    (user / ".dostuff" / "config.yaml").write_text(
        "loop:\ntracing:\npermissions:\nmemory:\nmodels:\n  - name: gemini-2.5-flash\n    provider: gemini\n    active: true\n    api_key_env: MY_KEY\n"
    )
    cfg = Config()
    assert cfg.max_iterations > 0 and cfg.permission_mode == "ask" and cfg.episodic_scope == "project"
    assert cfg.model_name == "gemini/gemini-2.5-flash"
    monkeypatch.setenv("MODEL", "openai/gpt-4o")
    assert cfg.model_name == "openai/gpt-4o"  # not frozen at construction
    assert cfg.api_key() is None
    monkeypatch.setenv("MY_KEY", "sk-test")
    assert cfg.api_key() == "sk-test"




def test_jobs_status_item_retired(home):
    """`jobs` moved from a configurable status-bar item to one that is always shown; old configs listing it get no warning."""
    user, _ = home
    assert "jobs" not in config_module.STATUS_BAR_ITEMS and "jobs" not in config_module.DEFAULT_STATUS_BAR
    (user / ".dostuff" / "config.yaml").write_text("ui:\n  status_bar: [mode, jobs]\n")
    config = Config()
    assert config.status_bar_items == ["mode"] and config.status_bar_unknown == []

def test_retry_settings(home):
    user, _ = home
    cfg_file = user / ".dostuff" / "config.yaml"
    cfg_file.write_text("loop:\n")
    cfg = Config()
    assert (cfg.max_retries, cfg.retry_max_delay) == (6, 60.0)
    cfg_file.write_text("loop:\n  max_retries: 0\n  retry_max_delay: 15\n")
    cfg = Config()
    assert (cfg.max_retries, cfg.retry_max_delay) == (0, 15.0)

class FakeMemory:
    def __init__(self, similar):
        self.similar = similar
        self.upserts = []
        self.min_similarity = None

    async def query(self, user_id, text, top_k=3, min_similarity=0.0):
        self.min_similarity = min_similarity
        return self.similar

    async def upsert_fact(self, user_id, fact):
        self.upserts.append(fact)


def _decide(monkeypatch, content):
    from litellm import ModelResponse

    async def fake_acompletion(**kwargs):
        return ModelResponse(choices=[{"message": {"role": "assistant", "content": content}}])
    monkeypatch.setattr(rmo, "acompletion", fake_acompletion)


def test_update_with_unknown_key_becomes_add(monkeypatch):
    _decide(monkeypatch, '{"operation": "UPDATE", "replace_key": "does_not_exist"}')
    store = FakeMemory([{"key": "color", "value": "blue", "source": "user_message"}])
    asyncio.run(rmo.resolve_memory_operation("u", {"key": "food", "value": "pizza"}, store))
    assert store.upserts == [{"key": "food", "value": "pizza"}]


def test_update_known_key_and_bad_json(monkeypatch):
    _decide(monkeypatch, '{"operation": "UPDATE", "replace_key": "color"}')
    store = FakeMemory([{"key": "color", "value": "blue", "source": "user_message"}])
    asyncio.run(rmo.resolve_memory_operation("u", {"key": "fav_color", "value": "green"}, store))
    assert store.upserts[0]["key"] == "color"
    _decide(monkeypatch, "not json at all")
    store = FakeMemory([{"key": "color", "value": "blue", "source": "user_message"}])
    asyncio.run(rmo.resolve_memory_operation("u", {"key": "x", "value": "y"}, store))
    assert store.upserts == [{"key": "x", "value": "y"}]


def test_save_memory_tool_reports_operation(monkeypatch):
    from dostuff.tools.memory import save_memory as sm
    monkeypatch.setattr(sm, "get_config", lambda: SimpleNamespace(get_user_id=lambda: "u"))

    store = FakeMemory([])
    monkeypatch.setattr(sm, "get_semantic_store", lambda: store)
    assert asyncio.run(sm.save_memory(" lang ", " Prefers Python ")).startswith("Saved memory 'lang'")
    assert store.upserts == [{"key": "lang", "value": "Prefers Python", "source": "user_message"}]

    _decide(monkeypatch, '{"operation": "NOOP", "replace_key": "lang"}')
    store = FakeMemory([{"key": "lang", "value": "Prefers Python", "source": "user_message"}])
    monkeypatch.setattr(sm, "get_semantic_store", lambda: store)
    assert "Already remembered" in asyncio.run(sm.save_memory("language", "Likes Python"))
    assert store.upserts == []

    with pytest.raises(ValueError):
        asyncio.run(sm.save_memory("k", "  "))


def test_save_episode_tool_stores_episode(monkeypatch, tmp_path):
    from dostuff.tools.memory import save_episode as se
    added = []

    class FakeEpisodic:
        async def find_duplicate(self, user_id, vector_text, working_dir):
            return None

        async def add(self, episodes):
            added.extend(episodes)
    monkeypatch.setattr(se, "get_config", lambda: SimpleNamespace(get_user_id=lambda: "u"))
    monkeypatch.setattr(se, "get_episodic_store", lambda: FakeEpisodic())
    monkeypatch.chdir(tmp_path)

    result = asyncio.run(se.save_episode("DECISION", "Use WAL mode", "Fixes lock errors", session_id="s1", tags=["sqlite"]))
    assert result.startswith("Saved episode ep_s1_")
    meta = added[0]["metadata"]
    assert meta["user_id"] == "u" and meta["session_id"] == "s1" and meta["event_type"] == "DECISION"
    assert meta["working_dir"] == str(tmp_path.resolve()) and meta["tags"] == '["sqlite"]'
    assert "context_window" not in meta


def test_save_episode_skips_a_duplicate(monkeypatch, tmp_path):
    from dostuff.tools.memory import save_episode as se
    added, asked = [], []

    class FakeEpisodic:
        async def find_duplicate(self, user_id, vector_text, working_dir):
            asked.append((user_id, working_dir))
            return "ep_old_1234abcd" if "WAL" in vector_text else None

        async def add(self, episodes):
            added.extend(episodes)
    monkeypatch.setattr(se, "get_config", lambda: SimpleNamespace(get_user_id=lambda: "u"))
    monkeypatch.setattr(se, "get_episodic_store", lambda: FakeEpisodic())
    monkeypatch.chdir(tmp_path)

    result = asyncio.run(se.save_episode("LEARNING", "WAL mode fixed the locks", "Same as before", session_id="s2"))
    assert result == "Already recorded as ep_old_1234abcd; nothing saved." and added == []
    assert asked == [("u", str(tmp_path.resolve()))]
    asyncio.run(se.save_episode("DECISION", "Use httpx", "Async support", session_id="s2"))
    assert len(added) == 1


class FakeCollection:
    """Chroma collection stand-in: query returns the given rows with their cosine distances."""
    def __init__(self, rows):
        self.rows = rows  # [(id, metadata, distance)]
        self.deleted = []

    def count(self):
        return len(self.rows)

    def query(self, query_texts, n_results, where):
        rows = self.rows[:n_results]
        return {"ids": [[r[0] for r in rows]], "metadatas": [[r[1] for r in rows]],
                "distances": [[r[2] for r in rows]], "documents": [["" for _ in rows]]}

    def get(self, ids=None, where=None):
        rows = [r for r in self.rows if ids is None or r[0] in ids]
        return {"ids": [r[0] for r in rows], "metadatas": [r[1] for r in rows]}

    def delete(self, ids):
        self.deleted.extend(ids)


def _store(cls, rows):
    store = cls.__new__(cls)
    store._collection = FakeCollection(rows)
    return store


def test_fact_recall_floor_and_dedup_floor():
    from dostuff.lib.memory.semantic_memory_store import MIN_FACT_SIMILARITY, SemanticMemoryStore
    store = _store(SemanticMemoryStore, [
        ("u:commit_style", {"key": "commit_style", "value": "Conventional commits", "source": "user_message"}, 0.53),
        ("u:name", {"key": "name", "value": "Varun", "source": "user_message"}, 0.83),
    ])
    assert [f["key"] for f in asyncio.run(store.query("u", "write a commit message", top_k=5))] == ["commit_style", "name"]
    recalled = asyncio.run(store.query("u", "write a commit message", top_k=5, min_similarity=MIN_FACT_SIMILARITY))
    assert [f["key"] for f in recalled] == ["commit_style"]  # 0.17 similar: left out

    # Dedup asks with its own (lower) floor, and a fact with nothing similar is added without a model call
    memory = FakeMemory([])
    assert asyncio.run(rmo.resolve_memory_operation("u", {"key": "editor", "value": "Uses Neovim"}, memory)) == ("ADD", "editor")
    assert memory.min_similarity == rmo.DEDUP_MIN_SIMILARITY < MIN_FACT_SIMILARITY


def test_episode_find_duplicate_list_and_delete():
    from dostuff.lib.memory.episodic_memory_store import DUPLICATE_EPISODE_SIMILARITY, EpisodicMemoryStore
    old = {"event_type": "DECISION", "anchor_event": "Use WAL", "created_at_iso": "2026-10-01T00:00:00Z", "created_at_epoch": 100}
    new = {"event_type": "LEARNING", "anchor_event": "Retry 429s", "created_at_iso": "2026-10-20T00:00:00Z", "created_at_epoch": 200}
    store = _store(EpisodicMemoryStore, [("ep_s1_aaaa1111", old, 0.1), ("ep_s2_bbbb2222", new, 0.5)])
    assert asyncio.run(store.find_duplicate("u", "Use WAL mode", "/proj")) == "ep_s1_aaaa1111"
    store._collection.rows[0] = ("ep_s1_aaaa1111", old, 1.0 - DUPLICATE_EPISODE_SIMILARITY + 0.05)
    assert asyncio.run(store.find_duplicate("u", "Use WAL mode", "/proj")) is None

    assert [ep["id"] for ep in asyncio.run(store.list_all("u"))] == ["ep_s2_bbbb2222", "ep_s1_aaaa1111"]
    assert asyncio.run(store.delete("u", "ep_s1_aaaa1111")) is True
    assert asyncio.run(store.delete("u", "ep_missing")) is False
    assert store._collection.deleted == ["ep_s1_aaaa1111"]


def test_memory_tools_registered_but_not_for_subagents():
    from dostuff.tools.definitions import TOOL_SCHEMAS
    from dostuff.tools.delegate_to_subagent import resolve_subagent_tools
    assert "session_id" not in TOOL_SCHEMAS["save_episode"]["function"]["parameters"]["properties"]
    assert resolve_subagent_tools(["save_memory", "save_episode", "read_file"]) == (["read_file"], ["save_memory", "save_episode"])


def _client_with_tools():
    client = MCPClient.__new__(MCPClient)
    client.mcp_tools = {}
    client.servers = {}
    for server, name, desc in [("gh", "create_issue", "Create a GitHub issue"), ("jira", "create_issue", "Create a Jira ticket"), ("gh", "list_repos", "List repositories")]:
        client.mcp_tools[tool_key(server, name)] = {
            "server_name": server, "tool_name": name, "schema": {"name": tool_key(server, name), "description": desc},
        }
    return client


def test_tools_keyed_by_server_and_resolved():
    client = _client_with_tools()
    assert client.resolve_tool("gh__create_issue")["server_name"] == "gh"
    assert client.resolve_tool("create_issue") is None  # ambiguous bare name
    assert client.resolve_tool("list_repos")["server_name"] == "gh"


def test_keyword_search_uses_token_overlap():
    client = _client_with_tools()
    names = [r["name"] for r in keyword_search("please create a github issue for me", client)]
    assert names[0] == "gh__create_issue"
    assert keyword_search("zzz", client) == []
