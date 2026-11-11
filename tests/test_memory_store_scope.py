"""Fact store: value-only documents, timestamps, user/project scope (memory recall spec REQ-1, D4, D6, D7).

Real Chroma in a temp dir; skipped without chromadb.
"""
import asyncio

import pytest

pytest.importorskip("chromadb")

from dostuff.lib.memory import semantic_memory_store as sms  # noqa: E402

USER = "u1"
PROJ_A = "/work/proj-a"
PROJ_B = "/work/proj-b"


@pytest.fixture
def store(tmp_path, monkeypatch):
    from dostuff.lib.memory import chroma

    monkeypatch.setattr(chroma, "_clients", {})
    return sms.SemanticMemoryStore(persist_path=str(tmp_path / "chroma"))


@pytest.fixture
def clock(monkeypatch):
    """A settable clock for the store's timestamps."""
    now = {"t": 1_000}
    monkeypatch.setattr(sms.time, "time", lambda: now["t"])
    return now


def _row(store, row_id):
    got = store._collection.get(ids=[row_id], include=["documents", "metadatas"])
    assert got["ids"], f"no row {row_id}"
    return got["documents"][0], got["metadatas"][0]


def _upsert(store, key, value, **scope):
    asyncio.run(store.upsert_fact(USER, {"key": key, "value": value, "source": "user_message"}, **scope))


# T1.1 -----------------------------------------------------------------------------------------------------------

def test_upsert_writes_value_as_document(store):
    _upsert(store, "preferred_language", "Rust for code unless told otherwise")
    document, meta = _row(store, f"{USER}:preferred_language")
    assert document == "Rust for code unless told otherwise"
    assert meta["scope"] == "user"
    assert "working_dir" not in meta


def test_upsert_sets_created_and_updated(store, clock):
    _upsert(store, "name", "Nemo")
    _, meta = _row(store, f"{USER}:name")
    assert meta["created_at_epoch"] == 1_000
    assert meta["updated_at_epoch"] == 1_000


def test_update_keeps_created_bumps_updated(store, clock):
    _upsert(store, "name", "Captain Nemo")
    clock["t"] = 5_000
    _upsert(store, "name", "Admiral Nemo")
    document, meta = _row(store, f"{USER}:name")
    assert document == "Admiral Nemo"
    assert meta["created_at_epoch"] == 1_000
    assert meta["updated_at_epoch"] == 5_000


# T1.2 -----------------------------------------------------------------------------------------------------------

def test_project_fact_id_and_metadata(store):
    _upsert(store, "linter", "ruff, not flake8", scope="project", working_dir=PROJ_A)
    row_id = sms.fact_id(USER, "linter", "project", PROJ_A)
    assert row_id.startswith(f"{USER}:p") and row_id.endswith(":linter")
    assert len(row_id.split(":")[1]) == 13  # "p" + 12 hex chars of sha1(working_dir)
    _, meta = _row(store, row_id)
    assert meta["scope"] == "project"
    assert meta["working_dir"] == PROJ_A


def test_same_key_two_projects_two_rows(store):
    _upsert(store, "linter", "ruff", scope="project", working_dir=PROJ_A)
    _upsert(store, "linter", "eslint", scope="project", working_dir=PROJ_B)
    assert _row(store, sms.fact_id(USER, "linter", "project", PROJ_A))[0] == "ruff"
    assert _row(store, sms.fact_id(USER, "linter", "project", PROJ_B))[0] == "eslint"


def test_user_and_project_same_key_two_rows(store):
    _upsert(store, "test_runner", "pytest everywhere")
    _upsert(store, "test_runner", "nox sessions", scope="project", working_dir=PROJ_A)
    assert _row(store, f"{USER}:test_runner")[0] == "pytest everywhere"
    assert _row(store, sms.fact_id(USER, "test_runner", "project", PROJ_A))[0] == "nox sessions"


# T1.3 -----------------------------------------------------------------------------------------------------------

@pytest.fixture
def mixed(store, clock):
    _upsert(store, "name", "Nemo")
    clock["t"] = 2_000
    _upsert(store, "linter", "ruff", scope="project", working_dir=PROJ_A)
    clock["t"] = 3_000
    _upsert(store, "linter", "eslint", scope="project", working_dir=PROJ_B)
    return store


def _keys(facts):
    return [(f["scope"], f.get("working_dir"), f["key"]) for f in facts]


def test_list_all_user_only(mixed):
    assert _keys(asyncio.run(mixed.list_all(USER, scope="user"))) == [("user", None, "name")]


def test_list_all_project_only(mixed):
    facts = asyncio.run(mixed.list_all(USER, scope="project", working_dir=PROJ_A))
    assert _keys(facts) == [("project", PROJ_A, "linter")]
    assert facts[0]["value"] == "ruff"


def test_list_all_both(mixed):
    # Newest first; project B's fact is not this project's
    facts = asyncio.run(mixed.list_all(USER, working_dir=PROJ_A))
    assert _keys(facts) == [("project", PROJ_A, "linter"), ("user", None, "name")]
    assert [f["updated_at_epoch"] for f in facts] == [2_000, 1_000]


def test_legacy_row_is_user_fact_sorted_oldest(mixed):
    # A row exactly as the store wrote it before this spec: no scope, no timestamps, "User's …" document
    mixed._collection.upsert(
        ids=[f"{USER}:editor"], documents=["User's editor is Neovim"],
        metadatas=[{"user_id": USER, "key": "editor", "value": "Neovim", "source": "user_message"}],
    )
    facts = asyncio.run(mixed.list_all(USER, scope="user"))
    assert _keys(facts) == [("user", None, "name"), ("user", None, "editor")]
    assert facts[-1]["updated_at_epoch"] == 0 and facts[-1]["created_at_epoch"] == 0
    # A scoped query still finds it as a user fact
    found = asyncio.run(mixed.query(USER, "which editor", top_k=5, scope="user"))
    assert "editor" in [f["key"] for f in found]


# T1.4 -----------------------------------------------------------------------------------------------------------

def test_delete_project_fact_leaves_user_fact(store):
    _upsert(store, "test_runner", "pytest everywhere")
    _upsert(store, "test_runner", "nox sessions", scope="project", working_dir=PROJ_A)
    assert asyncio.run(store.delete(USER, "test_runner", scope="project", working_dir=PROJ_A))
    assert not asyncio.run(store.delete(USER, "test_runner", scope="project", working_dir=PROJ_A))
    assert _keys(asyncio.run(store.list_all(USER))) == [("user", None, "test_runner")]
    assert asyncio.run(store.delete(USER, "test_runner"))
    assert asyncio.run(store.list_all(USER)) == []


# T1.5 -----------------------------------------------------------------------------------------------------------

def test_resolve_compares_same_scope_only(store, monkeypatch):
    from dostuff.helpers.memory import resolve_memory_operation as rmo

    shown: list[str] = []

    async def judge(**kwargs):
        shown.append(kwargs["messages"][0]["content"])
        from litellm import ModelResponse
        return ModelResponse(choices=[{"message": {"role": "assistant", "content": '{"operation": "ADD"}'}}])

    monkeypatch.setattr(rmo, "acompletion", judge)
    _upsert(store, "linting_tool", "Uses ruff for linting Python")
    project_fact = {"key": "linter", "value": "Uses ruff for linting Python code", "source": "user_message"}
    result = asyncio.run(rmo.resolve_memory_operation(USER, project_fact, store, scope="project", working_dir=PROJ_A))

    assert result == ("ADD", "linter")
    assert all("linting_tool" not in prompt for prompt in shown)  # never asked to merge with the user fact
    assert _row(store, f"{USER}:linting_tool")[0] == "Uses ruff for linting Python"
    assert _row(store, sms.fact_id(USER, "linter", "project", PROJ_A))[0] == "Uses ruff for linting Python code"


# T1.6 / T1.7 -----------------------------------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.calls = []

    async def query(self, user_id, text, top_k=3, min_similarity=0.0, scope=None, working_dir=None):
        return []

    async def upsert_fact(self, user_id, fact, scope="user", working_dir=None):
        self.calls.append((fact["key"], scope, working_dir))


def _patch_save(monkeypatch, recorder):
    from types import SimpleNamespace

    from dostuff.tools.memory import save_memory as sm
    monkeypatch.setattr(sm, "get_config", lambda: SimpleNamespace(get_user_id=lambda: USER))
    monkeypatch.setattr(sm, "get_semantic_store", lambda: recorder)
    return sm


def test_save_memory_project_scope_uses_cwd(monkeypatch, tmp_path):
    recorder = _Recorder()
    sm = _patch_save(monkeypatch, recorder)
    monkeypatch.chdir(tmp_path)
    result = asyncio.run(sm.save_memory("linter", "ruff, not flake8", scope="project"))
    assert recorder.calls == [("linter", "project", str(tmp_path.resolve()))]
    assert "this project" in result


def test_save_memory_default_scope_user(monkeypatch):
    recorder = _Recorder()
    sm = _patch_save(monkeypatch, recorder)
    asyncio.run(sm.save_memory("name", "Admiral Nemo"))
    assert recorder.calls == [("name", "user", None)]


def test_memory_instructions_name_new_blocks():
    from dostuff.helpers.agent.constants import MEMORY_INSTRUCTIONS
    assert "<user_facts>" in MEMORY_INSTRUCTIONS and "<project_facts>" in MEMORY_INSTRUCTIONS
    assert 'scope="project"' in MEMORY_INSTRUCTIONS
    assert "<relevant_user_facts>" not in MEMORY_INSTRUCTIONS
