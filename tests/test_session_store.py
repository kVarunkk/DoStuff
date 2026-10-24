"""SQLite session store: working_steps rows + migration from v0.1.6 DBs (REQ-7.4, CON-003)."""
import asyncio
import json
import sqlite3

from dostuff.lib.memory.session_store import SCHEMA_VERSION, SQLiteSessionStore


def _make_v016_db(path):
    """A DB exactly as dostuff 0.1.6 left it: working history as a JSON blob, no user_version."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE session_steps (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                                    step_order INTEGER NOT NULL, step_data TEXT NOT NULL);
        CREATE INDEX idx_session_id_order ON session_steps(session_id, step_order);
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, working_dir TEXT, created_at TEXT,
                               total_tokens INTEGER DEFAULT 0, last_used TEXT,
                               prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
                               compaction_notes TEXT DEFAULT '', working_history TEXT DEFAULT '',
                               last_input_tokens INTEGER DEFAULT 0);
        """
    )
    steps = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    for i, step in enumerate(steps):
        conn.execute("INSERT INTO session_steps (session_id, step_order, step_data) VALUES (?, ?, ?)", ("old", i, json.dumps(step)))
    conn.execute(
        "INSERT INTO sessions (session_id, working_dir, total_tokens, prompt_tokens, compaction_notes, working_history) VALUES (?, ?, ?, ?, ?, ?)",
        ("old", "/proj", 42, 30, "earlier notes", json.dumps(steps[1:])),
    )
    conn.commit()
    conn.close()
    return steps


def test_migrates_v016_db_and_resumes(tmp_path):
    db = str(tmp_path / "sessions.db")
    steps = _make_v016_db(db)

    async def scenario():
        store = SQLiteSessionStore(db)
        try:
            assert await store.load("old") == steps
            assert await store.load_working_history("old") == steps[1:]
            meta = await store.get_session_meta("old")
            assert meta["total_tokens"] == 42 and meta["compaction_notes"] == "earlier notes"
            await store.append("old", {"role": "user", "content": "again"})
            assert (await store.load("old"))[-1]["content"] == "again"
            assert [s["content"] for s in await store.load_working_history("old")] == ["hi", "again"]
        finally:
            await store.close()

    asyncio.run(scenario())
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert conn.execute("SELECT working_history FROM sessions").fetchone()[0] == ""  # emptied, not dropped
    conn.close()

    # Re-opening a migrated DB does not duplicate rows
    async def reopen():
        store = SQLiteSessionStore(db)
        try:
            return await store.load_working_history("old")
        finally:
            await store.close()
    assert len(asyncio.run(reopen())) == 2


def test_compaction_flags_rows_and_clear_resets(tmp_path):
    async def scenario():
        store = SQLiteSessionStore(str(tmp_path / "s.db"))
        try:
            for i in range(5):
                await store.append("s", {"role": "user", "content": str(i)})
            await store.update_session_tokens("s", total_tokens=10, prompt_tokens=7, completion_tokens=3, last_input_tokens=7)
            await store.compact_working_history("s", keep_last=2, compaction_notes="summary")
            assert [s["content"] for s in await store.load_working_history("s")] == ["3", "4"]
            assert len(await store.load("s")) == 5  # audit trail untouched
            assert (await store.get_session_meta("s"))["compaction_notes"] == "summary"

            await store.clear_session("s")
            assert await store.load("s") == []
            assert await store.load_working_history("s") == []
            meta = await store.get_session_meta("s")
            assert meta["total_tokens"] == 0 and meta["last_input_tokens"] == 0 and meta["compaction_notes"] == ""
        finally:
            await store.close()

    asyncio.run(scenario())


def test_v1_db_gets_title_column_and_session_listing(tmp_path, monkeypatch):
    db = str(tmp_path / "sessions.db")
    _make_v016_db(db)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA user_version = 1")  # a v1 DB: no title column yet
    conn.commit()
    conn.close()
    monkeypatch.chdir(tmp_path)

    async def scenario():
        store = SQLiteSessionStore(db_path=db)
        await store.append("new", {"role": "user", "content": "<bash-input>ls</bash-input>\n<bash-output>\na\n</bash-output>"})
        meta = await store.get_session_meta("new")  # the first step creates the row
        assert meta is not None and meta["title"] == ""
        await store.save_session_meta("blank")
        listed = await store.list_sessions()
        assert {r["session_id"] for r in listed} == {"old", "new"}  # "blank" has no steps
        assert {r["session_id"] for r in await store.list_sessions(include_empty=True)} == {"old", "new", "blank"}
        assert [r["session_id"] for r in await store.list_sessions(working_dir=str(tmp_path))] == ["new"]
        assert next(r for r in listed if r["session_id"] == "new")["title"] == "!ls"
        assert next(r for r in listed if r["session_id"] == "old")["title"] == "hello"

        await store.set_title("old", "  Renamed  ")
        assert (await store.get_session_meta("old"))["title"] == "Renamed"
        await store.delete_session("old")
        assert await store.get_session_meta("old") is None
        assert await store.load("old") == [] and await store.load_working_history("old") == []
        await store.close()

    asyncio.run(scenario())
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 3
    conn.close()


def test_session_cost_adds_up_and_clears(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def scenario():
        store = SQLiteSessionStore(db_path=str(tmp_path / "sessions.db"))
        await store.add_session_cost("s", 0.25)
        await store.add_session_cost("s", 0.5)
        assert (await store.get_session_meta("s"))["cost_usd"] == 0.75
        await store.clear_session("s")
        assert (await store.get_session_meta("s"))["cost_usd"] == 0
        await store.close()

    asyncio.run(scenario())


def test_call_cost_known_and_unknown_models():
    from dostuff.helpers.agent.model_cost import call_cost, format_cost
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 500}
    cost = call_cost("openai/gpt-4o-mini", usage)
    assert cost is not None and 0 < cost < call_cost("openai/gpt-4o-mini", {**usage, "cached_tokens": 0})
    assert call_cost("ollama/llama3", usage) is None  # free / unpriced: hidden
    assert call_cost("nobody/no-such-model", usage) is None
    assert format_cost(1.234) == "$1.23" and format_cost(0.0042) == "$0.0042"


def test_fork_copies_session_and_leaves_original(tmp_path):
    db = str(tmp_path / "sessions.db")

    async def scenario():
        store = SQLiteSessionStore(db)
        try:
            for step in ({"role": "user", "content": "first"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "second"}):
                await store.append("src", step)
            await store.compact_working_history("src", 1, "notes")
            await store.update_session_tokens("src", total_tokens=10, prompt_tokens=7, completion_tokens=3, last_input_tokens=5)
            await store.add_session_cost("src", 0.5)

            await store.fork_session("src", "fork")
            assert await store.load("fork") == await store.load("src")
            assert await store.load_working_history("fork") == [{"role": "user", "content": "second"}]
            meta = await store.get_session_meta("fork")
            assert meta["compaction_notes"] == "notes" and meta["total_tokens"] == 10 and meta["cost_usd"] == 0.5
            assert meta["title"] == "first (fork)"

            await store.append("fork", {"role": "assistant", "content": "only in fork"})
            assert len(await store.load("src")) == 3 and len(await store.load("fork")) == 4

            await store.fork_session("src", "named", "try plan B")
            assert (await store.get_session_meta("named"))["title"] == "try plan B"
        finally:
            await store.close()

    asyncio.run(scenario())
