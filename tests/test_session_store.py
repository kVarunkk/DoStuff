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
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 4
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


# --- subagents rebuild T1.3: sessions that used delegate_to_subagent still load and replay (CON-005) ---

def test_pin_old_delegate_history_replays(tmp_path, fake_model):
    from pathlib import Path

    from dostuff.agent.loop import loop
    snapshot = json.loads((Path(__file__).parent / "snapshots" / "old_delegate_session.json").read_text("utf-8"))

    async def scenario():
        store = SQLiteSessionStore(str(tmp_path / "sessions.db"))
        try:
            for step in snapshot:
                await store.append("old", step)
            assert await store.load("old") == snapshot
            working = await store.load_working_history("old")
            assert working == snapshot
            result = await loop(session_id="old", turn_id="t", user_text="and logout?",
                                dynamic_system_instruction="sys", mcp_client=object(),  # type: ignore[arg-type]
                                working_history=working + [{"role": "user", "content": "and logout?"}],
                                turn_type="interactive_loop", stream=False)
        finally:
            await store.close()
        return result

    assert asyncio.run(scenario()) == "done"
    [call] = fake_model.calls
    assert call["messages"][1:len(snapshot) + 1] == snapshot  # the old steps reach the model unchanged


# --- subagents rebuild T5.1–T5.3: agent transcripts in the store (spec D14) --------------------

def _make_v3_db(path):
    """A DB as dostuff 0.2.5 left it (schema v3): no agent tables."""
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
                               last_input_tokens INTEGER DEFAULT 0, title TEXT DEFAULT '', cost_usd REAL DEFAULT 0);
        CREATE TABLE working_steps (session_id TEXT NOT NULL, seq INTEGER NOT NULL, step_json TEXT NOT NULL,
                                    compacted INTEGER DEFAULT 0, PRIMARY KEY (session_id, seq));
        PRAGMA user_version = 3;
        """
    )
    step = json.dumps({"role": "user", "content": "hello"})
    conn.execute("INSERT INTO session_steps (session_id, step_order, step_data) VALUES ('old', 0, ?)", (step,))
    conn.execute("INSERT INTO working_steps (session_id, seq, step_json) VALUES ('old', 0, ?)", (step,))
    conn.execute("INSERT INTO sessions (session_id, working_dir) VALUES ('old', '/proj')")
    conn.commit()
    conn.close()


def _agent_row(agent_id, session_id, **over):
    row = {"id": agent_id, "session_id": session_id, "agent_type": "explore", "description": "find x",
           "model": "m", "status": "running", "background": False, "worktree": None, "branch": None,
           "started_at": "2026-11-01T10:00:00", "ended_at": None, "tokens": 0, "cost_usd": 0.0,
           "result": None, "compaction_notes": ""}
    return {**row, **over}


def test_agent_tables_migration(tmp_path):
    db = str(tmp_path / "sessions.db")
    _make_v3_db(db)

    async def scenario():
        store = SQLiteSessionStore(db)
        try:
            assert await store.load("old") == [{"role": "user", "content": "hello"}]
            assert await store.load_agent_runs("old") == []
            await store.upsert_agent_run(_agent_row("a1", "old"))
            await store.upsert_agent_run(_agent_row("a1", "old", status="completed", result="r", tokens=7,
                                                    cost_usd=0.25, ended_at="2026-11-01T10:01:00", background=True))
            for i in range(3):
                await store.append_agent_step("a1", {"role": "user", "content": f"s{i}"})
            [row] = await store.load_agent_runs("old")
            assert row == _agent_row("a1", "old", status="completed", result="r", tokens=7, cost_usd=0.25,
                                     ended_at="2026-11-01T10:01:00", background=True)
            assert [s["content"] for s in await store.load_agent_steps("a1")] == ["s0", "s1", "s2"]
            await store.compact_agent_steps("a1", 1, "agent notes")
            assert [s["content"] for s in await store.load_agent_steps("a1", working=True)] == ["s2"]
            assert len(await store.load_agent_steps("a1")) == 3  # the transcript keeps every step
            assert (await store.load_agent_runs("old"))[0]["compaction_notes"] == "agent notes"
            assert await store.load_agent_steps("nope") == []
        finally:
            await store.close()

    asyncio.run(scenario())
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 4
    columns = [r[1] for r in conn.execute("PRAGMA table_info(agent_runs)")]
    assert columns == ["id", "session_id", "agent_type", "description", "model", "status", "background",
                       "worktree", "branch", "started_at", "ended_at", "tokens", "cost_usd", "result",
                       "compaction_notes"]
    assert [r[1] for r in conn.execute("PRAGMA table_info(agent_steps)")] == ["agent_id", "seq", "step_json", "compacted"]
    conn.execute("PRAGMA user_version = 3")  # run the migration a second time: a no-op
    conn.commit()
    conn.close()

    async def reopen():
        store = SQLiteSessionStore(db)
        try:
            return await store.load_agent_runs("old"), await store.load_agent_steps("a1")
        finally:
            await store.close()
    runs, steps = asyncio.run(reopen())
    assert len(runs) == 1 and len(steps) == 3


def _usage(prompt_tokens=100):
    from types import SimpleNamespace
    return SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=10, total_tokens=prompt_tokens + 10)


def test_subagent_steps_stored(agents, fake_model, tmp_path, monkeypatch):
    from conftest import FakeAdapter, fake_response, fake_tool_call

    from dostuff.agent.loop import loop
    from dostuff.helpers.tools.project_root import use_project_root
    from dostuff.lib.worktree import Worktree
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    fake_model.script("[MAIN]", fake_response(tool_calls=[fake_tool_call("agent", {"description": "read two",
                                                                                   "prompt": "[A] read"}, "m1")]),
                      fake_response("all done"))
    fake_model.script("[A]",
                      fake_response(tool_calls=[fake_tool_call("read_file", {"path": "a.txt"}, "c1")], usage=_usage()),
                      fake_response(tool_calls=[fake_tool_call("read_file", {"path": "b.txt"}, "c2")], usage=_usage()),
                      fake_response("report A", usage=_usage()))
    statuses: list[str] = []

    async def go():
        store = SQLiteSessionStore(str(tmp_path / "sessions.db"))
        real_upsert = store.upsert_agent_run

        async def spy(row):
            statuses.append(row["status"])
            await real_upsert(row)
        monkeypatch.setattr(store, "upsert_agent_run", spy)
        agents.attach_store(store)
        try:
            first = {"role": "user", "content": "[MAIN] go"}
            steps, working, current = [first], [first], [first]
            await store.append("s1", first)
            with use_project_root(tmp_path):
                result = await loop(session_id="s1", turn_id="t1", user_text="[MAIN] go", dynamic_system_instruction="",
                                    mcp_client=object(), working_history=working, turn_type="interactive_loop",  # type: ignore[arg-type]
                                    steps_history=steps, current_session_history=current, store=store,
                                    stream=False, adapter=FakeAdapter())
            assert result == "all done"
            await agents.flush()
            [run] = agents.runs()
            stored = await store.load_agent_steps(run.id)
            main = await store.load("s1")
            [row] = await store.load_agent_runs("s1")

            # a run in a (kept) worktree stores its path and branch
            wt = Worktree(name="w", path=tmp_path / "wt", branch="dostuff/w", repo_root=tmp_path, base="abc")
            wt.path.mkdir()
            wt_run = await agents.start(owner="s1", agent_type="general-purpose", description="in wt",
                                        prompt="[W] x", worktree=wt, adapter=FakeAdapter())
            await wt_run.done
            await agents.flush()
            wt_row = next(r for r in await store.load_agent_runs("s1") if r["id"] == wt_run.id)
            return run, stored, main, steps, row, wt_row
        finally:
            await store.close()

    run, stored, main, steps, row, wt_row = asyncio.run(go())
    assert stored == run.history
    assert [s["role"] for s in stored] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert stored[0]["content"] == "[A] read" and stored[-1]["content"] == "report A"
    assert "alpha" in json.dumps(stored)
    # the main history gains only the agent call and its result
    assert main == steps
    assert [s["role"] for s in main] == ["user", "assistant", "tool", "assistant"]
    assert main[1]["tool_calls"][0]["function"]["name"] == "agent"
    assert "alpha" not in json.dumps(main)
    assert statuses[0] == "running" and statuses[-1] == "completed"
    assert row["id"] == run.id and row["session_id"] == "s1" and row["status"] == "completed"
    assert row["agent_type"] == "general-purpose" and row["description"] == "read two" and row["model"] == run.model
    assert row["tokens"] == run.tokens == 330 and row["cost_usd"] == run.cost_usd
    assert row["result"] == run.result and row["result"].startswith("report A")
    assert row["background"] is False and row["started_at"] and row["ended_at"]
    assert row["worktree"] is None and row["branch"] is None
    assert wt_row["worktree"] == str(tmp_path / "wt") and wt_row["branch"] == "dostuff/w"


def test_subagent_compaction_stored(agents, fake_model, tmp_path, monkeypatch):
    from conftest import FakeAdapter, fake_response, fake_tool_call

    from dostuff.agent import loop as loop_module
    from dostuff.helpers.tools.project_root import use_project_root
    (tmp_path / "a.txt").write_text("alpha")
    fake_model.script("[A]",
                      fake_response(tool_calls=[fake_tool_call("read_file", {"path": "a.txt"}, "c1")],
                                    usage=_usage(10_000_000)),
                      fake_response("report A", usage=_usage()))

    async def fake_compact(history, keep_budget, notes, token_limit=None):
        return history[-1:], "agent summary"
    monkeypatch.setattr(loop_module, "compact_context", fake_compact)

    async def go():
        store = SQLiteSessionStore(str(tmp_path / "sessions.db"))
        agents.attach_store(store)
        try:
            with use_project_root(tmp_path):
                run = await agents.start(owner="s1", agent_type="general-purpose", description="d",
                                         prompt="[A] go", adapter=FakeAdapter())
                await run.done
            await agents.flush()
            return run, await store.load_agent_runs("s1"), await store.load_agent_steps(run.id, working=True), \
                await store.load_agent_steps(run.id)
        finally:
            await store.close()

    run, [row], working, transcript = asyncio.run(go())
    assert run.compaction_notes == row["compaction_notes"] == "agent summary"
    assert working == run.history
    assert [s["role"] for s in working] == ["tool", "assistant"]
    assert [s["role"] for s in transcript] == ["user", "assistant", "tool", "assistant"]


def test_delete_session_deletes_agents(tmp_path):
    async def scenario():
        store = SQLiteSessionStore(str(tmp_path / "sessions.db"))
        try:
            await store.append("s1", {"role": "user", "content": "hi"})
            await store.append("s2", {"role": "user", "content": "other"})
            await store.upsert_agent_run(_agent_row("a1", "s1"))
            await store.upsert_agent_run(_agent_row("b1", "s2"))
            await store.append_agent_step("a1", {"role": "user", "content": "x"})
            await store.append_agent_step("b1", {"role": "user", "content": "y"})
            await store.delete_session("s1")
            assert await store.load_agent_runs("s1") == [] and await store.load_agent_steps("a1") == []
            assert len(await store.load_agent_runs("s2")) == 1 and len(await store.load_agent_steps("b1")) == 1
        finally:
            await store.close()
    asyncio.run(scenario())


def test_fork_session_skips_agents(tmp_path):
    async def scenario():
        store = SQLiteSessionStore(str(tmp_path / "sessions.db"))
        try:
            await store.append("src", {"role": "user", "content": "hi"})
            await store.upsert_agent_run(_agent_row("a1", "src"))
            await store.append_agent_step("a1", {"role": "user", "content": "x"})
            await store.fork_session("src", "fork")
            assert await store.load_agent_runs("fork") == []
            assert len(await store.load_agent_runs("src")) == 1  # the source keeps its agents
        finally:
            await store.close()
    asyncio.run(scenario())
