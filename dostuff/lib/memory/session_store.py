import copy
from abc import ABC, abstractmethod
from typing import Any
import json
import aiosqlite

class SessionStore(ABC):
    """Abstract interface for persisting agent conversation state.

    Swap the in-memory implementation for a DB-backed one later without
    touching any calling code.
    """

    @abstractmethod
    async def load(self, session_id: str) -> list[dict[str, Any]]:
        """Return the full steps_history for a session, or [] if none exists."""
        ...

    @abstractmethod
    async def save(self, session_id: str, steps_history: list[dict[str, Any]]) -> None:
        """Overwrite the full steps_history for a session."""
        ...

    @abstractmethod
    async def append(self, session_id: str, step: dict[str, Any]) -> None:
        """Append a single step. Preferred over save() for per-step checkpointing."""
        ...


class InMemorySessionStore(SessionStore):
    """Simple dict-backed store. Data is lost when the process exits.
    Swap for a DB-backed SessionStore later; interface stays the same.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, Any]]] = {}

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        return copy.deepcopy(self._sessions.get(session_id, []))

    async def save(self, session_id: str, steps_history: list[dict[str, Any]]) -> None:
        self._sessions[session_id] = copy.deepcopy(steps_history)

    async def append(self, session_id: str, step: dict[str, Any]) -> None:
        self._sessions.setdefault(session_id, []).append(copy.deepcopy(step))



class SQLiteSessionStore(SessionStore):
    """SQLite-backed store for persisting agent conversation state.
    
    Data persists across process restarts. Uses WAL mode for better concurrency.
    """

    def __init__(self, db_path: str = "sessions.db") -> None:
        self.db_path = db_path
        self._initialized = False

    async def _init_db(self) -> None:
        """Initialize table schema and WAL mode if not already done."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode for higher concurrency and performance
            await db.execute("PRAGMA journal_mode=WAL;")
            
            # Create table with auto-incrementing step_order to preserve sequence
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS session_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    step_order INTEGER NOT NULL,
                    step_data TEXT NOT NULL
                );
                """
            )
            # Index for fast lookup by session_id and sorting by step_order
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_id_order 
                ON session_steps(session_id, step_order);
                """
            )
            # Sessions metadata table (includes working_dir for resume)
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    working_dir TEXT,
                    created_at TEXT
                );
                """
            )
            await db.commit()

        self._initialized = True

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        """Return the full steps_history for a session, ordered chronologically."""
        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT step_data 
                FROM session_steps 
                WHERE session_id = ? 
                ORDER BY step_order ASC
                """,
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [json.loads(row[0]) for row in rows]

    async def save(self, session_id: str, steps_history: list[dict[str, Any]]) -> None:
        """Overwrite the full steps_history for a session inside a transaction."""
        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("BEGIN TRANSACTION;")
            try:
                # Clear existing steps for this session
                await db.execute(
                    "DELETE FROM session_steps WHERE session_id = ?",
                    (session_id,),
                )
                
                # Bulk insert new history
                records = [
                    (session_id, idx, json.dumps(step))
                    for idx, step in enumerate(steps_history)
                ]
                await db.executemany(
                    """
                    INSERT INTO session_steps (session_id, step_order, step_data)
                    VALUES (?, ?, ?)
                    """,
                    records,
                )
                await db.commit()
            except Exception:
                await db.rollback()
                raise

    async def append(self, session_id: str, step: dict[str, Any]) -> None:
        """Append a single step to the session history."""
        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            # Determine the next step_order index
            async with db.execute(
                "SELECT COALESCE(MAX(step_order), -1) + 1 FROM session_steps WHERE session_id = ?",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
                next_order = row[0] if row else 0

            await db.execute(
                """
                INSERT INTO session_steps (session_id, step_order, step_data)
                VALUES (?, ?, ?)
                """,
                (session_id, next_order, json.dumps(step)),
            )
            await db.commit()        
    async def save_session_meta(self, session_id: str, working_dir: str | None = None) -> None:
        import datetime, os
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO sessions (session_id, working_dir, created_at) VALUES (?, ?, ?)",
                (session_id, working_dir or os.getcwd(), datetime.datetime.now().isoformat()),
            )
            await db.execute(
                "UPDATE sessions SET working_dir = ? WHERE session_id = ?",
                (working_dir or os.getcwd(), session_id),
            )
            await db.commit()

    async def get_session_meta(self, session_id: str) -> dict | None:
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT session_id, working_dir FROM sessions WHERE session_id = ?", (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {"session_id": row[0], "working_dir": row[1]}
                return None

    def get_session_meta_sync(self, session_id: str) -> dict | None:
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.execute("SELECT session_id, working_dir FROM sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            return {"session_id": row[0], "working_dir": row[1]}
        return None

    async def list(self) -> list[tuple[str, str]]:
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT session_id, working_dir FROM sessions") as cursor:
                rows = await cursor.fetchall()
                return [(row[0], row[1] or "") for row in rows]
