from abc import ABC, abstractmethod
from typing import Any
import datetime
import json
import os
import aiosqlite
from dostuff.helpers.agent.file_mentions import strip_attachments
from dostuff.helpers.memory.shell_step import parse_shell_step

# PRAGMA user_version of the newest schema; bump and add a step to _migrate() to change it.
SCHEMA_VERSION = 3


class SessionStore(ABC):
    """Abstract interface for persisting agent conversation state."""

    @abstractmethod
    async def load(self, session_id: str) -> list[dict[str, Any]]:
        """Return the full steps_history (audit trail) for a session, or [] if none exists."""
        ...

    @abstractmethod
    async def append(self, session_id: str, step: dict[str, Any]) -> None:
        """Append a single step to the audit trail and the working history."""
        ...

    @abstractmethod
    async def load_working_history(self, session_id: str) -> list[dict[str, Any]]:
        """The steps the model currently sees (not yet compacted)."""
        ...

    @abstractmethod
    async def compact_working_history(self, session_id: str, keep_last: int, compaction_notes: str) -> None:
        """Mark all but the newest `keep_last` working steps compacted and store the summary."""
        ...

    @abstractmethod
    async def clear_session(self, session_id: str) -> None:
        """Forget everything the model sees: steps, working history, notes and token counters."""
        ...

    @abstractmethod
    async def update_session_tokens(self, session_id: str, total_tokens: int = 0, prompt_tokens: int = 0, completion_tokens: int = 0, last_input_tokens: int = 0) -> None:
        """Persist session token count and last-used timestamp."""
        ...

    @abstractmethod
    async def add_session_cost(self, session_id: str, cost_usd: float) -> None:
        """Add the USD cost of one model call to the session's total."""
        ...

    @abstractmethod
    async def save_session_meta(self, session_id: str, working_dir: str | None = None, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0, compaction_notes: str = "", last_input_tokens: int = 0) -> None:
        """Persist session meta (cwd, tokens, compaction notes)."""
        ...


def _now() -> str:
    return datetime.datetime.now().isoformat()


class SQLiteSessionStore(SessionStore):
    """SQLite-backed store for persisting agent conversation state.

    One aiosqlite connection per store instance (WAL mode); call close() when done.

    Tables:
      session_steps  — append-only audit trail shown on resume
      working_steps  — what the model sees; compaction flags rows instead of rewriting them
      sessions       — per-session meta (cwd, token counters, compaction notes)
    """

    def __init__(self, db_path: str = "sessions.db") -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            conn = aiosqlite.connect(self.db_path)
            # aiosqlite's worker thread is non-daemon: if close() is ever skipped it would block
            # interpreter exit. Committed WAL data is safe either way.
            thread = getattr(conn, "_thread", None)
            if thread is not None:
                thread.daemon = True
            db = await conn
            await db.execute("PRAGMA journal_mode=WAL;")
            await self._migrate(db)
            self._db = db
        return self._db

    async def close(self) -> None:
        if self._db is not None:
            db, self._db = self._db, None
            await db.close()

    @staticmethod
    async def _migrate(db: aiosqlite.Connection) -> None:
        async with db.execute("PRAGMA user_version") as cur:
            row = await cur.fetchone()
        version = int(row[0]) if row else 0
        if version >= SCHEMA_VERSION:
            return

        # Every step is idempotent, so one pass takes any older DB to the newest schema.
        # v0 → v1 covers DBs created by dostuff <= 0.1.6 (no user_version); v2 adds `title`, v3 `cost_usd`.
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
        await db.execute("CREATE INDEX IF NOT EXISTS idx_session_id_order ON session_steps(session_id, step_order);")
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                working_dir TEXT,
                created_at TEXT,
                total_tokens INTEGER DEFAULT 0,
                last_used TEXT
            );
            """
        )
        for column in (
            "prompt_tokens INTEGER DEFAULT 0",
            "completion_tokens INTEGER DEFAULT 0",
            "total_tokens INTEGER DEFAULT 0",
            "compaction_notes TEXT DEFAULT ''",
            "working_history TEXT DEFAULT ''",
            "last_used TEXT",
            "last_input_tokens INTEGER DEFAULT 0",
            "title TEXT DEFAULT ''",
            "cost_usd REAL DEFAULT 0",
        ):
            try:
                await db.execute(f"ALTER TABLE sessions ADD COLUMN {column};")
            except aiosqlite.OperationalError:
                pass  # column already exists
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS working_steps (
                session_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                step_json TEXT NOT NULL,
                compacted INTEGER DEFAULT 0,
                PRIMARY KEY (session_id, seq)
            );
            """
        )
        # Explode the legacy working_history JSON blobs into rows. The column is emptied, not
        # dropped, so an older dostuff can still open the DB.
        async with db.execute("SELECT session_id, working_history FROM sessions WHERE working_history IS NOT NULL AND working_history != ''") as cur:
            legacy = await cur.fetchall()
        for session_id, blob in legacy:
            try:
                steps = json.loads(blob)
            except (json.JSONDecodeError, TypeError):
                steps = []
            if isinstance(steps, list):
                await db.executemany(
                    "INSERT OR IGNORE INTO working_steps (session_id, seq, step_json) VALUES (?, ?, ?)",
                    [(session_id, seq, json.dumps(step)) for seq, step in enumerate(steps)],
                )
            await db.execute("UPDATE sessions SET working_history = '' WHERE session_id = ?", (session_id,))
        await db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        await db.commit()

    async def _ensure_session_row(self, db: aiosqlite.Connection, session_id: str) -> None:
        await db.execute(
            "INSERT OR IGNORE INTO sessions (session_id, working_dir, created_at, last_used) VALUES (?, ?, ?, ?)",
            (session_id, os.getcwd(), _now(), _now()),
        )

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        """Return the full steps_history for a session, ordered chronologically."""
        db = await self._conn()
        async with db.execute(
            "SELECT step_data FROM session_steps WHERE session_id = ? ORDER BY step_order ASC",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [json.loads(row[0]) for row in rows]

    async def append(self, session_id: str, step: dict[str, Any]) -> None:
        """Append one step: one INSERT into the audit trail, one into the working history."""
        db = await self._conn()
        payload = json.dumps(step)
        # The row is created with the first step, so launches that never send anything leave no session
        await self._ensure_session_row(db, session_id)
        await db.execute(
            "INSERT INTO session_steps (session_id, step_order, step_data) "
            "SELECT ?, COALESCE(MAX(step_order), -1) + 1, ? FROM session_steps WHERE session_id = ?",
            (session_id, payload, session_id),
        )
        await db.execute(
            "INSERT INTO working_steps (session_id, seq, step_json) "
            "SELECT ?, COALESCE(MAX(seq), -1) + 1, ? FROM working_steps WHERE session_id = ?",
            (session_id, payload, session_id),
        )
        await db.commit()

    async def load_working_history(self, session_id: str) -> list[dict[str, Any]]:
        db = await self._conn()
        async with db.execute(
            "SELECT step_json FROM working_steps WHERE session_id = ? AND compacted = 0 ORDER BY seq ASC",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [json.loads(row[0]) for row in rows]

    async def compact_working_history(self, session_id: str, keep_last: int, compaction_notes: str) -> None:
        db = await self._conn()
        async with db.execute(
            "SELECT seq FROM working_steps WHERE session_id = ? AND compacted = 0 ORDER BY seq DESC LIMIT 1 OFFSET ?",
            (session_id, max(keep_last, 0)),
        ) as cur:
            row = await cur.fetchone()
        if row is not None:
            await db.execute(
                "UPDATE working_steps SET compacted = 1 WHERE session_id = ? AND seq <= ?",
                (session_id, row[0]),
            )
        await self._ensure_session_row(db, session_id)
        await db.execute(
            "UPDATE sessions SET compaction_notes = ?, last_used = ? WHERE session_id = ?",
            (compaction_notes, _now(), session_id),
        )
        await db.commit()

    async def clear_session(self, session_id: str) -> None:
        db = await self._conn()
        await db.execute("DELETE FROM session_steps WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM working_steps WHERE session_id = ?", (session_id,))
        await db.execute(
            "UPDATE sessions SET prompt_tokens = 0, completion_tokens = 0, total_tokens = 0, "
            "last_input_tokens = 0, cost_usd = 0, compaction_notes = '', working_history = '', last_used = ? WHERE session_id = ?",
            (_now(), session_id),
        )
        await db.commit()

    async def save_session_meta(self, session_id: str, working_dir: str | None = None, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0, compaction_notes: str = "", last_input_tokens: int = 0) -> None:
        db = await self._conn()
        await self._ensure_session_row(db, session_id)
        await db.execute(
            "UPDATE sessions SET working_dir = ?, prompt_tokens = ?, completion_tokens = ?, total_tokens = ?, last_used = ?, compaction_notes = ?, last_input_tokens = ? WHERE session_id = ?",
            (working_dir or os.getcwd(), prompt_tokens, completion_tokens, total_tokens, _now(), compaction_notes, last_input_tokens, session_id),
        )
        await db.commit()

    async def get_session_meta(self, session_id: str) -> dict | None:
        db = await self._conn()
        async with db.execute(
            "SELECT session_id, working_dir, prompt_tokens, completion_tokens, total_tokens, last_input_tokens, last_used, compaction_notes, title, cost_usd FROM sessions WHERE session_id = ?", (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row:
            return {"session_id": row[0], "working_dir": row[1], "prompt_tokens": row[2] or 0, "completion_tokens": row[3] or 0, "total_tokens": row[4] or 0, "last_input_tokens": row[5] or 0, "last_used": row[6], "compaction_notes": row[7] or "", "title": row[8] or "", "cost_usd": row[9] or 0.0}
        return None

    async def add_session_cost(self, session_id: str, cost_usd: float) -> None:
        db = await self._conn()
        await self._ensure_session_row(db, session_id)
        await db.execute("UPDATE sessions SET cost_usd = COALESCE(cost_usd, 0) + ? WHERE session_id = ?", (cost_usd, session_id))
        await db.commit()

    async def update_session_tokens(self, session_id: str, total_tokens: int = 0, prompt_tokens: int = 0, completion_tokens: int = 0, last_input_tokens: int = 0) -> None:
        db = await self._conn()
        await self._ensure_session_row(db, session_id)
        await db.execute(
            "UPDATE sessions SET total_tokens = ?, prompt_tokens = ?, completion_tokens = ?, last_input_tokens = ?, last_used = ? WHERE session_id = ?",
            (total_tokens, prompt_tokens, completion_tokens, last_input_tokens, _now(), session_id),
        )
        await db.commit()

    async def list_sessions(self, working_dir: str | None = None, include_empty: bool = False) -> list[dict[str, Any]]:
        """Sessions newest first, with step count and title; optionally only those in working_dir.

        The title is the one set with /rename, else the first user prompt. Sessions with no steps
        (left by older versions, which created a row on every launch) are skipped unless include_empty.
        """
        db = await self._conn()
        async with db.execute(
            """
            SELECT s.session_id, s.working_dir, s.last_used, s.created_at, s.title, COUNT(st.id),
                   (SELECT step_data FROM session_steps
                    WHERE session_id = s.session_id AND step_data LIKE '{"role": "user"%'
                    ORDER BY step_order ASC LIMIT 1)
            FROM sessions s LEFT JOIN session_steps st ON st.session_id = s.session_id
            GROUP BY s.session_id
            ORDER BY s.last_used DESC
            """
        ) as cursor:
            rows = await cursor.fetchall()
        target = _norm_dir(working_dir) if working_dir else None
        sessions = []
        for session_id, wd, last_used, created_at, title, steps, first_user in rows:
            if not steps and not include_empty:
                continue
            if target is not None and _norm_dir(wd or "") != target:
                continue
            sessions.append({
                "session_id": session_id,
                "working_dir": wd or "",
                "last_used": last_used or "",
                "created_at": created_at or "",
                "steps": steps,
                "title": title or _session_title(first_user),
                "renamed": bool(title),
            })
        return sessions

    async def set_title(self, session_id: str, title: str) -> None:
        """Name a session (shown by `dostuff sessions`); an empty title goes back to the first prompt."""
        db = await self._conn()
        await self._ensure_session_row(db, session_id)
        await db.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (title.strip(), session_id))
        await db.commit()

    async def fork_session(self, source_id: str, new_id: str, title: str | None = None) -> None:
        """Copy a session (steps, working history, notes, token counters, cost) into new_id.

        Without a title the fork is named "<source title or first prompt> (fork)". Undo
        checkpoints and todos are keyed by session id and are not copied.
        """
        db = await self._conn()
        async with db.execute("SELECT title FROM sessions WHERE session_id = ?", (source_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            raise ValueError(f"No session '{source_id}'")
        if title is None:
            async with db.execute(
                "SELECT step_data FROM session_steps WHERE session_id = ? AND step_data LIKE '{\"role\": \"user\"%' "
                "ORDER BY step_order ASC LIMIT 1",
                (source_id,),
            ) as cur:
                first_user = await cur.fetchone()
            base = row[0] or _session_title(first_user[0] if first_user else None)
            title = f"{base} (fork)" if base else "(fork)"
        now = _now()
        await db.execute(
            "INSERT INTO sessions (session_id, working_dir, created_at, last_used, prompt_tokens, completion_tokens, "
            "total_tokens, last_input_tokens, compaction_notes, working_history, title, cost_usd) "
            "SELECT ?, working_dir, ?, ?, prompt_tokens, completion_tokens, total_tokens, last_input_tokens, "
            "compaction_notes, '', ?, cost_usd FROM sessions WHERE session_id = ?",
            (new_id, now, now, title.strip(), source_id),
        )
        await db.execute(
            "INSERT INTO session_steps (session_id, step_order, step_data) "
            "SELECT ?, step_order, step_data FROM session_steps WHERE session_id = ?",
            (new_id, source_id),
        )
        await db.execute(
            "INSERT INTO working_steps (session_id, seq, step_json, compacted) "
            "SELECT ?, seq, step_json, compacted FROM working_steps WHERE session_id = ?",
            (new_id, source_id),
        )
        await db.commit()

    async def delete_session(self, session_id: str) -> None:
        """Remove a session's steps, working history and meta row."""
        db = await self._conn()
        for table in ("session_steps", "working_steps", "sessions"):
            await db.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
        await db.commit()

    async def list(self) -> list[tuple[str, str, str | None]]:
        db = await self._conn()
        async with db.execute("SELECT session_id, working_dir, last_used FROM sessions ORDER BY last_used DESC") as cursor:
            rows = await cursor.fetchall()
        return [(row[0], row[1] or "", row[2] or "") for row in rows]


def _norm_dir(path: str) -> str:
    """Comparable form of a directory (case-insensitive on Windows, no trailing slash)."""
    return os.path.normcase(os.path.normpath(os.path.abspath(path))) if path else ""


def _session_title(step_json: str | None) -> str:
    """One-line label for a session: its first user prompt (`!cmd` for a shell command)."""
    if not step_json:
        return ""
    try:
        content = json.loads(step_json).get("content")
    except (json.JSONDecodeError, AttributeError):
        return ""
    if not isinstance(content, str):
        return ""
    shell = parse_shell_step(content)
    if shell is not None:
        content = "!" + shell[0]
    return " ".join(strip_attachments(content).split())
