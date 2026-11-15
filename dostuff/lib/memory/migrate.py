"""Bring the memories saved under the `chroma` backend (`<data>/chroma`) into memory.db, once.

memory.db is where the sqlite and hybrid backends keep facts and episodes. Before hybrid became the default, memories
lived in the Chroma store alone, so the first start on memory.db copies them over: every fact and episode of every
user, with its id and timestamps. Rows memory.db already has are left as they are, the Chroma store is left untouched
(switching back to `memory.backend: chroma` still finds it), and a `meta` row records the import so a memory forgotten
afterwards never comes back. Copying reads Chroma's metadata only, so it works on an index too damaged to search.
"""
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from dostuff.lib.memory import chroma
from dostuff.lib.memory.semantic_memory_store import FactScope, fact_id
from dostuff.lib.memory.sqlite_memory_store import SQLiteEpisodeStore, SQLiteFactStore, _Db, _enqueue

log = logging.getLogger(__name__)

IMPORT_KEY = "chroma_import"


@dataclass
class ImportResult:
    """What `import_from_chroma` did. `status`: "imported" (this call copied rows; counts say how many, 0 is
    possible), "done" (an earlier start did), "none" (there is no Chroma store), "needs-chromadb" (there is one, but
    chromadb isn't installed to read it), "failed" (`error` says why; it is tried again next start)."""
    status: str
    facts: int = 0
    episodes: int = 0
    error: str = ""

    def message(self) -> str:
        """What to tell the user, or "" when there is nothing to say."""
        if self.status == "imported" and (self.facts or self.episodes):
            return (f"Moved {self.facts} saved facts and {self.episodes} episodes into the new memory store "
                    "(memory.db). The old store is kept as it was.")
        if self.status == "needs-chromadb":
            return ("Memories saved by an earlier version are in the old Chroma store and aren't loaded: run "
                    f"`{chroma.MEMORY_INSTALL_HINT}` and restart to bring them in.")
        if self.status == "failed":
            return (f"Couldn't bring in the memories saved by an earlier version ({self.error}). They are kept and "
                    "the next start tries again; `dostuff memory repair` tries now.")
        return ""


def import_done(db_path: str) -> bool:
    with _Db(db_path).connect() as conn:
        return conn.execute("SELECT 1 FROM meta WHERE key = ?", (IMPORT_KEY,)).fetchone() is not None


def import_from_chroma(db_path: str, chroma_path: str, *, queue_for_index: bool) -> ImportResult:
    """Copy the Chroma store's facts and episodes into memory.db unless that was done before (module docstring).
    `queue_for_index`: also queue the new rows for the hybrid backend's embedding index."""
    if import_done(db_path):
        return ImportResult("done")
    if not (Path(chroma_path) / "chroma.sqlite3").exists():
        return ImportResult("none")
    if not chroma.chroma_available():
        return ImportResult("needs-chromadb")
    try:
        facts, episodes = _read_chroma(chroma_path)
        return _write(db_path, facts, episodes, queue_for_index)
    except Exception as e:  # noqa: BLE001 - a failed import must not stop dostuff from starting; it retries
        log.warning("importing memories from %s failed: %s", chroma_path, e)
        return ImportResult("failed", error=str(e))


def _read_chroma(chroma_path: str) -> tuple[list[tuple], list[tuple]]:
    client = chroma.get_chroma_client(chroma_path)
    names = {c.name for c in client.list_collections()}

    def metas(name: str) -> list[tuple[str, dict]]:
        if name not in names:
            return []
        got = client.get_collection(name).get(include=["metadatas"])
        return [(str(i), m) for i, m in zip(got.get("ids") or [], got.get("metadatas") or []) if m]

    facts = []
    for _, m in metas("semantic_memory"):
        user_id, key, value = str(m.get("user_id") or ""), str(m.get("key") or "").strip(), str(m.get("value") or "")
        scope: FactScope = "project" if m.get("scope") == "project" else "user"
        working_dir = str(m.get("working_dir") or "") or None
        if not user_id or not key or not value.strip() or (scope == "project" and not working_dir):
            continue
        facts.append((fact_id(user_id, key, scope, working_dir), user_id, scope,
                      working_dir if scope == "project" else None, key, value, str(m.get("source") or "unknown"),
                      int(m.get("created_at_epoch") or 0), int(m.get("updated_at_epoch") or 0)))
    episodes = []
    for episode_id, m in metas("episodic_memory"):
        if not m.get("user_id"):
            continue
        tags = m.get("tags")
        episodes.append((episode_id, m.get("user_id"), m.get("session_id"), m.get("working_dir"), m.get("event_type"),
                         m.get("anchor_event"), m.get("summary"), tags if isinstance(tags, str) else "[]",
                         m.get("created_at_epoch"), m.get("created_at_iso")))
    return facts, episodes


def _write(db_path: str, facts: list[tuple], episodes: list[tuple], queue_for_index: bool) -> ImportResult:
    """One transaction: the rows, their index queue entries and the done marker all land, or none does."""
    with _Db(db_path).connect() as conn:
        conn.execute("BEGIN IMMEDIATE")  # the write lock before the check, so two starts can't both import
        if conn.execute("SELECT 1 FROM meta WHERE key = ?", (IMPORT_KEY,)).fetchone():
            return ImportResult("done")  # another dostuff started at the same moment and won
        added_facts = [row[0] for row in facts if conn.execute(
            "INSERT OR IGNORE INTO facts (id, user_id, scope, working_dir, key, value, source, created_at_epoch, "
            "updated_at_epoch) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", row).rowcount]
        added_episodes = [row[0] for row in episodes if conn.execute(
            "INSERT OR IGNORE INTO episodes (id, user_id, session_id, working_dir, event_type, anchor_event, summary, "
            "tags, created_at_epoch, created_at_iso) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row).rowcount]
        if queue_for_index:
            _enqueue(conn, SQLiteFactStore.OUTBOX_KIND, added_facts)
            _enqueue(conn, SQLiteEpisodeStore.OUTBOX_KIND, added_episodes)
        conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (IMPORT_KEY, json.dumps(
            {"facts": len(added_facts), "episodes": len(added_episodes), "at": int(time.time())})))
    return ImportResult("imported", len(added_facts), len(added_episodes))
