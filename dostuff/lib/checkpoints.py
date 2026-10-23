"""File checkpoints: snapshot files before the agent's file tools change them, so /undo can restore them.

One checkpoint per turn, under `<data_dir>/checkpoints/<session_id>/<time_ns>_<turn_id>/`:
`manifest.json` maps each touched file to its content before the turn's first change (a blob
named by sha256, or "did not exist") and its hash after the latest change. /undo restores the
newest checkpoint and removes it, so repeated /undo walks back turn by turn.

Only write_file / edit_file / delete_file are tracked; changes made by shell commands are not.
"""
import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHECKPOINT_TOOLS = frozenset({"write_file", "edit_file", "delete_file"})
MAX_SNAPSHOT_BYTES = 5 * 1024 * 1024
MAX_CHECKPOINTS_PER_SESSION = 50
_MANIFEST = "manifest.json"
_BLOBS = "blobs"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str | None:
    return _sha(path.read_bytes()) if path.is_file() else None


@dataclass
class UndoResult:
    restored: list[str] = field(default_factory=list)  # written back to their old content
    removed: list[str] = field(default_factory=list)  # created by the agent, now deleted
    conflicts: list[str] = field(default_factory=list)  # changed since the agent touched them; left alone
    skipped: list[str] = field(default_factory=list)  # too large to snapshot
    remaining: bool = False  # the checkpoint still has files (conflicts) for `/undo force`

    @property
    def changed(self) -> bool:
        return bool(self.restored or self.removed)


class CheckpointStore:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()

    def _session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def _checkpoints(self, session_id: str) -> list[Path]:
        """This session's checkpoint dirs, oldest first."""
        session_dir = self._session_dir(session_id)
        if not session_dir.is_dir():
            return []
        return sorted((p for p in session_dir.iterdir() if p.is_dir()), key=lambda p: int(p.name.split("_", 1)[0]))

    def _turn_dir(self, session_id: str, turn_id: str, create: bool) -> Path | None:
        for p in reversed(self._checkpoints(session_id)):
            if p.name.endswith(f"_{turn_id}"):
                return p
        if not create:
            return None
        turn_dir = self._session_dir(session_id) / f"{time.time_ns()}_{turn_id}"
        (turn_dir / _BLOBS).mkdir(parents=True)
        self._prune(session_id)
        return turn_dir

    def _prune(self, session_id: str) -> None:
        for old in self._checkpoints(session_id)[:-MAX_CHECKPOINTS_PER_SESSION]:
            shutil.rmtree(old, ignore_errors=True)

    @staticmethod
    def _load(turn_dir: Path) -> dict[str, Any]:
        manifest = turn_dir / _MANIFEST
        if not manifest.is_file():
            return {"files": {}}
        return json.loads(manifest.read_text(encoding="utf-8"))

    @staticmethod
    def _save(turn_dir: Path, data: dict[str, Any]) -> None:
        tmp = turn_dir / f"{_MANIFEST}.tmp"
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, turn_dir / _MANIFEST)

    def before_change(self, session_id: str, turn_id: str, path: Path) -> None:
        """Snapshot `path` unless this turn already did (the turn's first snapshot is what /undo restores)."""
        with self._lock:
            turn_dir = self._turn_dir(session_id, turn_id, create=True)
            assert turn_dir is not None
            data = self._load(turn_dir)
            key = str(path)
            if key in data["files"]:
                return
            entry: dict[str, Any] = {"existed": path.is_file(), "blob": None}
            if path.is_file():
                if path.stat().st_size > MAX_SNAPSHOT_BYTES:
                    entry["too_large"] = True
                else:
                    content = path.read_bytes()
                    entry["blob"] = _sha(content)
                    blob = turn_dir / _BLOBS / entry["blob"]
                    if not blob.exists():
                        blob.write_bytes(content)
            data["files"][key] = entry
            self._save(turn_dir, data)

    def after_change(self, session_id: str, turn_id: str, path: Path) -> None:
        """Record the file's hash after the tool ran, so /undo can tell if someone changed it since."""
        with self._lock:
            turn_dir = self._turn_dir(session_id, turn_id, create=False)
            if turn_dir is None:
                return
            data = self._load(turn_dir)
            entry = data["files"].get(str(path))
            if entry is None:
                return
            entry["after"] = _file_sha(path)
            self._save(turn_dir, data)

    def latest_files(self, session_id: str) -> list[str]:
        """Files in the newest checkpoint (what /undo would touch)."""
        checkpoints = self._checkpoints(session_id)
        return list(self._load(checkpoints[-1])["files"]) if checkpoints else []

    def undo(self, session_id: str, force: bool = False) -> UndoResult | None:
        """Restore the newest checkpoint. None if there is nothing to undo.

        A file changed since the agent's last write is a conflict and left alone unless `force`.
        """
        with self._lock:
            checkpoints = self._checkpoints(session_id)
            if not checkpoints:
                return None
            turn_dir = checkpoints[-1]
            data = self._load(turn_dir)
            result = UndoResult()
            for key, entry in list(data["files"].items()):
                path = Path(key)
                if entry.get("too_large"):
                    result.skipped.append(key)
                    del data["files"][key]
                    continue
                if not force and "after" in entry and _file_sha(path) != entry["after"]:
                    result.conflicts.append(key)
                    continue
                if entry["existed"]:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((turn_dir / _BLOBS / entry["blob"]).read_bytes())
                    result.restored.append(key)
                elif path.exists():
                    path.unlink()
                    result.removed.append(key)
                del data["files"][key]
            if data["files"]:
                self._save(turn_dir, data)
                result.remaining = True
            else:
                shutil.rmtree(turn_dir, ignore_errors=True)
            return result


_store: CheckpointStore | None = None


def get_checkpoint_store() -> CheckpointStore:
    global _store
    if _store is None:
        from dostuff.config import get_config
        _store = CheckpointStore(get_config().data_dir / "checkpoints")
    return _store
