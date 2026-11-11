"""Rebuild a Chroma collection so its HNSW index matches its metadata again.

A row can survive in Chroma's SQLite metadata but be missing from the HNSW vector segment. Any
`query()` then raises `InternalError: Error executing plan: Internal error: Error finding id`
(chroma-core/chroma#7032). Chroma never repairs this on its own: deleting the HNSW segment folder
does not help, because the index is rebuilt from the same metadata that holds the orphan.

The repair is dump, recreate, re-add. It is destructive, so nothing is deleted until the replacement
has been built with distinct ids: if the re-add fails, the original collection is still intact.
Every rebuild also writes a JSON backup of the dumped rows first.

Two kinds of row are not carryable and are reported separately:
  - orphans: `get()` gives no document, so there is nothing to re-embed (the cause of the breakage)
  - unreadable: the row has no metadata, which Chroma refuses to mix with metadated rows
"""
import json
from pathlib import Path
from typing import Any

from dostuff.lib.memory.safe_query import clear_fallback_cache

# Rows are re-added in batches, so one huge `add` does not build a single enormous HNSW insert pass
REBUILD_BATCH_SIZE = 200

# Name of the temporary collection the replacement is built in before the original is deleted
STAGING_SUFFIX = "__rebuild"


def dump_rows(collection: Any) -> tuple[list[str], list[str], list[dict], dict[str, int]]:
    """Every re-addable row as (ids, documents, metadatas, skipped-by-reason).

    `get()` scans SQLite and never walks the HNSW graph, so it works on a diverged collection.
    A row is skipped when it has no document (an orphan: nothing to re-embed, and the cause of the
    breakage) or no metadata (Chroma refuses to mix those with metadated rows in one `add`).
    """
    rows = collection.get(include=["documents", "metadatas"])
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    orphans = missing_metadata = 0
    for row_id, document, metadata in zip(
        rows.get("ids") or [], rows.get("documents") or [], rows.get("metadatas") or []
    ):
        if not isinstance(document, str) or not document:
            orphans += 1
            continue
        if not isinstance(metadata, dict) or not metadata:
            missing_metadata += 1
            continue
        ids.append(str(row_id))
        documents.append(document)
        metadatas.append(metadata)
    return ids, documents, metadatas, {"orphans": orphans, "missing_metadata": missing_metadata}


def write_backup(collection_name: str, rows: dict[str, Any], backup_dir: Path) -> Path:
    """Write a readable copy of the whole collection (orphans included) before anything is deleted."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    path = backup_dir / f"{collection_name}.json"
    path.write_text(
        json.dumps(
            {
                "collection": collection_name,
                "ids": rows.get("ids") or [],
                "documents": rows.get("documents") or [],
                "metadatas": rows.get("metadatas") or [],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def test_current_filter(collection: Any) -> dict | None:
    """A filter the collection actually uses, so a probe tests the path real callers take."""
    for key in ("user_id", "server_name"):
        rows = collection.get(include=["metadatas"], limit=50)
        values = [
            meta[key] for meta in (rows.get("metadatas") or [])
            if isinstance(meta, dict) and isinstance(meta.get(key), str) and meta[key]
        ]
        if values:
            return {key: values[0]}
    return None


def _with_real_values(collection: Any, where: dict | None) -> dict | None:
    """Replace an empty filter value with one that exists, so the probe can actually match rows.

    `{"server_name": ""}` matches nothing and would make any collection look healthy.
    """
    if not where:
        return where
    key, value = next(iter(where.items()))
    if not isinstance(value, str) or value:
        return where
    current = test_current_filter(collection)
    return current if current and key in current else where


def check_collection(collection: Any, where: dict | None = None) -> str | None:
    """None if `collection` can answer queries, else the error message it raises.

    Two probes, because they fail independently. Resolving every stored vector (the first probe)
    is the only filter-independent test: a `limit=1` probe misses an orphan that is not the first
    row, which is how a diverged mcp_tools_index once passed as healthy. It is also why the second
    probe exists: an unfiltered `query()` can succeed where a filtered one fails.
    """
    if collection.count() <= 0:
        return None
    try:
        collection.get(include=["embeddings"])  # full scan: any orphan anywhere raises here
    except Exception as error:
        return str(error)
    filters = [where] if where else []
    current = test_current_filter(collection)
    if current and current != where:
        filters.append(current)
    for candidate in filters:
        try:
            collection.query(query_texts=["health check"], n_results=1, where=_with_real_values(collection, candidate))
        except Exception as error:
            return str(error)
    return None


def stage_collection(client: Any, name: str, ids: list[str], documents: list[str],
                     metadatas: list[dict], space: str) -> int:
    """Build the replacement under a temporary name. Returns the number of rows it holds."""
    staging = f"{name}{STAGING_SUFFIX}"
    try:
        client.delete_collection(staging)
    except Exception:
        pass
    fresh = client.create_collection(staging, metadata={"hnsw:space": space})
    for start in range(0, len(ids), REBUILD_BATCH_SIZE):
        fresh.add(
            ids=ids[start:start + REBUILD_BATCH_SIZE],
            documents=documents[start:start + REBUILD_BATCH_SIZE],
            metadatas=metadatas[start:start + REBUILD_BATCH_SIZE],
        )
    return int(fresh.count())


def rebuild_collection(client: Any, name: str, backup_dir: Path | None = None,
                       space: str = "cosine") -> dict[str, Any]:
    """Rebuild `name` in place. The original is untouched if staging fails.

    Chroma has no rename, so the swap is: build and verify a staging collection, delete the
    original, recreate it, and copy the staged rows back in batches.
    """
    collection = client.get_collection(name)
    before = collection.count()
    raw = collection.get(include=["documents", "metadatas"])
    ids, documents, metadatas, skipped = dump_rows(collection)
    backup_path = write_backup(name, raw, backup_dir) if backup_dir else None
    try:
        staged = stage_collection(client, name, ids, documents, metadatas, space)
        if staged != len(ids):
            raise RuntimeError(f"staged {staged} of {len(ids)} rows")
        client.delete_collection(name)
        fresh = client.create_collection(name, metadata={"hnsw:space": space})
        for start in range(0, len(ids), REBUILD_BATCH_SIZE):
            fresh.add(
                ids=ids[start:start + REBUILD_BATCH_SIZE],
                documents=documents[start:start + REBUILD_BATCH_SIZE],
                metadatas=metadatas[start:start + REBUILD_BATCH_SIZE],
            )
    finally:
        try:
            client.delete_collection(f"{name}{STAGING_SUFFIX}")
        except Exception:
            pass

    rebuilt = client.get_collection(name)
    # The old collection's cached fallbacks belong to an index that no longer exists
    clear_fallback_cache()
    return {
        "collection": name,
        "before": before,
        "kept": len(ids),
        "orphans": skipped["orphans"],
        "missing_metadata": skipped["missing_metadata"],
        "backup": str(backup_path) if backup_path else None,
        "error": check_collection(rebuilt),
    }
