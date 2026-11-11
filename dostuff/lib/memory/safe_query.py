"""Vector queries that survive a Chroma HNSW-vs-metadata divergence.

Chroma stores vectors in an HNSW segment and documents/metadata in SQLite. If a row survives in
SQLite but not in the HNSW graph (see chroma-core/chroma#7032), any `query()` that walks the graph
raises `InternalError: Error executing plan: Internal error: Error finding id`. Every read and write
in this package goes through a filtered `query()`, so one orphaned row breaks recall *and* saving
until the index is rebuilt.

The policy here is the same one the save tools already follow: a degraded read is far better than a
failed one. A duplicate fact is cosmetic; a fact that cannot be saved, or memory that cannot be
recalled at all, is functional damage. So on divergence we fall back to `get()` (a plain SQLite scan
that never touches the graph) and rank candidates in Python with the collection's embedding function.
This package never sets an embedding function, so all three collections use Chroma's default
(ONNXMiniLM_L6_V2, 384 dims), which is what the fallback ranks with.

`is_divergence` is deliberately string-based: Chroma 1.5.x raises a bare `InternalError` with no
subtype or error code, so the message is the only signal available.
"""
import math
import time
from typing import Any

from dostuff.lib.memory.chroma import is_index_divergence

# Re-embedding every document costs seconds on a real collection, and a broken index stays broken
# until it is repaired, so a fallback is cached for this long instead of paid on every recall turn
FALLBACK_CACHE_SECONDS = 300.0

# Documents kept for ranking in the fallback, so a huge collection cannot stall a recall turn
FALLBACK_CANDIDATES = 200

# (collection id, where, query_text) -> (expires at, collection size, ranked result)
_fallback_cache: dict[tuple[int, str, str], tuple[float, int, dict[str, Any]]] = {}

# Collections already announced as degraded, so a broken index is reported once, not every turn
_warned: set[int] = set()


def _warn_once(collection: Any) -> None:
    """Tell the user their index is damaged, once per collection: degrading silently looks like
    a memory that keeps forgetting things."""
    collection_id = id(collection)
    if collection_id in _warned:
        return
    _warned.add(collection_id)
    name = getattr(collection, "name", "memory")
    try:
        from dostuff.helpers.ui.emit import emit
        emit(
            f"Long-term memory ({name}) is running in a degraded mode: its vector index is out of "
            f"sync with its metadata, so some entries can't be found.\n"
            f"Nothing was lost. Run `dostuff memory repair` to rebuild it.",
            "error",
        )
    except Exception:
        pass


def is_divergence(error: BaseException) -> bool:
    """True if `error` is Chroma's HNSW-vs-metadata divergence rather than a real failure."""
    return is_index_divergence(error)


def _cosine(a: Any, b: Any) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm = math.sqrt(sum(float(x) * float(x) for x in a)) * math.sqrt(sum(float(y) * float(y) for y in b))
    return dot / norm if norm else 0.0


def clear_fallback_cache() -> None:
    """Forget cached fallbacks and warnings (after a repair, the index answers queries again)."""
    _fallback_cache.clear()
    _warned.clear()


def _rank_in_python(collection: Any, query_text: str, rows: dict[str, Any], top_k: int) -> dict[str, Any]:
    """Rank rows from `get()` by cosine similarity, shaped like a `query()` result.

    Rows without a document or embedding are dropped: those are the orphaned rows that caused the
    divergence, and Chroma cannot embed them either.
    """
    documents = rows.get("documents") or []
    metadatas = rows.get("metadatas") or []
    ids = rows.get("ids") or []
    embed = getattr(collection, "_embedding_function", None)
    scored: list[tuple[float, str, Any, Any]] = []
    if embed is not None and documents:
        try:
            # Only embeds candidate documents, so the cost is bounded by the collection size cap
            query_vec = embed([query_text])[0]
            doc_vecs = embed([doc if isinstance(doc, str) else "" for doc in documents])
            for i, (row_id, doc, meta) in enumerate(zip(ids, documents, metadatas)):
                # A row with no document or no metadata is an orphan: it is what broke the index,
                # and callers assume every returned row has metadata (meta.get(...))
                if not isinstance(doc, str) or not doc or not isinstance(meta, dict):
                    continue
                scored.append((_cosine(query_vec, doc_vecs[i]), row_id, doc, meta))
        except Exception:
            scored = []
    if not scored:
        # No usable embeddings: return the readable rows unscored, still better than nothing
        scored = [(0.0, row_id, doc, meta)
                  for row_id, doc, meta in zip(ids, documents, metadatas)
                  if isinstance(doc, str) and doc and isinstance(meta, dict)]
    scored.sort(key=lambda row: row[0], reverse=True)
    top = scored[:max(top_k, 1)]
    return {
        "ids": [[row[1] for row in top]],
        "documents": [[row[2] for row in top]],
        "metadatas": [[row[3] for row in top]],
        # `query()` returns cosine *distance*; callers convert with `1.0 - distance`
        "distances": [[1.0 - row[0] for row in top]],
    }


def safe_query(collection: Any, query_text: str, top_k: int, where: dict, include: list[str] | None = None) -> dict[str, Any]:
    """Filtered vector `query()`, degrading to a metadata scan when the HNSW index is broken.

    Returns a `query()`-shaped dict (single-element lists inside the result lists) or `{}` when
    there is nothing to search. Raises only errors that are not the divergence bug.
    """
    # An empty `where` means "no filter": omit the key rather than passing {} to Chroma
    extra = {"where": where} if where else {}
    try:
        n = min(top_k, collection.count())
        if n <= 0:
            return {}
        return collection.query(query_texts=[query_text], n_results=n, **extra)
    except Exception as error:
        if not is_divergence(error):
            raise
    # Index is diverged: read the rows without touching the graph and rank them here
    _warn_once(collection)
    key = (id(collection), str(sorted(where.items()) if where else ""), query_text)
    cached = _fallback_cache.get(key)
    if cached and cached[0] > time.monotonic() and cached[1] == collection.count():
        return cached[2]
    rows = collection.get(include=include or ["documents", "metadatas"], limit=FALLBACK_CANDIDATES, **extra)
    ranked = _rank_in_python(collection, query_text, rows, top_k)
    _fallback_cache[key] = (time.monotonic() + FALLBACK_CACHE_SECONDS, collection.count(), ranked)
    return ranked
