import time
from pathlib import Path
from typing import Any, Literal

from dostuff.config import get_config
from dostuff.helpers.agent.turn_context import select_injected_facts
from dostuff.lib.memory.chroma import MEMORY_INSTALL_HINT
from dostuff.lib.memory.stores import get_episodic_store, get_semantic_store

# Lower than the turn's own recall (0.25 facts, 0.30 episodes): the model chose to look, so a weaker match is
# still worth showing it (memory recall D12)
SEARCH_MIN_SIMILARITY = 0.2
MAX_RESULTS = 10


async def search_memory(query: str, kind: Literal["all", "facts", "episodes"] = "all", limit: int = 5) -> str:
    """Searches long-term memory for saved facts and past episodes that the context doesn't show.

    Your fact list holds only the newest saved facts, and each turn recalls at most a few more facts and
    episodes. Search when the lists say more are saved and the task may depend on them: before changing
    something that may have history (the database, CI, dependencies, the build, code style or project
    conventions), when an error looks familiar, or when the user asks what was decided or what they told
    you. One search per topic; don't search for facts already listed.

    Args:
        query: What to look for, in plain words, e.g. 'code style preferences' or
            'sqlite database is locked'.
        kind: 'facts', 'episodes' or 'all'.
        limit: Results per kind, 1-10.

    Returns:
        Matching facts not already in your fact list (yours and this project's) and matching
        episodes, most relevant first, each with its date; older ones may be out of date.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must be non-empty.")
    if kind not in ("all", "facts", "episodes"):
        raise ValueError("kind must be 'all', 'facts' or 'episodes'.")
    limit = max(1, min(int(limit), MAX_RESULTS))
    fact_store, episode_store = get_semantic_store(), get_episodic_store()
    if fact_store is None or episode_store is None:
        raise RuntimeError(f"Long-term memory is off (chromadb is not installed: {MEMORY_INSTALL_HINT}).")
    config = get_config()
    user_id = config.get_user_id()
    here = str(Path.cwd().resolve())
    parts = []
    if kind in ("all", "facts"):
        parts.append(await _facts(fact_store, user_id, here, query, limit))
    if kind in ("all", "episodes"):
        working_dir = here if config.episodic_scope == "project" else None
        parts.append(await _episodes(episode_store, user_id, working_dir, query, limit))
    return "\n\n".join(parts)


async def _facts(store: Any, user_id: str, here: str, query: str, limit: int) -> str:
    """Only the facts the facts step leaves out: the listed ones are in context already (D12)."""
    facts = await store.list_all(user_id, working_dir=here)
    _, overflow = select_injected_facts(facts)
    if not overflow:
        return f"Facts: All {len(facts)} saved facts are already listed in your fact list."
    wanted = {(f["scope"], f["key"]) for f in overflow}
    ranked = await store.query(user_id, query_text=query, top_k=len(facts), min_similarity=SEARCH_MIN_SIMILARITY,
                               working_dir=here)
    found = [f for f in ranked if (f.get("scope", "user"), f["key"]) in wanted][:limit]
    if not found:
        return f"Facts: nothing among the {len(overflow)} not in your fact list matches."
    lines = [f"- [{f['scope']}] {f['key']}: {f['value']} (saved {_date(f.get('updated_at_epoch'))})" for f in found]
    return f"Facts not in your fact list ({len(found)} of {len(overflow)}, most relevant first):\n" + "\n".join(lines)


async def _episodes(store: Any, user_id: str, working_dir: str | None, query: str, limit: int) -> str:
    # Relevance alone, no recency decay: each result shows its date and type, and the model weighs age (D12)
    found = await store.query(user_id=user_id, query_text=query, top_k=limit, working_dir=working_dir,
                              recency_decay_rate=0.0, min_similarity=SEARCH_MIN_SIMILARITY)
    if not found:
        return "Episodes: Nothing saved matches."
    lines = [f"- [{e.get('event_type') or 'EVENT'}, {_date(e.get('created_at_epoch'))}] {e.get('anchor_event')}\n"
             f"  {e.get('summary')}" for e in found]
    return f"Episodes ({len(found)}, most relevant first):\n" + "\n".join(lines)


def _date(epoch: Any) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(int(epoch))) if epoch else "unknown date"
