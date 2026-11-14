import time
from pathlib import Path
from typing import Any, Literal, get_args

from dostuff.config import get_config
from dostuff.helpers.agent.turn_context import select_injected_facts
from dostuff.lib.memory.chroma import MEMORY_INSTALL_HINT
from dostuff.lib.memory.stores import get_episodic_store, get_semantic_store
from dostuff.lib.memory.types import EpisodeType

# Lower than the turn's own recall (0.25 facts, 0.30 episodes): the model chose to look, so a weaker match is
# still worth showing it (memory recall D12)
SEARCH_MIN_SIMILARITY = 0.2
# Per kind. A fact is one short line, so more of them fit: one broad query over dozens of overflow facts ranked a
# wanted one 11th or lower in the live eval (S10); an episode is a few lines
FACT_RESULTS, MAX_FACT_RESULTS = 10, 20
EPISODE_RESULTS, MAX_EPISODE_RESULTS = 5, 10


async def search_memory(query: str, kind: Literal["all", "facts", "episodes"] = "all",
                        limit: int | None = None, event_type: EpisodeType | None = None) -> str:
    """Searches long-term memory for saved facts and past episodes that the context doesn't show.

    Your fact list holds only the newest saved facts, and each turn recalls at most a few more facts and
    episodes. Search when the lists say more are saved and the task may depend on them: before changing
    something that may have history (the database, CI, dependencies, the build, code style or project
    conventions), when an error looks familiar, or when the user asks what was decided or what they told
    you. Don't search for facts already listed. A broad query can miss a fact that a narrower one finds: for
    several topics (say docstrings, exports and error handling), search each one. Episodes (not facts) each have
    a type: to find a kind of episode, such as what was decided, what was learned or what the user corrected,
    pass event_type as well as the query.

    Args:
        query: What to look for, in plain words, e.g. 'code style preferences' or
            'sqlite database is locked'.
        kind: 'facts', 'episodes' or 'all'.
        limit: Results per kind: facts 10 by default, up to 20; episodes 5, up to 10.
        event_type: Episodes of one type first: DECISION, LEARNING, CORRECTION, EVENT or PREFERENCE. Use
            it when the question asks for a kind of memory (what was decided, what we learned, what the
            user corrected), whose wording may share no words with the query. The other matches follow.

    Returns:
        Matching facts not already in your fact list (yours and this project's) and matching
        episodes, most relevant first, each with its date; older ones may be out of date.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must be non-empty.")
    if kind not in ("all", "facts", "episodes"):
        raise ValueError("kind must be 'all', 'facts' or 'episodes'.")
    if event_type is not None and event_type not in get_args(EpisodeType):
        raise ValueError(f"event_type must be one of {', '.join(get_args(EpisodeType))}.")
    if event_type is not None and kind == "facts":
        raise ValueError("event_type filters episodes; use kind 'episodes' or 'all'.")
    fact_store, episode_store = get_semantic_store(), get_episodic_store()
    if fact_store is None or episode_store is None:
        raise RuntimeError(f"Long-term memory is off (chromadb is not installed: {MEMORY_INSTALL_HINT}, or set memory.backend: sqlite).")
    config = get_config()
    user_id = config.get_user_id()
    here = str(Path.cwd().resolve())
    parts = []
    if kind in ("all", "facts"):
        parts.append(await _facts(fact_store, user_id, here, query, _clamp(limit, FACT_RESULTS, MAX_FACT_RESULTS)))
    if kind in ("all", "episodes"):
        working_dir = here if config.episodic_scope == "project" else None
        parts.append(await _episodes(episode_store, user_id, working_dir, query,
                                     _clamp(limit, EPISODE_RESULTS, MAX_EPISODE_RESULTS), event_type))
    return "\n\n".join(parts)


async def _facts(store: Any, user_id: str, here: str, query: str, limit: int) -> str:
    """Only the facts the facts step leaves out: the listed ones are in context already (D12)."""
    facts = await store.list_all(user_id, working_dir=here)
    _, overflow = select_injected_facts(facts)
    if not overflow:
        return f"Facts: All {len(facts)} saved facts are already listed in your fact list."
    wanted = {(f["scope"], f["key"]) for f in overflow}
    ranked = await store.query(user_id, query_text=query, top_k=len(facts), min_similarity=getattr(store, "SEARCH_FLOOR", SEARCH_MIN_SIMILARITY),
                               working_dir=here)
    matched = [f for f in ranked if (f.get("scope", "user"), f["key"]) in wanted]
    found = matched[:limit]
    if not found:
        return f"Facts: nothing among the {len(overflow)} not in your fact list matches."
    lines = [f"- [{f['scope']}] {f['key']}: {f['value']} (saved {_date(f.get('updated_at_epoch'))})" for f in found]
    text = f"Facts not in your fact list ({len(found)} of {len(overflow)}, most relevant first):\n" + "\n".join(lines)
    if len(matched) > len(found):
        # Only when true: the cut is where a wanted fact goes missing under a broad query (S10)
        text += (f"\nMore facts match ({len(matched) - len(found)} not shown): search again with narrower words, "
                 "one topic at a time.")
    return text


async def _episodes(store: Any, user_id: str, working_dir: str | None, query: str, limit: int,
                    event_type: str | None = None) -> str:
    # Relevance alone, no recency decay: each result shows its date and type, and the model weighs age (D12)
    found = await store.query(user_id=user_id, query_text=query, top_k=limit, working_dir=working_dir,
                              recency_decay_rate=0.0,
                              min_similarity=getattr(store, "SEARCH_FLOOR", SEARCH_MIN_SIMILARITY))
    if event_type is None:
        if not found:
            return "Episodes: Nothing saved matches."
        return f"Episodes ({len(found)}, most relevant first):\n" + _episode_lines(found)
    # A type names a kind of memory, not its words: a correction worded by what happened scores under the floor
    # against "what did I correct?", so that type is ranked without it. The unfiltered results follow, so a wrong
    # type never hides what the search finds without one (memory recall t2_search benchmark, 2026-11-13).
    typed = await store.query(user_id=user_id, query_text=query, top_k=limit, working_dir=working_dir,
                              recency_decay_rate=0.0, min_similarity=-1.0, event_type=event_type)
    shown = {e["id"] for e in typed}
    others = [e for e in found if e["id"] not in shown]
    if typed:
        text = (f"{event_type} episodes ({len(typed)}, most relevant first; every {event_type} is ranked, so the "
                f"last may be unrelated):\n" + _episode_lines(typed))
    else:
        text = f"No {event_type} episodes are saved."
    if others:
        text += f"\n\nOther episodes matching the query ({len(others)}):\n" + _episode_lines(others)
    return text


def _episode_lines(episodes: list[dict]) -> str:
    return "\n".join(f"- [{e.get('event_type') or 'EVENT'}, {_date(e.get('created_at_epoch'))}] "
                     f"{e.get('anchor_event')}\n  {e.get('summary')}" for e in episodes)


def _clamp(limit: int | None, default: int, most: int) -> int:
    return default if limit is None else max(1, min(int(limit), most))


def _date(epoch: Any) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(int(epoch))) if epoch else "unknown date"
