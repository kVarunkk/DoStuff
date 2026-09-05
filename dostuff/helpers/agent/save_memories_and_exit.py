import asyncio
from dostuff.helpers.memory.extract_semantic_memories import extract_memories
from dostuff.helpers.memory.extract_episodic_memory import extract_episodes 
from dostuff.lib.memory.semantic_memory_store import SemanticMemoryStore
from dostuff.lib.memory.episodic_memory_store import EpisodicMemoryStore 

async def save_memories_and_exit(
    steps_history: list[dict],
    user_id: str,
    session_id: str,
    memory_store: SemanticMemoryStore,
    episodic_store: EpisodicMemoryStore,
) -> None:
    from dostuff.helpers.ui.emit import emit
    emit("\nWE ARE NOW SAVING YOUR MEMORIES. THIS MIGHT TAKE SOME TIME...", msg_type="system")
    async def _process_semantic_memory():
        from dostuff.helpers.ui.emit import emit
        try:
            facts = await extract_memories(steps_history)
            if facts:
                await memory_store.add(user_id, facts)
                emit(f" Saved {len(facts)} semantic fact(s).", msg_type="system")
        except Exception as e:
            emit(f"⚠️ Error saving semantic memory: {e}", msg_type="error")
            # print(f"⚠️ Error saving semantic memory: {e}")

    async def _process_episodic_memory():
        from dostuff.helpers.ui.emit import emit
        try:
            episodes = await extract_episodes(
                steps_history=steps_history,
                session_id=session_id,
                user_id=user_id,
            )

            emit(f"EPISODES: {episodes}", msg_type="system")
            if episodes:
                await episodic_store.add(episodes)
                emit(f" Saved {len(episodes)} atomic episode(s).", msg_type="system")
        except Exception as e:
            # print(f"⚠️ Error saving episodic memory: {e}")
            emit(f"⚠️ Error saving episodic memory: {e}", msg_type="error")

    await asyncio.gather(
        _process_semantic_memory(),
        _process_episodic_memory(),
        return_exceptions=True, 
    )

    from dostuff.helpers.ui.emit import emit
    emit("SAVED LONG TERM & EPISODIC MEMORY. EXITING.", msg_type="system")