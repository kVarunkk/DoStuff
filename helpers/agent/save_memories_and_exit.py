import asyncio
from helpers.memory.extract_semantic_memories import extract_memories
from helpers.memory.extract_episodic_memory import extract_episodes 
from lib.memory.semantic_memory_store import ChromaMemoryStore
from lib.memory.episodic_memory_store import ChromaEpisodicStore 

async def save_memories_and_exit(
    steps_history: list[dict],
    user_id: str,
    session_id: str,
    memory_store: ChromaMemoryStore,
    episodic_store: ChromaEpisodicStore,
) -> None:
    print("\nWE ARE NOW SAVING YOUR MEMORIES. THIS MIGHT TAKE SOME TIME...")

    async def _process_semantic_memory():
        try:
            facts = await extract_memories(steps_history)
            if facts:
                await memory_store.add(user_id, facts)
                print(f" Saved {len(facts)} semantic fact(s).")
        except Exception as e:
            print(f"⚠️ Error saving semantic memory: {e}")

    async def _process_episodic_memory():
        try:
            episodes = await extract_episodes(
                steps_history=steps_history,
                session_id=session_id,
                user_id=user_id,
            )

            print(f"\n\nEPISODES: {episodes}\n\n")
            if episodes:
                await episodic_store.add(episodes)
                print(f" Saved {len(episodes)} atomic episode(s).")
        except Exception as e:
            print(f"⚠️ Error saving episodic memory: {e}")

    await asyncio.gather(
        _process_semantic_memory(),
        _process_episodic_memory(),
        return_exceptions=True, 
    )

    print("SAVED LONG TERM & EPISODIC MEMORY. EXITING.")