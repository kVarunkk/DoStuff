from lib.memory.session_store import  SessionStore
from lib.tracing import  session_id_var, turn_id_var
import asyncio
import uuid
from helpers.agent.constants import COMMANDS
from helpers.agent.print_history import print_history
from lib.memory.semantic_memory_store import MemoryStore
from helpers.agent.append_step import append_step
import copy
from lib.mcp.mcp_client import MCPClient
from agent.loop import loop
from lib.memory.episodic_memory_store import ChromaEpisodicStore

async def run_agent(session_id: str, user_id: str, store: SessionStore, memory_store: MemoryStore, episodic_store: ChromaEpisodicStore, system_instructions:str, mcp_client: MCPClient, current_session_history: list) -> None:
    session_id_var.set(session_id)
    steps_history = await store.load(session_id)
    working_history = copy.deepcopy(steps_history)

    try:
        while True:
            user_text = await asyncio.to_thread(input, "\n\nUser: ")
    
            if not user_text:
                continue
    
            command = user_text.strip().lower()
    
            if command == "/exit":
                raise
    
            if command == "/history":
                print_history(steps_history)
                continue
    
            if command == "/clear":
                steps_history = []
                await store.save(session_id, steps_history)
                print("History cleared.")
                continue
    
            if command == "/help":
                print(f"Commands: {', '.join(sorted(COMMANDS))}")
                continue
    
            turn_id = str(uuid.uuid4())
            turn_id_var.set(turn_id)
    
            user_step = {
                "role": "user",
                "content": user_text,
            }    
           
            await append_step(user_step, steps_history, working_history, current_session_history, session_id, store, 'interactive_loop')
    
            # 1. Query Semantic Memories (Facts)
            memories = await memory_store.query(user_id, query_text=user_text, top_k=5)
            memory_text = "\n".join(f"- {m['key']}: {m['value']}" for m in memories)
            
            # 2. Query Episodic Memories (Past Experiences)
            episodes = await episodic_store.query(
                user_id=user_id,
                session_id=session_id,
                query_text=user_text,
                top_k=3
            )
            
            # 3. Format Episodic Memory for Context Injection
            episodic_text = ""
            if episodes:
                formatted_episodes = []
                for ep in episodes:
                    formatted_episodes.append(
                        f"  • [{ep['event_type']}] {ep['anchor_event']}\n"
                        f"    Summary: {ep['summary']}\n"
                        f"    Date: {ep['created_at_iso']}"
                    )
                episodic_text = "\n".join(formatted_episodes)
            
            # 4. Assemble Dynamic System Instructions
            dynamic_system_instruction = system_instructions
            
            if memory_text:
                dynamic_system_instruction += f"\n\n<relevant_user_facts>\n{memory_text}\n</relevant_user_facts>"
            
            if episodic_text:
                dynamic_system_instruction += f"\n\n<past_episodes>\n{episodic_text}\n</past_episodes>"

            await loop(session_id, turn_id, user_text, dynamic_system_instruction, mcp_client, working_history, 'interactive_loop', current_session_history, steps_history, store)
    except Exception:
        raise 

           