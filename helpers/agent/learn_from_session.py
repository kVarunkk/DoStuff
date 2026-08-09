from helpers.agent.constants import  SYSTEM_INSTRUCTION_FOR_SELF_LEARNING
from helpers.memory.format_transcript import format_transcript
from lib.mcp.mcp_client import MCPClient
import uuid
from lib.tracing import  session_id_var, turn_id_var
from agent.loop import loop

async def learn_from_session(steps_history: list[dict], mcp_client: MCPClient, session_id: str):
    conversation_text = format_transcript(steps_history, type="skill_update")
    if not conversation_text.strip() or len(conversation_text) < 100:
        print("[Learning Loop] Transcript too short for skill reflection.")
        return

    user_prompt = f"""
Analyze the completed session transcript below and update the agent's
persistent procedural skills.

<session_transcript>
{conversation_text}
</session_transcript>

Proceed step-by-step:
1. List existing skills in the `skills/` directory.
2. Read relevant existing `SKILL.md` files if updating.
3. If the workflow involves executable logic, write it as a real script under
   skills/<skill-name>/scripts/ — never as prose inside the SKILL.md.
4. If you wrote or modified a script in step 3, run it via run_code against a
   realistic input to confirm it actually works, per the Script Validation rule
   in your system instructions. Fix and re-test if it fails. Do not proceed to
   step 5 with an unvalidated script.
5. Synthesize and write the new or updated SKILL.md, referencing the validated
   script by its exact path if one was created.
6. End with a brief text summary of what was created or updated — including
   confirmation that any bundled script was executed and passed validation, or
   an explanation of why no skill changes were needed."""

    await learn_from_session_loop(user_prompt, system_instruction=SYSTEM_INSTRUCTION_FOR_SELF_LEARNING, mcp_client=mcp_client, session_id=session_id)
   

async def learn_from_session_loop(user_prompt: str, system_instruction: str, mcp_client: MCPClient, session_id: str):
    working_history = [
        {
            "type": "user_input",
            "content": [{"type": "text", "text": user_prompt}],
        }
    ]
    session_id_var.set(session_id)
    turn_id = str(uuid.uuid4())
    turn_id_var.set(turn_id)

    await loop(session_id, turn_id, user_prompt, system_instruction, mcp_client, working_history, "learning_loop")