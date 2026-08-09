import json
from lib.genai_client import get_client
import os
from dotenv import load_dotenv
from helpers.memory.format_transcript import format_transcript

load_dotenv()  
model = os.getenv("MODEL")


EXTRACTION_PROMPT = """Review this conversation history and extract durable facts worth
remembering about the user for future conversations — preferences, recurring names,
standing instructions, or personal details they shared.

Do NOT include: one-off meeting details, tool call mechanics, or anything only relevant
to this specific conversation.

Return a JSON array of objects like:
[{{"key": "name", "value": "User's name is Varun Kumawat"}}, {{"key": "favorite_color", "value": "User's favorite color is pink"}}]

Use a short, stable key per fact category (e.g. "name", "favorite_color", "meeting_preference").

If there is nothing durable worth remembering, return an empty array: []

Conversation:
{conversation}
"""

async def extract_memories(steps_history: list[dict]) -> list[dict]:
    client = get_client()

    conversation_text = format_transcript(steps_history, type="memory_update")
    prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)

    interaction = await client.interactions.create(
        model=model,
        input=prompt,
    )

    output_text = getattr(interaction, "output_text", "") or ""

    try:
        cleaned = output_text.strip().removeprefix("```json").removesuffix("```").strip()
        facts = json.loads(cleaned)
        if isinstance(facts, list):
            return [
                f for f in facts
                if isinstance(f, dict)
                and isinstance(f.get("key"), str) and f["key"].strip()
                and isinstance(f.get("value"), str) and f["value"].strip()
            ]
    except (json.JSONDecodeError, AttributeError):
        pass

    return []