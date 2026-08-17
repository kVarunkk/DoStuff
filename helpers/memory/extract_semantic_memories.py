import json
from helpers.memory.format_transcript import format_transcript
from litellm import acompletion, ModelResponse
from lib.model import MODEL
from lib.memory.types import SessionSemanticMemoriesExtraction

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
    conversation_text = format_transcript(steps_history, type="memory_update")
    prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)

    response = await acompletion(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format=SessionSemanticMemoriesExtraction
    )

    if isinstance(response, ModelResponse):
        raw_json_string = response.choices[0].message.content or ""
    else:
        raw_json_string = ""    

    cleaned = raw_json_string.strip().removeprefix("```json").removesuffix("```").strip()
    data = json.loads(cleaned)

    if isinstance(data, dict):
        facts = data.get("facts", [])
    elif isinstance(data, list):
        facts = data
    else:
        facts = []

    if not isinstance(facts, list):
        return []    

    try:
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