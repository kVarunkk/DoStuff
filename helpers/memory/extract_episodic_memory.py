import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from lib.memory.types import SessionEpisodicExtraction
from lib.genai_client import get_client
from helpers.memory.format_transcript import format_transcript

load_dotenv()
model = os.getenv("MODEL")

EPISODIC_EXTRACTION_PROMPT = """Review this conversation transcript and extract significant, discrete moments worth remembering as episodic memories for future sessions.

Focus ONLY on extracting:
1. DECISION: Key architectural, technical, or procedural choices made.
2. PREFERENCE: Explicit user preferences, coding styles, or constraints expressed.
3. CORRECTION: Mistakes caught or direct corrections given during the interaction.
4. EVENT / LEARNING: Unexpected errors, tool failures, and how they were resolved.

Do NOT extract generic chatter, greetings, or basic informational Q&A.
For each extracted moment, include the immediate 2-3 conversational turns in `context_window` to preserve exact framing.

If there are no significant episodic moments, return an empty array: []

Conversation:
{conversation}
"""

async def extract_episodes(
    steps_history: list[dict], 
    session_id: str, 
    user_id: str, 
) -> list[dict]:
    """Extracts atomic episodic memory cards formatted ready for ChromaDB storage."""
    client = get_client()

    conversation_text = format_transcript(steps_history, type="memory_update")
    prompt = EPISODIC_EXTRACTION_PROMPT.format(conversation=conversation_text)

    try:
        interaction = await client.interactions.create(
            model=model,
            input=prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": SessionEpisodicExtraction.model_json_schema(),
            },
        )

        output_text = getattr(interaction, "output_text", "") or ""

        cleaned = output_text.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(cleaned)

        if isinstance(data, dict):
            episodes = data.get("episodes", [])
        elif isinstance(data, list):
            episodes = data
        else:
            episodes = []
        
        if not isinstance(episodes, list):
            return []

        now_epoch = int(time.time())
        now_iso = datetime.utcnow().isoformat() + "Z"

        formatted_records = []
        for index, ep in enumerate(episodes):
            if not isinstance(ep, dict) or not ep.get("anchor_event"):
                continue

            # Dense Vector Text (What gets embedded)
            vector_text = f"- EVENT TYPE: {ep.get('event_type')}\n  - ANCHOR EVENT: {ep.get('anchor_event')}\n  - SUMMARY: {ep.get('summary')}"

            metadata = {
                "user_id": user_id,
                "session_id": session_id,
                "created_at_epoch": now_epoch,
                "created_at_iso": now_iso,
                "event_type": ep.get("event_type", "EVENT"),
                "anchor_event": ep.get("anchor_event", ""),
                "summary": ep.get("summary", ""),
                "tags": json.dumps(ep.get("tags", [])),
                "context_window": json.dumps(ep.get("context_window", []))
            }

            formatted_records.append({
                "id": f"ep_{session_id}_{index}_{now_epoch}",
                "vector_text": vector_text,
                "metadata": metadata
            })

        return formatted_records

    except (json.JSONDecodeError, AttributeError, Exception) as e:
        print(f"⚠️ Error during episodic memory extraction: {e}")
        return []