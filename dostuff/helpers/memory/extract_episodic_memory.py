import json
import time
from datetime import datetime
from dostuff.lib.memory.types import SessionEpisodicExtraction
from dostuff.helpers.memory.format_transcript import format_transcript
from litellm import acompletion, ModelResponse
from dostuff.lib.model import MODEL
from dostuff.helpers.ui.emit import emit

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
    conversation_text = format_transcript(steps_history, type="memory_update")
    prompt = EPISODIC_EXTRACTION_PROMPT.format(conversation=conversation_text)

    try:
        response = await acompletion(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format=SessionEpisodicExtraction, 
        )

        if isinstance(response, ModelResponse):
            raw_json_string = response.choices[0].message.content or ""
        else:
            raw_json_string = ""    

        cleaned = raw_json_string.strip().removeprefix("```json").removesuffix("```").strip()
        extracted_data = SessionEpisodicExtraction.model_validate_json(cleaned)
        data = extracted_data.model_dump()

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
        emit(f"⚠️ Error during episodic memory extraction: {e}", msg_type="error")
        return []