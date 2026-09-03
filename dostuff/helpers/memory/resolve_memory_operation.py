import json
from litellm import acompletion, ModelResponse
from dostuff.lib.model import MODEL

async def resolve_memory_operation(user_id: str, new_fact: dict, memory_store) -> None:
    similar = await memory_store.query(user_id, new_fact["value"], top_k=3)

    if not similar:
        await memory_store.upsert_fact(user_id, new_fact)
        print(f"FACT SAVED IN LONG TERM MEMORY-> OPERATION: SIMILAR NOT FOUND, KEY: {new_fact["key"] or "unknown"}, VALUE: {new_fact["value"]}")
        return

    decision_prompt = f"""You manage a user's long-term memory. Given a NEW fact and
EXISTING related memories, decide ONE operation:
- ADD: new fact is genuinely new information, unrelated to existing ones
- UPDATE: new fact replaces/refines one of the existing ones (same topic, more/updated info)
- NOOP: new fact is already captured by an existing memory, no change needed

Return JSON: {{"operation": "ADD"|"UPDATE"|"NOOP", "replace_key": "<existing key or null>"}}

NEW FACT: {new_fact["value"]}

EXISTING MEMORIES:
{json.dumps(similar, indent=2)}
"""
    response = await acompletion(
        model=MODEL,
        messages=[
            {"role": "user", "content": decision_prompt}
        ],
    )
    
    if isinstance(response, ModelResponse):
        raw_json_string = response.choices[0].message.content or ""
    else:
        raw_json_string = ""

    cleaned = raw_json_string.strip().removeprefix("```json").removesuffix("```").strip()
    decision = json.loads(cleaned)
    
    if decision["operation"] == "NOOP":
        return
    elif decision["operation"] == "UPDATE":
        key_to_replace = decision.get("replace_key") or new_fact["key"]
        await memory_store.upsert_fact(user_id, {"key": key_to_replace, "value": new_fact["value"]})
        print(f"FACT SAVED IN LONG TERM MEMORY-> OPERATION: UPDATE, KEY: {key_to_replace}, VALUE: {new_fact["value"]}")
    else:  # ADD
        await memory_store.upsert_fact(user_id, new_fact)
        print(f"FACT SAVED IN LONG TERM MEMORY-> OPERATION: ADD, KEY: {new_fact["key"] or "unknown"}, VALUE: {new_fact["value"]}")