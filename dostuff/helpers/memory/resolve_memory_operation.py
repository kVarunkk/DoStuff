import json
from typing import Any, Literal

from litellm import acompletion, ModelResponse
from dostuff.lib.model import completion_kwargs


def _parse_decision(raw: str) -> dict[str, Any]:
    cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        decision = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return {"operation": "ADD"}
    return decision if isinstance(decision, dict) else {"operation": "ADD"}


async def resolve_memory_operation(
    user_id: str, new_fact: dict, memory_store
) -> tuple[Literal["ADD", "UPDATE", "NOOP"], str]:
    """Decide ADD / UPDATE / NOOP for one new fact against similar stored ones, then apply it.

    Returns the operation applied and the key it touched (the existing key for UPDATE / NOOP).
    """
    similar = await memory_store.query(user_id, new_fact["value"], top_k=3)

    if not similar:
        await memory_store.upsert_fact(user_id, new_fact)
        return "ADD", new_fact["key"]

    existing = [{"key": m["key"], "value": m["value"]} for m in similar]
    decision_prompt = f"""You manage a user's long-term memory. Given a NEW fact and
EXISTING related memories, decide ONE operation:
- ADD: new fact is genuinely new information, unrelated to existing ones
- UPDATE: new fact replaces/refines one of the existing ones (same topic, more/updated info)
- NOOP: new fact is already captured by an existing memory, no change needed

Return JSON: {{"operation": "ADD"|"UPDATE"|"NOOP", "replace_key": "<existing key or null>"}}

NEW FACT: {new_fact.get('value', '')}

EXISTING MEMORIES:
{json.dumps(existing, indent=2)}
"""
    try:
        response = await acompletion(
            **completion_kwargs(),
            messages=[{"role": "user", "content": decision_prompt}],
        )
        raw_json_string = (response.choices[0].message.content or "") if isinstance(response, ModelResponse) else ""
    except Exception:
        raw_json_string = ""
    decision = _parse_decision(raw_json_string)
    operation = str(decision.get("operation", "ADD")).upper()

    if operation == "NOOP":
        noop_key = decision.get("replace_key")
        return "NOOP", noop_key if isinstance(noop_key, str) and noop_key else similar[0]["key"]
    if operation == "UPDATE":
        replace_key = decision.get("replace_key")
        # An UPDATE must target a key that actually exists; otherwise it is an ADD
        if isinstance(replace_key, str) and replace_key in {m["key"] for m in existing}:
            await memory_store.upsert_fact(user_id, {**new_fact, "key": replace_key})
            return "UPDATE", replace_key
    await memory_store.upsert_fact(user_id, new_fact)
    return "ADD", new_fact["key"]
