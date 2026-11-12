from litellm import acompletion, ModelResponse
from dostuff.lib.model import completion_kwargs


async def is_same_episode(saved: dict, event_type: str, anchor_event: str, summary: str) -> bool:
    """Ask the model whether a new episode records the same moment as `saved` (memory recall D10: the 0.6-0.85 band,
    where a reworded moment and a related but different one score alike).

    True only on a clear "yes"; an error or any other answer is False, so the episode is saved: a duplicate costs a
    line of context, a lost episode is gone for good.
    """
    prompt = f"""Two episodes from a coding assistant's long-term memory, saved in the same project. Do they record \
the SAME moment (one event, decision or lesson, described twice), or two different ones (even if related)?

Answer with one word: YES if the same moment, NO if different.

SAVED EPISODE [{saved.get('event_type', '')}]: {saved.get('anchor_event', '')}
{saved.get('summary', '')}

NEW EPISODE [{event_type}]: {anchor_event}
{summary}
"""
    try:
        response = await acompletion(**completion_kwargs(), messages=[{"role": "user", "content": prompt}])
        answer = (response.choices[0].message.content or "") if isinstance(response, ModelResponse) else ""
    except Exception:
        return False
    words = answer.strip().strip("*`\"'").split()
    return bool(words) and words[0].strip(".,:;!*").upper() == "YES"
