from litellm import acompletion, ModelResponse
from dostuff.lib.model import completion_kwargs


async def is_same_episode(saved: dict, event_type: str, anchor_event: str, summary: str) -> bool:
    """Ask the model whether a new episode is `saved` described again, adding nothing (memory recall D10: the 0.6-0.85
    band, where a reworded moment and a related but different one score alike; save_episode asks only within one
    event type).

    True only on a clear "yes"; an error or any other answer is False, so the episode is saved: a duplicate costs a
    line of context, a lost episode is gone for good.
    """
    prompt = f"""Two episodes from a coding assistant's long-term memory, saved in the same project. Is the NEW one \
the same moment as the SAVED one, described again, so that it adds nothing the saved one lacks?

If the new one adds a cause, a fix, a decision, an outcome or another detail the saved one doesn't have, or is a \
different moment (even if related), it is not a duplicate.

Answer with one word: YES if the new one adds nothing, NO otherwise.

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
