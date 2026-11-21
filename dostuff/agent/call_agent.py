from typing import Any

from litellm import acompletion
from dostuff.config import get_config
from dostuff.helpers.agent.attachments import content_text, hydrate
from dostuff.tools.definitions import TOOL_SCHEMAS, default_tool_names
from dostuff.lib.model import completion_kwargs, current_max_tokens, supports_reasoning, uses_explicit_prompt_caching
from dostuff.lib.tracing import content_attr, tracer
from opentelemetry.trace import Status, StatusCode


# Names of attachments whose stored file was gone when a request was built (REQ-1.3); the loop reports each once
unavailable_attachments: list[str] = []
_reported_unavailable: set[str] = set()


def take_unavailable_attachments() -> list[str]:
    """Missing attachments not reported yet in this process (every later request still sends the stand-in)."""
    names = [n for n in dict.fromkeys(unavailable_attachments) if n not in _reported_unavailable]
    unavailable_attachments.clear()
    _reported_unavailable.update(names)
    return names


# Two breakpoints: the system prompt (tools + instructions, stable across turns) and the last
# message, which moves each call, so the next call reads everything before it from the cache.
CACHE_BREAKPOINTS: list[dict[str, Any]] = [
    {"location": "message", "role": "system"},
    {"location": "message", "index": -1},
]


def request_max_tokens() -> tuple[int, str]:
    """The output-token cap a model call sends, and the setting it came from (for error hints); 0 = none sent.
    The running subagent type's own cap wins over loop.max_tokens (main agent, forks, untyped runs)."""
    if own := current_max_tokens.get():
        return own, "the agent type's max_tokens:"
    return int(getattr(get_config(), "max_tokens", 0) or 0), "loop.max_tokens"


def build_request_kwargs(stream: bool) -> dict[str, Any]:
    """Model, api key and only the optional params this model/stream mode can use."""
    kwargs = completion_kwargs()
    kwargs["drop_params"] = True
    kwargs["stream"] = stream
    if stream:
        kwargs["stream_options"] = {"include_usage": True}
    config = get_config()
    max_tokens, _ = request_max_tokens()
    if max_tokens:
        kwargs["max_tokens"] = max_tokens  # litellm maps it per provider (max_output_tokens on the Responses route)
    effort = config.reasoning_effort
    if effort != "auto" and supports_reasoning(kwargs["model"]):
        # litellm maps this to each provider's own knob (Anthropic budget, Gemini thinking budget, OpenAI effort);
        # "off" is its "none". For Gemini it also decides whether thoughts are streamed.
        kwargs["reasoning_effort"] = "none" if effort == "off" else effort
    elif supports_reasoning(kwargs["model"]) and kwargs["model"].startswith(("gemini/", "vertex_ai/")):
        # Ask Gemini to stream its thoughts (rendered as partial_thought)
        kwargs["thinking_config"] = {"include_thoughts": True}
    if uses_explicit_prompt_caching():
        # Fresh copies: litellm may write internal keys into the points
        kwargs["cache_control_injection_points"] = [dict(p) for p in CACHE_BREAKPOINTS]
    return kwargs


def _as_blocks(content: Any) -> list[Any]:
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": content}] if content else []


def _is_blank(content: Any) -> bool:
    """No text worth sending: None, "", whitespace, or a block list made only of such text blocks."""
    if isinstance(content, list):
        return all(isinstance(b, dict) and b.get("type") == "text" and not str(b.get("text") or "").strip()
                   for b in content)
    if content is None or isinstance(content, str):
        return not (content or "").strip()
    return False


def _join_user_content(first: Any, second: Any) -> Any:
    if isinstance(first, list) or isinstance(second, list):
        return _as_blocks(first) + _as_blocks(second)  # an image block must survive, so no flattening to text
    return "\n\n".join(text for text in (first, second) if text)


def provider_messages(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The history as a provider may see it. The steps themselves are left alone: the store and every reader need
    them apart, while a request needs valid turns.

    `meta` is dostuff's own (the turn-context marker), so it is dropped. Adjacent user steps become one message,
    because Perplexity and strict chat templates (vLLM's Gemma, Mistral) reject two in a row, and a turn's context
    step always sits right before its prompt. Same history, same result, so the cached prefix holds.

    An assistant step that only calls tools goes out with `content: None`, not "": litellm turns an empty
    text into a "[System: Empty message content sanitised ...]" block on Anthropic routes, which the model
    then reads as its own words and starts repeating. Stored steps keep their "", so resumed sessions benefit too.
    """
    messages: list[dict[str, Any]] = []
    for step in steps:
        message = {k: v for k, v in step.items() if k != "meta"} if "meta" in step else step
        if (message.get("role") == "assistant" and message.get("tool_calls") and message.get("content") is not None
                and _is_blank(message["content"])):
            message = {**message, "content": None}
        if messages and message.get("role") == "user" and messages[-1].get("role") == "user":
            messages[-1] = {**messages[-1], "content": _join_user_content(messages[-1].get("content"), message.get("content"))}
        else:
            messages.append(message)
    return messages


async def call_agent(steps_history: list, system_instruction: str, tool_names: list[str] | None = None, stream: bool = False):
    if tool_names is None:
        tool_names = default_tool_names()
    active_schemas = [TOOL_SCHEMAS[name] for name in tool_names if name in TOOL_SCHEMAS]
    # History holds attachment refs; only the request carries their bytes (CON-004)
    messages = [{"role": "system", "content": system_instruction}] + hydrate(provider_messages(steps_history),
                                                                             missing=unavailable_attachments)
    request = build_request_kwargs(stream)

    span = tracer.start_span("model_call")
    span.set_attribute("model.name", request["model"])
    span.set_attribute("model.stream", stream)
    last_msg = steps_history[-1] if steps_history else None
    if last_msg:
        span.set_attribute("model.input.last_message_role", last_msg.get("role", ""))
        span.set_attribute("model.input.last_message", content_attr(content_text(last_msg.get("content")) or last_msg.get("tool_calls") or ""))
    span.set_attribute("model.input.step_count", len(steps_history))

    try:
        interaction = await acompletion(
            messages=messages,
            tools=active_schemas or None,
            **request,
        )
        return interaction, span
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise
