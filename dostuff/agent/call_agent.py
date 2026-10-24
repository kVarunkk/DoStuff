from typing import Any

import litellm
from litellm import acompletion
from dostuff.tools.definitions import TOOL_SCHEMAS, default_tool_names
from dostuff.lib.model import completion_kwargs, get_provider, uses_explicit_prompt_caching
from dostuff.lib.tracing import content_attr, tracer
from opentelemetry.trace import Status, StatusCode


# Two breakpoints: the system prompt (tools + instructions, stable within a turn) and the last
# message, which moves each call, so the next call reads everything before it from the cache.
CACHE_BREAKPOINTS: list[dict[str, Any]] = [
    {"location": "message", "role": "system"},
    {"location": "message", "index": -1},
]


def _supports_reasoning(model: str) -> bool:
    try:
        return bool(litellm.supports_reasoning(model=model))
    except Exception:
        return False


def build_request_kwargs(stream: bool) -> dict[str, Any]:
    """Model, api key and only the optional params this model/stream mode can use."""
    kwargs = completion_kwargs()
    kwargs["drop_params"] = True
    kwargs["stream"] = stream
    if stream:
        kwargs["stream_options"] = {"include_usage": True}
    if _supports_reasoning(kwargs["model"]) and get_provider() in ("gemini", "vertex_ai"):
        # Ask Gemini to stream its thoughts (rendered as partial_thought)
        kwargs["thinking_config"] = {"include_thoughts": True}
    if uses_explicit_prompt_caching():
        # Fresh copies: litellm may write internal keys into the points
        kwargs["cache_control_injection_points"] = [dict(p) for p in CACHE_BREAKPOINTS]
    return kwargs


async def call_agent(steps_history: list, system_instruction: str, tool_names: list[str] | None = None, stream: bool = False):
    if tool_names is None:
        tool_names = default_tool_names()
    active_schemas = [TOOL_SCHEMAS[name] for name in tool_names if name in TOOL_SCHEMAS]
    messages = [{"role": "system", "content": system_instruction}] + steps_history
    request = build_request_kwargs(stream)

    span = tracer.start_span("model_call")
    span.set_attribute("model.name", request["model"])
    span.set_attribute("model.stream", stream)
    last_msg = steps_history[-1] if steps_history else None
    if last_msg:
        span.set_attribute("model.input.last_message_role", last_msg.get("role", ""))
        span.set_attribute("model.input.last_message", content_attr(last_msg.get("content") or last_msg.get("tool_calls") or ""))
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
