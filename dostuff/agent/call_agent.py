from dostuff.tools.definitions import TOOL_SCHEMAS
from litellm import acompletion
from dostuff.lib.model import MODEL
from dostuff.lib.tracing import tracer
from opentelemetry.trace import Status, StatusCode

async def call_agent(steps_history: list, system_instruction: str, tool_names: list[str] | None = None, stream: bool = False):
    if tool_names is None:
        tool_names = list(TOOL_SCHEMAS.keys())
    active_schemas = [TOOL_SCHEMAS[name] for name in tool_names if name in TOOL_SCHEMAS]
    messages = [{"role": "system", "content": system_instruction}] + steps_history

    span = tracer.start_span("model_call")
    span.set_attribute("model.name", MODEL)
    span.set_attribute("model.stream", stream)
    last_msg = steps_history[-1] if steps_history else None
    if last_msg:
        span.set_attribute("model.input.last_message_role", last_msg.get("role", ""))
        span.set_attribute("model.input.last_message", str(last_msg.get("content") or last_msg.get("tool_calls") or "")[:2000])
    span.set_attribute("model.input.step_count", len(steps_history))

    try:
        interaction = await acompletion(
            model=MODEL,
            messages=messages,
            tools=active_schemas,
            drop_invalid_params=True,
            stream=stream,
            stream_options={"include_usage": True},
            thinking_config={"include_thoughts": True},
        )
        return interaction, span
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise

    
