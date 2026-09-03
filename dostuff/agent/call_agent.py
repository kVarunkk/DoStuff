from dostuff.lib.tracing import traced
from opentelemetry import trace as otel_trace
from dostuff.tools.definitions import TOOL_SCHEMAS
from litellm import acompletion
from dostuff.lib.model import MODEL

@traced("model_call")
async def call_agent(steps_history: list, system_instruction: str, tool_names: list[str] | None = None):
    if tool_names is None:
        tool_names = list(TOOL_SCHEMAS.keys())

    active_schemas = [
        TOOL_SCHEMAS[name]
        for name in tool_names
        if name in TOOL_SCHEMAS
    ]

    messages = [{"role": "system", "content": system_instruction}] + steps_history    

    async def _make_request():
        return await acompletion(
            model=MODEL,
            messages=messages,
            tools=active_schemas,
            drop_invalid_params=True 
        )

    try:
        interaction = await _make_request()
    except Exception as e:
        if "malformed_tool_call" not in str(e):
            raise
        from dostuff.helpers.ui.emit import emit
        emit("Model returned malformed JSON — retrying once...", msg_type="system")
        interaction = await _make_request()

    usage = getattr(interaction, "usage", None)
    if usage is not None:
        span = otel_trace.get_current_span()
        span.set_attribute("usage.total_tokens", getattr(usage, "total_tokens", 0))
        span.set_attribute("usage.input_tokens", getattr(usage, "prompt_tokens", 0))
        span.set_attribute("usage.output_tokens", getattr(usage, "completion_tokens", 0))

    return interaction