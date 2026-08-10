from google.genai._gaos.lib.compat_errors import BadRequestError
from lib.tracing import traced
from google.genai._gaos.types.interactions.step import StepParam
from lib.genai_client import get_client
from opentelemetry import trace as otel_trace
import os
from dotenv import load_dotenv
from tools.definitions import TOOL_SCHEMAS

load_dotenv()  
model = os.getenv("MODEL")

@traced("model_call")
async def call_agent(steps_history: list[StepParam], system_instruction: str, tool_names: list[str] | None = None):
    client = get_client()

    if tool_names is None:
        tool_names = list(TOOL_SCHEMAS.keys())

    active_schemas = [
        {"type": "function", **TOOL_SCHEMAS[name]}
        for name in tool_names
        if name in TOOL_SCHEMAS
    ]    

    async def _make_request():
        return await client.interactions.create(
            model=model,
            input=steps_history,
            # tools=[{"type": "function", **schema} for schema in tool_schemas],
            tools=active_schemas,
            store=False,
            system_instruction=system_instruction,
        )

    try:
        interaction = await _make_request()
    except BadRequestError as e:
        if "malformed_tool_call" not in str(e):
            raise
        print("Model returned malformed JSON — retrying once...")
        interaction = await _make_request()

    usage = getattr(interaction, "usage", None)
    if usage is not None:
        span = otel_trace.get_current_span()
        span.set_attribute("usage.total_tokens", getattr(usage, "total_tokens", 0))
        span.set_attribute("usage.input_tokens", getattr(usage, "total_input_tokens", 0))
        span.set_attribute("usage.output_tokens", getattr(usage, "total_output_tokens", 0))

    return interaction