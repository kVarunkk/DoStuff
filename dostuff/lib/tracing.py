"""Tracing setup — gated by Config / env; defaults OFF.

Backward-compatible: existing `from dostuff.lib.tracing import tracer` still works.
When disabled, tracer is a NoOp tracer (spans are no-ops, no exports, no warnings).
"""
from __future__ import annotations
import os
import atexit

from dostuff.config import Config

_config = Config()
_service = (_config.tracing_service or "dostuff")

_enabled = _config.tracing_enabled
_endpoint = _config.tracing_endpoint
_service = _config.tracing_service
_exporter = _config.tracing_exporter

# Always import opentelemetry — provides a real TracerProvider or NoOp
from opentelemetry import trace
from opentelemetry.trace import NoOpTracer, TracerProvider
import contextvars

session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="")
turn_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("turn_id", default="")


def traced(span_name: str):
    """Decorator: open a span around an async function. No-op when tracing disabled."""
    def decorator(fn):
        async def wrapper(*args, **kwargs):
            if tracer is None or not hasattr(tracer, "start_as_current_span"):
                return await fn(*args, **kwargs)
            with tracer.start_as_current_span(span_name) as span:
                try:
                    result = await fn(*args, **kwargs)
                    from opentelemetry.trace import Status, StatusCode
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    from opentelemetry.trace import Status, StatusCode
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        return wrapper
    return decorator

if _enabled:
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    resource = Resource(attributes={SERVICE_NAME: str(_service or "dostuff")})
    provider = SdkTracerProvider(resource=resource)

    if _exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif _exporter in ("otlp", ""):
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=_endpoint, insecure=True))
        )
    # else: "none" → no exporter, spans are recorded but not exported

    trace.set_tracer_provider(provider)

    # Flush on exit so we don't drop pending spans
    def _flush():
        try:
            provider.force_flush(timeout_millis=2000)
        except Exception:
            pass
    atexit.register(_flush)

    tracer = trace.get_tracer("agent")
else:
    # No-op tracer: cheap, no exports, no localhost:4317 errors
    tracer = NoOpTracer()
