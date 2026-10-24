import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass
class StreamProgress:
    """What a stream has shown the user so far; lets the loop mark an interrupted stream."""
    emitted_text: bool = False


def _emit(adapter: Any, event_type: str, data: Any) -> None:
    if adapter is not None and hasattr(adapter, "emit"):
        try:
            adapter.emit(event_type, data)
        except Exception:
            pass


def _is_cancelled(adapter: Any) -> bool:
    event = getattr(adapter, "_cancel_event", None)
    return bool(getattr(adapter, "_cancelled", False) or (event is not None and event.is_set()))


async def consume_stream(stream_iter, adapter, emit_fn=None, progress: StreamProgress | None = None):
    """Consume a litellm stream; accumulate text, reasoning and tool-call chunks.

    Emits cumulative `partial` (answer) and `partial_thought` (reasoning) text, then
    `stream_end`. Returns a SimpleNamespace shaped like a non-streaming response, keeping
    the provider's tool-call ids and finish_reason.
    """
    progress = progress if progress is not None else StreamProgress()
    content_parts: list[str] = []
    thought_parts: list[str] = []
    tool_call_chunks: dict[str, dict[str, str]] = {}  # stream key -> {id, name, args_text}
    finish_reason: str | None = None
    stream_usage = None
    try:
        async for chunk in stream_iter:
            if _is_cancelled(adapter):
                break
            # Usage-only chunks have empty choices
            u = getattr(chunk, "usage", None)
            if u is not None:
                stream_usage = u
            try:
                choice = chunk.choices[0]
            except (AttributeError, IndexError, TypeError):
                continue
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            # Reasoning chunks (OpenAI o-series, Gemini thinking, Anthropic extended thinking)
            reasoning_chunk = getattr(delta, "reasoning_content", None) or getattr(delta, "thinking", None) or ""
            if isinstance(reasoning_chunk, str) and reasoning_chunk:
                thought_parts.append(reasoning_chunk)
                _emit(adapter, "partial_thought", "".join(thought_parts))
            text_chunk = getattr(delta, "content", None) or ""
            if text_chunk:
                content_parts.append(text_chunk)
                progress.emitted_text = True
                _emit(adapter, "partial", "".join(content_parts))
            raw_tool_calls = getattr(delta, "tool_calls", None)
            if raw_tool_calls:
                for tc_chunk in (raw_tool_calls if isinstance(raw_tool_calls, list) else [raw_tool_calls]):
                    fn = getattr(tc_chunk, "function", None)
                    idx = getattr(tc_chunk, "index", None)
                    chunk_id = getattr(tc_chunk, "id", None)
                    # Chunks of one call share an index; only the first usually carries the id
                    key = f"idx_{idx}" if idx is not None else (chunk_id or f"anon_{len(tool_call_chunks)}")
                    entry = tool_call_chunks.setdefault(key, {"id": "", "name": "", "args_text": ""})
                    if chunk_id and not entry["id"]:
                        entry["id"] = chunk_id
                    if fn is not None:
                        if getattr(fn, "name", None):
                            entry["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            entry["args_text"] += fn.arguments
    finally:
        _emit(adapter, "stream_end", None)

    msg = SimpleNamespace(content="".join(content_parts), role="assistant", tool_calls=None)
    tc_list = []
    for info in tool_call_chunks.values():
        if not info["name"]:
            continue
        tc_ns = SimpleNamespace(
            id=info["id"] or f"call_{uuid.uuid4().hex[:24]}",
            type="function",
            function=SimpleNamespace(name=info["name"], arguments=info["args_text"]),
        )
        tc_ns.model_dump = lambda tc=tc_ns: {
            "id": tc.id,
            "type": tc.type,
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        tc_list.append(tc_ns)
    if tc_list:
        msg.tool_calls = tc_list

    usage = None
    if stream_usage is not None:
        p = getattr(stream_usage, "prompt_tokens", 0) or 0
        c = getattr(stream_usage, "completion_tokens", 0) or 0
        t = getattr(stream_usage, "total_tokens", 0) or (p + c)
        # prompt_tokens_details carries cached_tokens (cheaper; used for the cost)
        usage = SimpleNamespace(prompt_tokens=p, completion_tokens=c, total_tokens=t,
                                prompt_tokens_details=getattr(stream_usage, "prompt_tokens_details", None))
    choice_ns = SimpleNamespace(
        message=msg,
        index=0,
        finish_reason=finish_reason or ("tool_calls" if tc_list else "stop"),
    )
    return SimpleNamespace(
        choices=[choice_ns],
        usage=usage,
        id="stream",
        model="streamed",
        object="chat.completion",
        created=0,
    )
