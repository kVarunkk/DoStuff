
from types import SimpleNamespace


async def consume_stream(stream_iter, adapter, emit_fn):
    """Consume a litellm stream iterator; accumulate text + tool_call chunks.
    Returns a SimpleNamespace response with .choices[0].message ."""
    content_parts = []
    tool_call_chunks = {}  # id -> {name, args_parts}
    chunk_count = 0
    last_chunk = None
    stream_usage = None
    # Track usage if final chunk provides it
    # For simplicity: accumulate all delta.content; build message at end
    async for chunk in stream_iter:
        chunk_count += 1
        last_chunk = chunk
        # Capture usage-only chunk (empty choices, real usage) BEFORE skipping
        u = getattr(chunk, "usage", None)
        if u is not None:
            stream_usage = u
        # If no delta, skip content/tool accumulation (usage-only or null chunk)
        delta = None
        try:
            delta = chunk.choices[0].delta
        except Exception:
            pass
        if delta is None:
            continue
        # Reasoning / thought process chunks (OpenAI o1/o3, Gemini thinking, Anthropic extended)
        reasoning_chunk = getattr(delta, "reasoning_content", None) or getattr(delta, "thinking", None) or ""
        if reasoning_chunk and adapter and hasattr(adapter, "emit"):
            try:
                adapter.emit("partial_thought", reasoning_chunk)
            except Exception:
                pass
        # Content chunks
        text_chunk = getattr(delta, "content", None) or ""
        if text_chunk:
            content_parts.append(text_chunk)
        # Tool call chunks (streamed in partial chunks by id/index)
        raw_tool_calls = getattr(delta, "tool_calls", None)
        if raw_tool_calls:
            for tc_chunk in (raw_tool_calls if isinstance(raw_tool_calls, list) else [raw_tool_calls]):
                try:
                    fn = getattr(tc_chunk, "function", None)
                    if fn:
                        # Track by index (primary) or id (fallback)
                        idx = getattr(tc_chunk, "index", None)
                        tc_key = f"idx_{idx}" if idx is not None else (getattr(tc_chunk, "id", None) or "unknown")
                        fn_name = getattr(fn, "name", None) or ""
                        args_text = getattr(fn, "arguments", None) or ""
                        if args_text or fn_name:
                            if tc_key not in tool_call_chunks:
                                tool_call_chunks[tc_key] = {"name": fn_name, "args_text": ""}
                            # Update name if this chunk provides it (first chunk usually has it)
                            if fn_name:
                                tool_call_chunks[tc_key]["name"] = fn_name
                            if args_text:
                                tool_call_chunks[tc_key]["args_text"] += args_text
                except Exception:
                    pass
        # Emit partial text to adapter (streaming display — cumulative text)
        accumulated = "".join(content_parts)
        try:
            if adapter and hasattr(adapter, "emit") and accumulated:
                adapter.emit("partial", accumulated)
        except Exception:
            pass
    # Debug: report what we got
    # if adapter and hasattr(adapter, "emit"):
    #     adapter.emit("system", f"[stream] {chunk_count} chunks, content_len={len(''.join(content_parts))}, tool_calls={list(tool_call_chunks.keys())}")
    # Assemble final message
    full_content = "".join(content_parts)
    # Build message object
    msg = SimpleNamespace(
        content=full_content,
        role="assistant",
        tool_calls=None,
    )
    if tool_call_chunks:
        msg.tool_calls = None
        tc_list = []
        for tc_key, info in tool_call_chunks.items():
            # Convert idx_X key to a real id
            real_id = tc_key[4:] if tc_key.startswith("idx_") else tc_key
            tc_ns = SimpleNamespace(
                id=real_id,
                type="function",
                function=SimpleNamespace(
                    name=info["name"],
                    arguments=info["args_text"],
                ),
            )
            tc_ns.model_dump = lambda tc=tc_ns: {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            tc_list.append(tc_ns)
        msg.tool_calls = tc_list
    # Real response wrapper using stdlib SimpleNamespace
    choice = SimpleNamespace(
        message=msg,
        index=0,
        finish_reason="stop",
    )
    # Build real usage from stream if available
    usage = stream_usage
    if usage is not None:
        try:
            p = getattr(usage, "prompt_tokens", 0) or 0
            c = getattr(usage, "completion_tokens", 0) or 0
            t = getattr(usage, "total_tokens", 0) or (p + c)
            usage = SimpleNamespace(prompt_tokens=p, completion_tokens=c, total_tokens=t)
        except Exception:
            pass
    response = SimpleNamespace(
        choices=[choice],
        usage=usage,
        id="stream",
        model="streamed",
        object="chat.completion",
        created=0,
    )
    return response