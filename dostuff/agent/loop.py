from dostuff.helpers.agent.get_model_token_limit import get_model_token_limit
from dostuff.lib.exceptions import ConfirmationRequired
from dostuff.lib.mcp.mcp_client import MCPClient
from dostuff.lib.tracing import tracer
from dostuff.helpers.agent.constants import MAX_ITERATIONS
from dostuff.helpers.agent.manage_context import compact_context
from dostuff.agent.call_agent import call_agent
from opentelemetry.trace import Status, StatusCode
from dostuff.helpers.agent.append_step import append_step
import asyncio
from dostuff.agent.run_tool import run_tool
import json
from dostuff.helpers.agent.consume_stream import consume_stream as _consume_stream

async def loop(session_id: str, turn_id: str, user_text: str, dynamic_system_instruction: str, mcp_client: MCPClient, working_history: list, turn_type: ..., current_session_history=None, steps_history=None, store=None, adapter=None, compaction_notes: str = "", session_prompt_tokens: int = 0, session_completion_tokens: int = 0, session_total_tokens: int = 0, last_input_tokens: int = 0, stream: bool = True) -> str:

    async def _emit(et, data):
        if adapter and hasattr(adapter, "emit"):
            try:
                adapter.emit(et, data)
            except Exception:
                pass

    iteration = 0
    token_limit = get_model_token_limit()
    context_token_threshold = int(token_limit * 0.5)
    keep_recent_token_budget = min(int(token_limit * 0.15), 20000)
    compaction_notes = compaction_notes or ""
    current_session_history = current_session_history if current_session_history is not None else []
    steps_history = steps_history if steps_history is not None else []
    
    # auto compaction triggers at 50% of the model's token limit and keeps at least 15%/20k tokens of the most recent steps in the working history. Rest are summarized into a compact summary appended to the system instruction. Manual compaction can be triggered with the /compact command, which will also produce a summary appended to the system instruction.
    if last_input_tokens > context_token_threshold:
        working_history, new_summary = await compact_context(working_history, keep_recent_token_budget, compaction_notes or "")
        compaction_notes = f"{new_summary}".strip() if new_summary else compaction_notes
        with tracer.start_as_current_span("context_compaction") as compaction_span:
            compaction_span.set_attribute("steps_after", len(working_history))
        if store is not None and hasattr(store, "save_session_meta"):
            try:
                await store.save_session_meta(session_id, prompt_tokens=session_prompt_tokens, completion_tokens=session_completion_tokens, total_tokens=session_total_tokens, compaction_notes=compaction_notes, working_history=json.dumps(working_history), last_input_tokens=last_input_tokens)
            except Exception:
                pass
        await _emit("system", "Auto compaction performed as token threshold exceeded.")
        await _emit("system", f"Context compacted. Notes: {compaction_notes}")

    with tracer.start_as_current_span("turn") as turn_span:
        turn_span.set_attribute("session_id", session_id)
        turn_span.set_attribute("turn_type", turn_type)
        turn_span.set_attribute("turn_id", turn_id)
        turn_span.set_attribute("user_input", user_text)

        while iteration < MAX_ITERATIONS:
            iteration += 1
            with tracer.start_as_current_span("iteration") as iter_span:
                iter_span.set_attribute("iteration_number", iteration)
    
                # Check for user cancel via ESC (event-based, thread-safe)
                if adapter and hasattr(adapter, "_cancel_event"):
                    if adapter._cancel_event.is_set() or adapter._cancelled:
                        adapter._cancelled = False
                        adapter._cancel_event.clear()
                        await _emit("system", "Turn cancelled by user (ESC).")
                        return "Cancelled by user (ESC)."
    
                # agent call, with 3 retries on failure (manual span)
                interaction = None
                last_err = None
                model_span = None
                for attempt in range(3):
                    model_span = None
                    try:
                        interaction, model_span = await call_agent(
                            steps_history=working_history,
                            system_instruction=dynamic_system_instruction + (f"\n\n[Summary of earlier conversation]: {compaction_notes}" if compaction_notes else ""),
                            stream=stream,
                        )
                        if hasattr(interaction, "__aiter__"):
                            interaction = await _consume_stream(interaction, adapter, _emit)
                        break
                    except Exception as e:
                        last_err = e
                        if model_span is not None and model_span.is_recording():
                            model_span.record_exception(e)
                            model_span.set_status(Status(StatusCode.ERROR, str(e)))
                            model_span.end()
                        if attempt < 2:
                            await _emit("system", f"Retrying... ({attempt+1}/3) — {e}")
                        continue
                if interaction is None:
                    await _emit("error", f"Agent call failed after 3 retries: {last_err}")
                    return f"Failed after 3 attempts: {last_err}"
    
                # Re-check cancel after call_agent returns (ESC may have fired during the call)
                if adapter and hasattr(adapter, "_cancel_event"):
                    if adapter._cancel_event.is_set() or adapter._cancelled:
                        adapter._cancelled = False
                        adapter._cancel_event.clear()
                        await _emit("system", "Turn cancelled by user (ESC).")
                        return "Cancelled by user (ESC)."

                usage_dict = None
                # token tracking
                usage = getattr(interaction, "usage", None)
                choices = getattr(interaction, "choices", None)
                message = choices[0].message if choices else None
                tool_calls = getattr(message, "tool_calls", None) if message else None
                content = getattr(message, "content", None) if message else None

                if usage:
                    usage_dict = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                    }
    
                # close model_span exactly once, here, before any continue/return below
                if model_span is not None and model_span.is_recording():
                    if usage_dict:
                        model_span.set_attribute("usage.prompt_tokens", usage_dict["prompt_tokens"])
                        model_span.set_attribute("usage.completion_tokens", usage_dict["completion_tokens"])
                        model_span.set_attribute("usage.total_tokens", usage_dict["total_tokens"])
                    if content:
                        model_span.set_attribute("model.output.content", str(content)[:2000])
                    if tool_calls:
                        model_span.set_attribute(
                            "model.output.tool_calls",
                            str([{"name": tc.function.name, "args": tc.function.arguments} for tc in tool_calls])[:2000]
                        )    
                    model_span.set_status(Status(StatusCode.OK))
                    model_span.end()
    
                if usage_dict:
                    await _emit("usage", usage_dict)
                    session_total_tokens += usage_dict.get("total_tokens", 0)
                    session_prompt_tokens += usage_dict.get("prompt_tokens", 0)
                    session_completion_tokens += usage_dict.get("completion_tokens", 0)
                    current_context_tokens = usage_dict.get("prompt_tokens", 0)
                    if store is not None and hasattr(store, "update_session_tokens"):
                        await store.update_session_tokens(session_id, session_total_tokens, session_prompt_tokens, session_completion_tokens, current_context_tokens)
                    percent = (current_context_tokens / token_limit * 100) if token_limit > 0 else 0
                    await _emit("status", {"context_window_percent": f"{percent:.1f}%", "token_limit": token_limit})
    
                if not choices:
                    iter_span.set_status(Status(StatusCode.OK))
                    continue
    
                if content and not tool_calls:
                    model_step = {
                        "role": "assistant",
                        "content": content,
                    }
                    await append_step(model_step, steps_history, working_history, current_session_history, session_id, store, turn_type)
    
                    turn_span.set_attribute("outcome", "success")
                    turn_span.set_status(Status(StatusCode.OK))
                    iter_span.set_status(Status(StatusCode.OK))
                    return content
    
                if not tool_calls:
                    iter_span.set_status(Status(StatusCode.OK))
                    continue
    
                function_calls = []
                for tool_call in tool_calls:
                    if tool_call.type == "function":
                        fn_name = tool_call.function.name
                        fn_args = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                        fn_id = tool_call.id
                        function_calls.append((fn_name, fn_args, fn_id))
    
                assistant_tool_step = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [t.model_dump() for t in tool_calls]
                }
    
                for fn_name, fn_args, _ in function_calls:
                    await _emit("tool_call", f"{fn_name}({fn_args})")
    
                results = await asyncio.gather(
                    *(asyncio.wait_for(
                        run_tool(fn_name=fn_name, fn_args=dict(fn_args), mcp_client=mcp_client, session_id=session_id, turn_id=turn_id),
                        timeout=120.0
                    ) for fn_name, fn_args, _ in function_calls),
                    return_exceptions=True,
                )
    
                # Check cancel after tool execution (ESC may have fired during tool calls)
                if adapter and hasattr(adapter, "_cancel_event"):
                    if adapter._cancel_event.is_set() or adapter._cancelled:
                        adapter._cancelled = False
                        adapter._cancel_event.clear()
                        await _emit("system", "Turn cancelled by user (ESC).")
                        return "Cancelled by user (ESC)."
    
                final_results = []
                for (fn_name, fn_args, fn_id), result in zip(function_calls, results):
                    if isinstance(result, ConfirmationRequired):
                        if adapter and hasattr(adapter, "ask"):
                            confirm = await adapter.ask(result.message, result.resume_args)
                        else:
                            if adapter and hasattr(adapter, "emit"):
                                adapter.emit("confirm", result.message)
                            confirm = await asyncio.to_thread(input, "Allow this? [y/n]: ")
                            confirm = confirm.strip().lower() == "y"
    
                        if confirm:
                            resumed_args = {**fn_args, **result.resume_args}
                            result = await run_tool(fn_name=fn_name, fn_args=resumed_args, mcp_client=mcp_client, session_id=session_id, turn_id=turn_id)
                        else:
                            result = "Error: User declined to allow this action."
    
                    elif isinstance(result, Exception):
                        result = f"Error: {result}"
    
                    final_results.append((fn_name, fn_id, result))
                    await _emit("tool_result", f"{fn_name}: {str(result)}")

                # append the assistant step with tool calls to the history before appending the tool results, so that there is no history corruption if the loop crashes after the assistant step but before the tool results are appended
                await append_step(assistant_tool_step, steps_history, working_history, current_session_history, session_id, store, turn_type)    
    
                for fn_name, fn_id, result in final_results:
                    result_step = {
                        "role": "tool",
                        "name": fn_name,
                        "tool_call_id": fn_id,
                        "content": str(result)
                    }
                    await append_step(result_step, steps_history, working_history, current_session_history, session_id, store, turn_type)
                    iter_span.set_status(Status(StatusCode.OK))
        else:
            msg = f"Reached maximum iterations ({MAX_ITERATIONS}) without receiving a model output. Ending the agent loop."
            if adapter and hasattr(adapter, "emit"):
                adapter.emit("system", msg)
            turn_span.set_attribute("outcome", "max_iterations_exceeded")
            turn_span.set_status(Status(StatusCode.ERROR, "max_iterations_exceeded"))
            return msg
    
