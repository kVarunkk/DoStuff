from dostuff.helpers.agent.get_model_token_limit import get_model_token_limit
from dostuff.lib.exceptions import ConfirmationRequired
from dostuff.lib.mcp.mcp_client import MCPClient
from dostuff.lib.memory.session_store import SessionStore
from dostuff.lib.tracing import tracer
from dostuff.helpers.agent.constants import MAX_ITERATIONS
from dostuff.helpers.agent.manage_context import compact_context
from typing import Literal, Optional, Any
from dostuff.agent.call_agent import call_agent
from opentelemetry.trace import Status, StatusCode
from dostuff.helpers.agent.append_step import append_step
import asyncio
from dostuff.agent.run_tool import run_tool
import json

async def loop(session_id: str, turn_id: str, user_text: str, dynamic_system_instruction: str, mcp_client: MCPClient,  working_history: list,   turn_type: Literal['interactive_loop', 'learning_loop', 'subagent_loop'], current_session_history: list = [], steps_history: list = [],  store: SessionStore | None = None, adapter: Optional[Any] = None) -> str:

    async def _emit(et, data):
        if adapter and hasattr(adapter, "emit"):
            try:
                adapter.emit(et, data)
            except Exception:
                pass

    iteration = 0
    last_input_tokens = 0
    token_limit = get_model_token_limit()
    context_token_threshold = int(token_limit * 0.5)
    keep_recent_token_budget = int(context_token_threshold * 0.15)
    compaction_notes = ""

    with tracer.start_as_current_span("turn") as turn_span:
        turn_span.set_attribute("session_id", session_id)
        turn_span.set_attribute("turn_type", turn_type)
        turn_span.set_attribute("turn_id", turn_id)
        turn_span.set_attribute("user_input", user_text)
        while iteration < MAX_ITERATIONS:
            iteration += 1
            with tracer.start_as_current_span("iteration") as iter_span:
                iter_span.set_attribute("iteration_number", iteration)

                # context compaction
                if last_input_tokens > context_token_threshold:
                    working_history, new_summary = await compact_context(working_history, keep_recent_token_budget)
                    compaction_notes = f"{compaction_notes}\n{new_summary}".strip()
                    with tracer.start_as_current_span("context_compaction") as compaction_span:
                        compaction_span.set_attribute("steps_after", len(working_history))
                
                # Check for user cancel via ESC
                if adapter and hasattr(adapter, "_cancelled") and adapter._cancelled:
                    await _emit("system", "Turn cancelled by user (ESC).")
                    return "Cancelled by user (ESC)."

                # agent call — with 3 retries on failure
                interaction = None
                last_err = None
                for attempt in range(3):
                    try:
                        interaction = await call_agent(steps_history=working_history, system_instruction=dynamic_system_instruction + (f"\n\n[Summary of earlier conversation]: {compaction_notes}" if compaction_notes else ""))
                        break
                    except Exception as e:
                        last_err = e
                        if attempt < 2:
                            await _emit("system", f"Retrying... ({attempt+1}/3) — {e}")
                        continue
                if interaction is None:
                    await _emit("error", f"Agent call failed after 3 retries: {last_err}")
                    return f"Failed after 3 attempts: {last_err}"

                # token tracking
                usage = getattr(interaction, "usage", None)
                if usage:
                    last_input_tokens = getattr(usage, "total_tokens", last_input_tokens)

                # CustomStreamWrapper does not expose ``choices`` in its static type,
                # although the completed interaction provides it at runtime.
                choices = getattr(interaction, "choices", None)
                if not choices:
                    iter_span.set_status(Status(StatusCode.OK))
                    continue
                message = choices[0].message
                tool_calls = getattr(message, "tool_calls", None)
                content = getattr(message, "content", None)

                if content and not tool_calls:
                    model_step = {
                        "role": "assistant",
                        "content": content,
                    }
                    await append_step(model_step, steps_history, working_history, current_session_history, session_id, store, turn_type)
                    
                    turn_span.set_attribute("outcome", "success")
                    turn_span.set_status(Status(StatusCode.OK))
                    iter_span.set_status(Status(StatusCode.OK))
                    # print(f"\n\nAgent: {content}")
                    if usage:
                        usage_dict = {
                            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                        }
                        await _emit("usage", usage_dict)
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
                await append_step(assistant_tool_step, steps_history, working_history, current_session_history, session_id, store, turn_type)
    
    
                for fn_name, fn_args, _ in function_calls:
                    await _emit("tool_call", f"{fn_name}({fn_args})")
    
                results = await asyncio.gather(
                    *(asyncio.wait_for(
                        run_tool(fn_name=fn_name, fn_args=dict(fn_args), mcp_client=mcp_client, session_id=session_id, turn_id=turn_id),
                        timeout=120.0  # 2 min per tool call; prevents hung MCP servers from blocking the turn
                    ) for fn_name, fn_args, _ in function_calls),
                    return_exceptions=True,
                )
                final_results = []
                for (fn_name, fn_args, fn_id), result in zip(function_calls, results):
                    if isinstance(result, ConfirmationRequired):
                        if adapter and hasattr(adapter, "ask"):
                            confirm = await adapter.ask(result.message, result.resume_args)
                        else:
                            if adapter and hasattr(adapter, "emit"):
                                adapter.emit("confirm", result.message)
                            # else:
                            #     print(f"\nConfirmation needed: {result.message}")
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
                    await _emit("tool_result", f"{fn_name}: {str(result)[:200]}")
                
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
            # Suppress stdout noise in TUI mode; adapter handles display
            turn_span.set_attribute("outcome", "max_iterations_exceeded")  
            turn_span.set_status(Status(StatusCode.ERROR, "max_iterations_exceeded"))
            return msg
    