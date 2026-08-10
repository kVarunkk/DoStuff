from helpers.agent.get_model_token_limit import get_model_token_limit
from lib.exceptions import ConfirmationRequired
from lib.mcp.mcp_client import MCPClient
from lib.memory.session_store import SessionStore
from lib.tracing import tracer
from helpers.agent.constants import MAX_ITERATIONS
from helpers.agent.manage_context import compact_context
from typing import Literal
from agent.call_agent import call_agent
from opentelemetry.trace import Status, StatusCode
from helpers.agent.append_step import append_step
from helpers.agent.extract_text import extract_text
import asyncio
from agent.run_tool import run_tool


async def loop(session_id: str, turn_id: str, user_text: str, dynamic_system_instruction: str, mcp_client: MCPClient,  working_history: list,   turn_type: Literal['interactive_loop', 'learning_loop', 'subagent_loop'], current_session_history: list = [], steps_history: list = [],  store: SessionStore | None = None) -> str:

    iteration = 0
    last_input_tokens = 0
    token_limit = await get_model_token_limit()
    context_token_threshold = int(token_limit * 0.8)
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

                # context compaction if needed
                if last_input_tokens > context_token_threshold:
                    working_history, new_summary = await compact_context(working_history, keep_recent_token_budget)
                    compaction_notes = f"{compaction_notes}\n{new_summary}".strip()
                    with tracer.start_as_current_span("context_compaction") as compaction_span:
                        compaction_span.set_attribute("steps_after", len(working_history))
                
                # agent call  
                interaction = await call_agent(steps_history=working_history, system_instruction=dynamic_system_instruction + (f"\n\n[Summary of earlier conversation]: {compaction_notes}" if compaction_notes else ""))
                usage = getattr(interaction, "usage", None)
                if usage:
                    last_input_tokens = getattr(usage, "total_input_tokens", last_input_tokens)

                interaction_steps = getattr(interaction, "steps", None)
    
                if not interaction_steps:
                    iter_span.set_status(Status(StatusCode.OK))
                    continue
    
                for step in interaction_steps:
                    dumped = step.model_dump()
                    await append_step(dumped, steps_history, working_history, current_session_history, session_id, store, turn_type)
    
                last_step = interaction_steps[-1]

                if getattr(last_step, "type", None) == "model_output":
                    final_text = extract_text(last_step)
                    if final_text is not None:
                        turn_span.set_attribute("outcome", "success")
                        turn_span.set_status(Status(StatusCode.OK))
                        iter_span.set_status(Status(StatusCode.OK))
                        print(f"\n\nAgent: {final_text}")
                        return final_text
    
                function_calls = [
                    (
                        getattr(step, "name", None),
                        getattr(step, "arguments", None) or {},
                        getattr(step, "id", None),
                    )
                    for step in interaction_steps
                    if getattr(step, "type", None) == "function_call"
                ]
    
                if not function_calls:
                    iter_span.set_status(Status(StatusCode.OK)) 
                    continue
    
                for fn_name, fn_args, _ in function_calls:
                    print(f"-> Calling local tool: {fn_name}({fn_args})")
    
                results = await asyncio.gather(
                    *(run_tool(fn_name=fn_name, fn_args=dict(fn_args), mcp_client=mcp_client, session_id=session_id, turn_id=turn_id) for fn_name, fn_args, _ in function_calls),
                    return_exceptions=True,
                )
                final_results = []
                for (fn_name, fn_args, fn_id), result in zip(function_calls, results):
                    if isinstance(result, ConfirmationRequired):
                        print(f"\nConfirmation needed: {result.message}")
                        confirm = await asyncio.to_thread(input, "Allow this? [y/n]: ")
                
                        if confirm.strip().lower() == "y":
                            resumed_args = {**fn_args, **result.resume_args}
                            result = await run_tool(fn_name=fn_name, fn_args=resumed_args, mcp_client=mcp_client, session_id=session_id, turn_id=turn_id)
                        else:
                            result = "Error: User declined to allow this action."
                
                    elif isinstance(result, Exception):
                        result = f"Error: {result}"
                
                    final_results.append((fn_name, fn_id, result))
                
                for fn_name, fn_id, result in final_results:
                    result_step = {
                        "name": fn_name,
                        "result": result,
                        "id": fn_id,
                        "type": "function_result",
                    }
                    await append_step(result_step, steps_history, working_history, current_session_history, session_id, store, turn_type)
                    iter_span.set_status(Status(StatusCode.OK))
    
        else:
            msg = f"Reached maximum iterations ({MAX_ITERATIONS}) without receiving a model output. Ending the agent loop."
            print(msg)
            turn_span.set_attribute("outcome", "max_iterations_exceeded")  
            turn_span.set_status(Status(StatusCode.ERROR, "max_iterations_exceeded"))
            return msg
    