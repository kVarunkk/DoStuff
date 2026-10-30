from dostuff.helpers.agent.loop_check_cancelled import is_cancelled, loop_check_cancelled
from dostuff.helpers.agent.get_model_token_limit import get_model_token_limit
from dostuff.helpers.agent.model_cost import call_cost
from dostuff.lib.model import current_session_id, get_litellm_model
from dostuff.helpers.agent.loop_emit import loop_emit
from dostuff.lib.mcp.mcp_client import MCPClient
from dostuff.lib.tracing import content_attr, tracer
from dostuff.helpers.agent.constants import MAX_ITERATIONS
from dostuff.helpers.agent.manage_context import compact_context
from dostuff.agent.call_agent import call_agent
from opentelemetry.trace import Status, StatusCode
from dostuff.helpers.agent.append_step import append_step
import asyncio
import inspect
import random
from dataclasses import dataclass
from dostuff.agent.permissions import PLAN_MODE_INSTRUCTION, SessionPermissions
from dostuff.agent.run_tool import (
    CANCELLED_RESULT, authorize_batch, authorize_tool, dispatch_tool, post_tool_use, pre_tool_use,
)
from dostuff.helpers.hooks import HookOutcome, run_hooks
from dostuff.helpers.hooks.dispatch import report_outcome
from dostuff.helpers.hooks.events import (
    STOP_LIMIT_MESSAGE, STOP_NOTE, session_context_suffix, session_start, stop_event_for,
)
from dostuff.helpers.hooks.run_hook import payload_for
from dostuff.config import AUTO_COMPACT_THRESHOLD, get_config
from dostuff.tools.definitions import TOOL_MAP
from typing import Any, Literal
import json
import litellm
from dostuff.helpers.agent.consume_stream import StreamProgress, consume_stream as _consume_stream
from dostuff.helpers.agent.tool_output import EMPTY_RESPONSE_NUDGE, format_call_label, truncate_tool_output

TOOL_TIMEOUT_S = 120.0
RETRY_BASE_DELAY_S = 1.0
CANCELLED_RUNNING_RESULT = "Cancelled by user (may still be running)"
# Read-only tools that must still run one at a time (a subagent may prompt for approval)
_SEQUENTIAL_READ_TOOLS = frozenset({"delegate_to_subagent"})
# Tools exempt from TOOL_TIMEOUT_S: a subagent is bounded by its iteration cap, each of its own
# tool calls has the normal timeout, and its approval prompts must not count against a deadline
_UNTIMED_TOOLS = frozenset({"delegate_to_subagent"})

_RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    litellm.RateLimitError, litellm.Timeout, litellm.APIConnectionError,
    litellm.InternalServerError, litellm.ServiceUnavailableError,
    asyncio.TimeoutError, ConnectionError,
)


def classify_model_error(err: BaseException) -> Literal["retry", "compact", "fail"]:
    """Retry transient errors, compact on context overflow, fail fast on everything else (design D9)."""
    if isinstance(err, litellm.ContextWindowExceededError):
        return "compact"
    if isinstance(err, (litellm.AuthenticationError, litellm.NotFoundError, litellm.PermissionDeniedError, litellm.BadRequestError)):
        return "fail"
    if isinstance(err, _RETRYABLE_ERRORS):
        return "retry"
    status = getattr(err, "status_code", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return "retry"
    return "fail"


def _retry_after_s(err: BaseException) -> float | None:
    """The provider's Retry-After header in seconds, if the error carries one."""
    headers = getattr(err, "litellm_response_headers", None) or getattr(getattr(err, "response", None), "headers", None)
    if not headers:
        return None
    try:
        if headers.get("retry-after-ms") is not None:
            return max(float(headers["retry-after-ms"]) / 1000, 0.0)
        if headers.get("retry-after") is not None:
            return max(float(headers["retry-after"]), 0.0)
    except (TypeError, ValueError):  # HTTP-date form or garbage: fall back to backoff
        pass
    return None


def retry_delay(attempt: int, err: BaseException, max_delay: float) -> float:
    """Wait before retry `attempt` (0-based): Retry-After if given, else exponential backoff with jitter; capped at max_delay."""
    retry_after = _retry_after_s(err)
    if retry_after is not None:
        return min(retry_after, max_delay)
    return min(RETRY_BASE_DELAY_S * 2 ** attempt, max_delay) * random.uniform(0.5, 1.0)


@dataclass
class _ToolCall:
    name: str
    args: dict[str, Any]
    id: str
    error: str | None = None  # set when the call can't run (bad JSON, unavailable tool, denied)
    result: str | None = None
    hook: HookOutcome | None = None  # its PreToolUse outcome, an input to the permission check


def _parse_tool_calls(tool_calls: list[Any], tool_names: list[str] | None) -> list[_ToolCall]:
    calls: list[_ToolCall] = []
    for tool_call in tool_calls:
        if getattr(tool_call, "type", "function") != "function":
            continue
        name = tool_call.function.name
        raw = tool_call.function.arguments
        call = _ToolCall(name=name, args={}, id=tool_call.id)
        if isinstance(raw, dict):
            call.args = raw
        else:
            try:
                parsed = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError) as e:
                call.error = f"Error: arguments for '{name}' are not valid JSON ({e}). Retry the call with a valid JSON object."
                parsed = {}
            if isinstance(parsed, dict):
                call.args = parsed
            elif call.error is None:
                call.error = f"Error: arguments for '{name}' must be a JSON object, got {type(parsed).__name__}."
        if call.error is None and tool_names is not None and name not in tool_names:
            call.error = f"Error: tool '{name}' is not available here."
        calls.append(call)
    return calls


def _call_label(call: _ToolCall) -> str:
    """How a call is shown in the TUI; the todo list's JSON args would only repeat its result."""
    if call.name == "todo_write":
        return "Update todos"
    return format_call_label(call.name, call.args)


def _runs_in_parallel(name: str) -> bool:
    spec = TOOL_MAP.get(name)
    return spec is not None and spec.kind == "read" and name not in _SEQUENTIAL_READ_TOOLS


def _is_thread_tool(name: str) -> bool:
    spec = TOOL_MAP.get(name)
    return spec is not None and not inspect.iscoroutinefunction(spec.impl)


async def _wait_or_cancel(tasks: list[asyncio.Task], adapter: Any) -> bool:
    """Wait for all tasks; on ESC cancel the rest and return True."""
    pending = set(tasks)
    while pending:
        if is_cancelled(adapter):
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            return True
        _, pending = await asyncio.wait(pending, timeout=0.2)
    return False


async def _sleep_or_cancel(seconds: float, adapter: Any) -> bool:
    """Sleep in short steps; True if the user cancelled meanwhile."""
    remaining = seconds
    while remaining > 0:
        if is_cancelled(adapter):
            return True
        await asyncio.sleep(min(0.2, remaining))
        remaining -= 0.2
    return is_cancelled(adapter)


async def loop(session_id: str, turn_id: str, user_text: str, dynamic_system_instruction: str, mcp_client: MCPClient, working_history: list, turn_type: Literal['interactive_loop', 'subagent_loop'], current_session_history=None, steps_history=None, store=None, adapter=None, compaction_notes: str = "", session_prompt_tokens: int = 0, session_completion_tokens: int = 0, session_total_tokens: int = 0, last_input_tokens: int = 0, stream: bool = True, max_iterations: int = MAX_ITERATIONS, tool_names: list[str] | None = None, permissions: SessionPermissions | None = None, session_hook_context: dict[str, str] | None = None) -> str:
    """Run one agent turn. `working_history` is mutated in place (appends and compaction).

    `session_hook_context` is the caller's per-session SessionStart context (hooks spec REQ-2.4): added to
    every model call's system instruction, refreshed here on compaction, never written to history.
    """
    current_session_id.set(session_id)
    iteration = 0
    token_limit = get_model_token_limit()
    config = get_config()
    threshold_percent = getattr(config, "compact_threshold", AUTO_COMPACT_THRESHOLD)  # None: auto-compaction off (overflow still compacts)
    context_token_threshold = int(token_limit * threshold_percent / 100) if threshold_percent else None
    keep_recent_token_budget = min(int(token_limit * 0.15), 20000)
    max_tool_output_chars = config.max_tool_output_chars
    max_retries = getattr(config, "max_retries", 6)
    retry_max_delay = getattr(config, "retry_max_delay", 60.0)
    compaction_notes = compaction_notes or ""
    current_session_history = current_session_history if current_session_history is not None else []
    steps_history = steps_history if steps_history is not None else []

    async def compact(reason: str, keep_budget: int) -> bool:
        """Summarise old working steps into compaction_notes. True if anything was compacted."""
        nonlocal compaction_notes
        try:
            recent, new_summary = await compact_context(list(working_history), keep_budget, compaction_notes, token_limit)
        except Exception as e:
            await loop_emit("error", f"Compaction failed: {e}", adapter=adapter)
            return False
        if not new_summary:
            return False
        working_history[:] = recent
        compaction_notes = new_summary.strip()
        with tracer.start_as_current_span("context_compaction") as compaction_span:
            compaction_span.set_attribute("steps_after", len(working_history))
        if turn_type == "interactive_loop" and store is not None and hasattr(store, "compact_working_history"):
            try:
                await store.compact_working_history(session_id, len(working_history), compaction_notes)
            except Exception as e:
                await loop_emit("error", f"Could not save compaction: {e}", adapter=adapter)
        await loop_emit("system", f"{reason} Context compacted. Notes: {compaction_notes}", adapter=adapter)
        if session_hook_context is not None:
            # Earlier SessionStart context may be stale now; refresh it under its own key (REQ-2.4)
            await session_start("compact", session_hook_context, session_id=session_id, turn_id=turn_id,
                                turn_type=turn_type, adapter=adapter)
        return True

    async def check_cancelled() -> bool:
        return await loop_check_cancelled(steps_history, working_history, current_session_history, session_id, store, turn_type, adapter=adapter)

    # auto compaction triggers at loop.auto_compact_threshold (default 50%) of the model's token limit
    # and keeps at least 15%/20k tokens of the most recent steps; the rest is summarised into the
    # system instruction. Checked at turn start and after every model call.
    if context_token_threshold is not None and last_input_tokens > context_token_threshold:
        await compact("Token threshold exceeded.", keep_recent_token_budget)

    with tracer.start_as_current_span("turn") as turn_span:
        turn_span.set_attribute("session_id", session_id)
        turn_span.set_attribute("turn_type", turn_type)
        turn_span.set_attribute("turn_id", turn_id)
        turn_span.set_attribute("user_input", content_attr(user_text))

        max_iter = max_iterations
        nudged = False
        stop_blocks = 0  # Stop-hook continuations this turn, bounded by hooks.stop_block_limit
        while iteration < max_iter:
            iteration += 1
            with tracer.start_as_current_span("iteration") as iter_span:
                iter_span.set_attribute("iteration_number", iteration)

                if await check_cancelled():
                    return "Cancelled by user."

                # Model call with classified retries and exponential backoff
                interaction = None
                model_span = None
                attempt = 0
                compacted_for_overflow = False
                while True:
                    model_span = None
                    progress = StreamProgress()
                    try:
                        # Read per call: the mode can change mid-turn (Shift+Tab)
                        plan_mode = ((permissions.mode if permissions else None) or getattr(config, "permission_mode", "ask")) == "plan"
                        interaction, model_span = await call_agent(
                            steps_history=working_history,
                            system_instruction=dynamic_system_instruction
                            + (f"\n\n{PLAN_MODE_INSTRUCTION}" if plan_mode else "")
                            + (f"\n\n[Summary of earlier conversation]: {compaction_notes}" if compaction_notes else "")
                            + session_context_suffix(session_hook_context),
                            tool_names=tool_names,
                            stream=stream,
                        )
                        if hasattr(interaction, "__aiter__"):
                            interaction = await _consume_stream(interaction, adapter, loop_emit, progress)
                        break
                    except Exception as e:
                        if model_span is not None and model_span.is_recording():
                            model_span.record_exception(e)
                            model_span.set_status(Status(StatusCode.ERROR, str(e)))
                            model_span.end()
                        if progress.emitted_text:
                            # Don't silently replay a half-shown answer: mark it as interrupted
                            await loop_emit("stream_interrupted", str(e), adapter=adapter)
                        action = classify_model_error(e)
                        if action == "compact" and not compacted_for_overflow:
                            compacted_for_overflow = True
                            if await compact("Context window exceeded.", keep_recent_token_budget // 2):
                                continue
                        if action == "retry" and attempt < max_retries:
                            delay = retry_delay(attempt, e, retry_max_delay)
                            attempt += 1
                            await loop_emit("system", f"Retrying in {delay:.0f}s ({attempt}/{max_retries}) — {e}", adapter=adapter)
                            if await _sleep_or_cancel(delay, adapter):
                                break
                            continue
                        await loop_emit("error", f"Model call failed: {e}", adapter=adapter)
                        turn_span.set_attribute("outcome", "model_error")
                        turn_span.set_status(Status(StatusCode.ERROR, str(e)))
                        return f"Model call failed: {e}"

                if await check_cancelled():
                    return "Cancelled by user."
                if interaction is None:
                    return "Cancelled by user."

                usage_dict = None
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
                        # Prompt tokens read from the provider's prompt cache (0 if none / unsupported)
                        "cached_tokens": getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
                    }
                    # USD from litellm's price table; None if unknown (the TUI then hides the cost)
                    usage_dict["cost"] = call_cost(get_litellm_model(), usage_dict)

                # close model_span exactly once, here, before any continue/return below
                if model_span is not None and model_span.is_recording():
                    if usage_dict:
                        model_span.set_attribute("usage.prompt_tokens", usage_dict["prompt_tokens"])
                        model_span.set_attribute("usage.completion_tokens", usage_dict["completion_tokens"])
                        model_span.set_attribute("usage.total_tokens", usage_dict["total_tokens"])
                        model_span.set_attribute("usage.cached_tokens", usage_dict["cached_tokens"])
                    if content:
                        model_span.set_attribute("model.output.content", content_attr(content))
                    if tool_calls:
                        model_span.set_attribute(
                            "model.output.tool_calls",
                            content_attr([{"name": tc.function.name, "args": tc.function.arguments} for tc in tool_calls])
                        )
                    model_span.set_status(Status(StatusCode.OK))
                    model_span.end()

                current_context_tokens = 0
                if usage_dict:
                    await loop_emit("usage", usage_dict, adapter=adapter)
                    session_total_tokens += usage_dict.get("total_tokens", 0)
                    session_prompt_tokens += usage_dict.get("prompt_tokens", 0)
                    session_completion_tokens += usage_dict.get("completion_tokens", 0)
                    current_context_tokens = usage_dict.get("prompt_tokens", 0)
                    if turn_type == "interactive_loop" and store is not None and hasattr(store, "update_session_tokens"):
                        await store.update_session_tokens(session_id, session_total_tokens, session_prompt_tokens, session_completion_tokens, current_context_tokens)
                        if usage_dict["cost"] and hasattr(store, "add_session_cost"):
                            await store.add_session_cost(session_id, usage_dict["cost"])
                    percent = (current_context_tokens / token_limit * 100) if token_limit > 0 else 0
                    await loop_emit("status", {"context_window_percent": f"{percent:.1f}%", "token_limit": token_limit}, adapter=adapter)

                if content and not tool_calls:
                    model_step = {
                        "role": "assistant",
                        "content": content,
                    }
                    await append_step(model_step, steps_history, working_history, current_session_history, session_id, store, turn_type)

                    # Stop / SubagentStop: only here, on a final answer, never on cancel, errors or the
                    # iteration cap. A block keeps the turn going, at most stop_block_limit times (REQ-2.3)
                    stop_event = stop_event_for(turn_type)
                    stop = await run_hooks(stop_event, payload_for(
                        stop_event, session_id=session_id, turn_id=turn_id, turn_type=turn_type,
                        last_assistant_message=content, stop_hook_active=stop_blocks > 0,
                    ), adapter=adapter)
                    await report_outcome(stop_event, stop, adapter=adapter)
                    if stop.decision == "deny" and not stop.cancelled:
                        name = stop.decided_by or stop_event
                        if stop_blocks < config.hooks_stop_block_limit:
                            stop_blocks += 1
                            note = STOP_NOTE.format(name=name, reason=stop.reason or "keep going")
                            await append_step({"role": "user", "content": note}, steps_history, working_history, current_session_history, session_id, store, turn_type)
                            iter_span.set_status(Status(StatusCode.OK))
                            continue
                        await loop_emit("system", STOP_LIMIT_MESSAGE.format(
                            event=stop_event, name=name, limit=config.hooks_stop_block_limit), adapter=adapter)

                    turn_span.set_attribute("outcome", "success")
                    turn_span.set_status(Status(StatusCode.OK))
                    iter_span.set_status(Status(StatusCode.OK))
                    return content

                if not tool_calls:
                    # Empty response: nudge once, then give up
                    if nudged:
                        msg = "The model returned an empty response twice; ending the turn."
                        await loop_emit("error", msg, adapter=adapter)
                        turn_span.set_attribute("outcome", "empty_response")
                        turn_span.set_status(Status(StatusCode.ERROR, "empty_response"))
                        return msg
                    nudged = True
                    await append_step({"role": "user", "content": EMPTY_RESPONSE_NUDGE}, steps_history, working_history, current_session_history, session_id, store, turn_type)
                    iter_span.set_status(Status(StatusCode.OK))
                    continue
                nudged = False

                calls = _parse_tool_calls(tool_calls, tool_names)
                assistant_tool_step = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [t.model_dump() for t in tool_calls],
                }

                for call in calls:
                    await loop_emit("tool_call", _call_label(call), adapter=adapter)

                cancelled = False
                hook_ids: dict[str, Any] = {"session_id": session_id, "turn_id": turn_id, "turn_type": turn_type, "adapter": adapter}

                # PreToolUse hooks run for every call, one at a time, before any prompt: a hook deny costs
                # the user no prompt, and a rewrite is what the permission layer sees (hooks spec D2, D6)
                for call in calls:
                    if call.error is not None:
                        continue
                    call.args, call.hook = await pre_tool_use(call.name, call.args, tool_use_id=call.id, **hook_ids)
                    if call.hook.cancelled:
                        call.error = call.result = CANCELLED_RESULT
                        cancelled = True
                        break
                    if call.hook.decision == "deny":
                        call.error = call.hook.reason

                # Permission checks run first, outside the tool timeout so a slow human doesn't time the
                # call out: one prompt for all calls that need approval, else one at a time.
                pending = [c for c in calls if c.error is None] if not cancelled else []
                batch = await authorize_batch([(c.name, dict(c.args)) for c in pending], adapter=adapter, permissions=permissions, hooks=[c.hook for c in pending])
                decided = dict(zip(map(id, pending), batch)) if batch is not None else {}
                for call in calls:
                    if cancelled:
                        break
                    if call.error is None:
                        if id(call) in decided:
                            call.error = decided[id(call)]
                        else:
                            call.error = await authorize_tool(call.name, dict(call.args), adapter=adapter, permissions=permissions, hook=call.hook)
                    if call.error is not None:
                        call.result = call.error
                        await loop_emit("tool_result", f"{call.name}: {call.result}", adapter=adapter)
                    if is_cancelled(adapter):
                        cancelled = True
                        break

                def start(call: _ToolCall) -> asyncio.Task:
                    return asyncio.ensure_future(asyncio.wait_for(
                        dispatch_tool(call.name, dict(call.args), mcp_client=mcp_client, session_id=session_id, turn_id=turn_id, adapter=adapter, permissions=permissions),
                        timeout=None if call.name in _UNTIMED_TOOLS else TOOL_TIMEOUT_S,
                    ))

                async def run_batch(batch: list[_ToolCall]) -> bool:
                    tasks = [start(call) for call in batch]
                    was_cancelled = await _wait_or_cancel(tasks, adapter)
                    for call, task in zip(batch, tasks):
                        if task.cancelled():
                            call.result = CANCELLED_RUNNING_RESULT if _is_thread_tool(call.name) else CANCELLED_RESULT
                            continue
                        exc = task.exception()
                        if isinstance(exc, asyncio.TimeoutError):
                            call.result = f"Error: tool '{call.name}' timed out after {TOOL_TIMEOUT_S:.0f}s."
                        elif exc is not None:
                            call.result = f"Error: {exc}"
                        else:
                            call.result = str(task.result())
                        if not was_cancelled:
                            # PostToolUse runs here, after the dispatch task and outside TOOL_TIMEOUT_S, so the
                            # parallel read-only batch stays intact and its context reaches the tool step (D5)
                            call.result, _ = await post_tool_use(call.name, call.args, call.result, tool_use_id=call.id, **hook_ids)
                        await loop_emit("tool_result", f"{call.name}: {call.result}", adapter=adapter)
                    return was_cancelled

                runnable = [c for c in calls if c.error is None]
                if not cancelled:
                    # Read-only calls in parallel first, then mutating calls one by one, in order
                    parallel = [c for c in runnable if _runs_in_parallel(c.name)]
                    if parallel:
                        cancelled = await run_batch(parallel)
                    for call in (c for c in runnable if not _runs_in_parallel(c.name)):
                        if cancelled:
                            break
                        cancelled = await run_batch([call])

                # Keep the assistant step and every result (completed or not) so history stays valid
                await append_step(assistant_tool_step, steps_history, working_history, current_session_history, session_id, store, turn_type)
                for call in calls:
                    result_text = call.result if call.result is not None else call.error or CANCELLED_RESULT
                    result_step = {
                        "role": "tool",
                        "name": call.name,
                        "tool_call_id": call.id,
                        "content": truncate_tool_output(result_text, max_tool_output_chars),
                    }
                    await append_step(result_step, steps_history, working_history, current_session_history, session_id, store, turn_type)
                iter_span.set_status(Status(StatusCode.OK))

                if cancelled or is_cancelled(adapter):
                    await check_cancelled()
                    return "Cancelled by user."

                if context_token_threshold is not None and current_context_tokens > context_token_threshold:
                    await compact("Token threshold exceeded.", keep_recent_token_budget)
        else:
            msg = f"Reached maximum iterations ({max_iter}) without receiving a model output. Ending the agent loop."
            await loop_emit("system", msg, adapter=adapter)
            turn_span.set_attribute("outcome", "max_iterations_exceeded")
            turn_span.set_status(Status(StatusCode.ERROR, "max_iterations_exceeded"))
            return msg
    return "Cancelled by user."
