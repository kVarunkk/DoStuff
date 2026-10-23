"""Agent-loop robustness (REQ-5.4, 6.1 – 6.9)."""
import asyncio
import json
import time
from types import SimpleNamespace

import litellm
import pytest

from dostuff.agent import loop as loop_module
from dostuff.agent.loop import (
    CANCELLED_RESULT,
    EMPTY_RESPONSE_NUDGE,
    classify_model_error,
    loop,
    truncate_tool_output,
)
from dostuff.helpers.agent.consume_stream import consume_stream
from dostuff.helpers.agent.manage_context import split_by_token_budget


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=None,
    )


def _tc(name, args, call_id):
    tc = SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=args))
    tc.model_dump = lambda: {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}
    return tc


class Adapter:
    def __init__(self):
        self.events = []
        self._cancelled = False
        self._cancel_event = asyncio.Event()

    def emit(self, event_type, data):
        self.events.append((event_type, data))

    async def ask(self, message, **kwargs):
        return "yes"


def _run(monkeypatch, responses, adapter=None, dispatch=None, tool_names=None):
    """Run loop() with call_agent replaced by a scripted list of responses."""
    calls = []

    async def fake_call_agent(steps_history, system_instruction, tool_names=None, stream=False):
        calls.append([dict(s) for s in steps_history])
        item = responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item, None

    monkeypatch.setattr(loop_module, "call_agent", fake_call_agent)
    monkeypatch.setattr(loop_module, "get_model_token_limit", lambda: 100_000)
    if dispatch is not None:
        monkeypatch.setattr(loop_module, "dispatch_tool", dispatch)
    working: list = []
    result = asyncio.run(loop(
        session_id="s", turn_id="t", user_text="hi", dynamic_system_instruction="sys",
        mcp_client=None, working_history=working, turn_type="subagent_loop",
        adapter=adapter, stream=False, max_iterations=5, tool_names=tool_names,
    ))
    return result, working, calls


def test_truncate_tool_output_keeps_head_and_tail():
    text = "H" * 600 + "M" * 1000 + "T" * 400
    out = truncate_tool_output(text, 1000)
    assert out.startswith("H" * 600) and out.endswith("T" * 400)
    assert "1000 chars truncated" in out and "read_file" in out
    assert truncate_tool_output("short", 1000) == "short"


def test_classify_model_error():
    assert classify_model_error(litellm.RateLimitError("x", "openai", "m")) == "retry"
    assert classify_model_error(litellm.AuthenticationError("x", "openai", "m")) == "fail"
    assert classify_model_error(litellm.ContextWindowExceededError("x", "m", "openai")) == "compact"
    assert classify_model_error(ValueError("x")) == "fail"


def test_malformed_tool_json_becomes_tool_error(monkeypatch):
    responses = [_msg(tool_calls=[_tc("read_file", "{not json", "c1")]), _msg(content="done")]
    result, working, _ = _run(monkeypatch, responses)
    assert result == "done"
    tool_step = next(s for s in working if s["role"] == "tool")
    assert tool_step["tool_call_id"] == "c1" and "not valid JSON" in tool_step["content"]


def test_empty_response_nudges_once_then_ends(monkeypatch):
    result, working, _ = _run(monkeypatch, [_msg(), _msg(content="ok")])
    assert result == "ok"
    assert {"role": "user", "content": EMPTY_RESPONSE_NUDGE} in working
    result, _, _ = _run(monkeypatch, [_msg(), _msg()])
    assert "empty response twice" in result


def test_retry_with_backoff_then_fail_fast(monkeypatch):
    monkeypatch.setattr(loop_module, "retry_delay", lambda *a: 0.0)
    responses = [litellm.RateLimitError("slow down", "openai", "m"), _msg(content="ok")]
    assert _run(monkeypatch, responses)[0] == "ok"
    result, _, calls = _run(monkeypatch, [litellm.AuthenticationError("bad key", "openai", "m")])
    assert result.startswith("Model call failed") and len(calls) == 1


def test_retries_stop_at_configured_max(monkeypatch):
    monkeypatch.setattr(loop_module, "retry_delay", lambda *a: 0.0)
    monkeypatch.setattr(loop_module, "get_config", lambda: SimpleNamespace(
        max_tool_output_chars=1000, max_retries=2, retry_max_delay=60.0))
    responses = [litellm.RateLimitError("slow down", "openai", "m") for _ in range(4)]
    result, _, calls = _run(monkeypatch, responses)
    assert result.startswith("Model call failed") and len(calls) == 3  # first try + 2 retries


def test_retry_delay_backoff_jitter_and_retry_after():
    err = litellm.RateLimitError("x", "openai", "m")
    for attempt, full in [(0, 1.0), (3, 8.0), (10, 60.0)]:
        assert full * 0.5 <= loop_module.retry_delay(attempt, err, 60.0) <= full
    err.response.headers["retry-after"] = "7"
    assert loop_module.retry_delay(0, err, 60.0) == 7.0
    err.response.headers["retry-after"] = "300"
    assert loop_module.retry_delay(0, err, 60.0) == 60.0
    err.response.headers["retry-after"] = "Wed, 21 Oct 2015 07:28:00 GMT"
    assert loop_module.retry_delay(0, err, 60.0) <= 1.0


def test_tool_output_truncated_before_history(monkeypatch):
    async def dispatch(name, args, **kw):
        return "x" * 50_000
    monkeypatch.setattr(loop_module, "get_config", lambda: SimpleNamespace(max_tool_output_chars=1000))
    responses = [_msg(tool_calls=[_tc("read_file", json.dumps({"path": "a"}), "c1")]), _msg(content="done")]
    _, working, _ = _run(monkeypatch, responses, dispatch=dispatch)
    tool_step = next(s for s in working if s["role"] == "tool")
    assert len(tool_step["content"]) < 1200 and "truncated" in tool_step["content"]


def test_read_only_tools_run_in_parallel(monkeypatch):
    async def dispatch(name, args, **kw):
        await asyncio.sleep(0.3)
        return name
    calls = [_tc("read_file", json.dumps({"path": str(i)}), f"c{i}") for i in range(4)]
    start = time.monotonic()
    _run(monkeypatch, [_msg(tool_calls=calls), _msg(content="done")], dispatch=dispatch)
    assert time.monotonic() - start < 1.0  # 4 × 0.3 s sequentially would be 1.2 s


def test_mutating_tools_run_in_order_after_reads(monkeypatch):
    order = []

    async def dispatch(name, args, **kw):
        order.append(args.get("path"))
        return "ok"
    calls = [
        _tc("write_file", json.dumps({"path": "w1", "content": ""}), "c1"),
        _tc("read_file", json.dumps({"path": "r1"}), "c2"),
        _tc("write_file", json.dumps({"path": "w2", "content": ""}), "c3"),
    ]
    monkeypatch.setattr(loop_module, "authorize_tool", lambda *a, **k: _none())
    _run(monkeypatch, [_msg(tool_calls=calls), _msg(content="done")], dispatch=dispatch)
    assert order == ["r1", "w1", "w2"]


async def _none():
    return None


def test_cancel_keeps_completed_results(monkeypatch):
    adapter = Adapter()
    calls = [
        _tc("read_file", json.dumps({"path": "fast"}), "c1"),
        _tc("delegate_to_subagent", json.dumps({"task": "t", "role": "r"}), "c2"),
    ]
    monkeypatch.setattr(loop_module, "authorize_tool", lambda *a, **k: _none())

    async def slow_dispatch(name, args, **kw):
        if name == "delegate_to_subagent":
            adapter._cancelled = True
            await asyncio.sleep(5)
        return f"read {args.get('path')}"
    result, working, _ = _run(monkeypatch, [_msg(tool_calls=calls)], adapter=adapter, dispatch=slow_dispatch)
    assert result == "Cancelled by user."
    roles = [s["role"] for s in working]
    assert roles == ["assistant", "tool", "tool", "assistant"]
    assert working[1]["content"] == "read fast"
    assert working[2]["content"] == CANCELLED_RESULT


def test_unavailable_tool_rejected(monkeypatch):
    responses = [_msg(tool_calls=[_tc("delete_file", json.dumps({"path": "a"}), "c1")]), _msg(content="done")]
    _, working, _ = _run(monkeypatch, responses, tool_names=["read_file"])
    assert "not available" in next(s for s in working if s["role"] == "tool")["content"]


# ── consume_stream ───────────────────────────────────────────────────────────

def test_consume_stream_keeps_ids_and_finish_reason():
    def chunk(delta=None, finish=None, usage=None):
        choices = [SimpleNamespace(delta=delta, finish_reason=finish)] if delta or finish else []
        return SimpleNamespace(choices=choices, usage=usage)

    def tc(index, call_id=None, name=None, args=None):
        return SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=args))

    chunks = [
        chunk(SimpleNamespace(content="Hel", tool_calls=None)),
        chunk(SimpleNamespace(content="lo", reasoning_content="thinking", tool_calls=None)),
        chunk(SimpleNamespace(content=None, tool_calls=[tc(0, "call_abc", "read_file", '{"pa')])),
        chunk(SimpleNamespace(content=None, tool_calls=[tc(0, None, None, 'th": "x"}')])),
        chunk(finish="tool_calls"),
        chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)),
    ]

    async def stream():
        for c in chunks:
            yield c

    adapter = Adapter()
    response = asyncio.run(consume_stream(stream(), adapter))
    msg = response.choices[0].message
    assert msg.content == "Hello"
    assert msg.tool_calls[0].id == "call_abc"
    assert json.loads(msg.tool_calls[0].function.arguments) == {"path": "x"}
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.usage.total_tokens == 15
    assert ("partial", "Hello") in adapter.events
    assert ("partial_thought", "thinking") in adapter.events
    assert adapter.events[-1] == ("stream_end", None)


# ── split_by_token_budget ────────────────────────────────────────────────────

def test_split_always_makes_progress_and_keeps_tool_pairs():
    big = "x" * 4000
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": big},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "2"}]},
        {"role": "tool", "tool_call_id": "2", "content": big},
    ]
    old, recent = split_by_token_budget(history, keep_token_budget=10)
    assert old and recent
    assert recent[0]["role"] != "tool"
    assert old + recent == history
