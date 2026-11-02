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
        _tc("agent", json.dumps({"description": "d", "prompt": "t"}), "c2"),
    ]
    monkeypatch.setattr(loop_module, "authorize_tool", lambda *a, **k: _none())

    async def slow_dispatch(name, args, **kw):
        if name == "agent":
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


def test_call_label_clips_long_arguments():
    from dostuff.helpers.agent.tool_output import format_call_label
    content = "line one\n" + "x" * 5000
    label = format_call_label("write_file", {"path": "game.py", "content": content})
    assert label.startswith('write_file(path="game.py", content="line one\\n')
    assert "\n" not in label and len(label) <= 200 and "…" in label
    # history stores the raw JSON string; it renders the same way
    assert format_call_label("write_file", json.dumps({"path": "game.py", "content": content})) == label
    assert format_call_label("read_file", '{"path": "a"}') == 'read_file(path="a")'
    assert format_call_label("x", "{bad json") == "x({bad json)"


# --- ask_user in the loop (plan/2026-11-02-ask-user, T2.2 / T2.6) ---------------------------------

def test_ask_user_batch_is_sequential(monkeypatch):
    """A batch of read_file + ask_user never overlaps: the panel must not open while output streams in."""
    spans = []

    async def dispatch(name, args, **kw):
        spans.append(("enter", name))
        await asyncio.sleep(0.05)
        spans.append(("exit", name))
        return "ok"

    calls = [_tc("read_file", json.dumps({"path": "a"}), "c1"),
             _tc("ask_user", json.dumps({"questions": [{"question": "Which db?"}]}), "c2")]
    _, working, _ = _run(monkeypatch, [_msg(tool_calls=calls), _msg(content="done")], dispatch=dispatch)
    assert spans.index(("exit", "read_file")) < spans.index(("enter", "ask_user"))
    tool_steps = [s for s in working if s["role"] == "tool"]
    assert [s["name"] for s in tool_steps] == ["read_file", "ask_user"]


def test_ask_user_is_untimed(monkeypatch):
    """A user may think for minutes: ask_user is exempt from TOOL_TIMEOUT_S, read_file is not."""
    monkeypatch.setattr(loop_module, "TOOL_TIMEOUT_S", 0.01)

    async def dispatch(name, args, **kw):
        await asyncio.sleep(0.05)
        return "Asked the user 1 question; they answered:\n1. Which db?\n   → SQLite"

    result, working, _ = _run(monkeypatch, [_msg(tool_calls=[_tc("ask_user", json.dumps({"questions": [{"question": "Which db?"}]}), "c1")]), _msg(content="done")], dispatch=dispatch)
    assert "timed out" not in next(s for s in working if s["role"] == "tool")["content"]

    calls = [_tc("read_file", json.dumps({"path": "a"}), "c1")]
    _, working, _ = _run(monkeypatch, [_msg(tool_calls=calls), _msg(content="done")], dispatch=dispatch)
    assert "timed out" in next(s for s in working if s["role"] == "tool")["content"]


def test_second_ask_gets_cancelled_step(monkeypatch):
    """Esc on the first ask_user ends the turn; the second never starts and still gets its one step."""
    adapter = Adapter()
    calls = [_tc("ask_user", json.dumps({"questions": [{"question": "One?"}]}), "c1"),
             _tc("ask_user", json.dumps({"questions": [{"question": "Two?"}]}), "c2")]

    async def dispatch(name, args, **kw):
        await asyncio.sleep(0.01)  # let the task start, as a real adapter would before the user Escapes
        adapter._cancelled = True
        return "The user cancelled the questions; no answers.\nUse your best judgement and say what you assumed."

    result, working, _ = _run(monkeypatch, [_msg(tool_calls=calls)], adapter=adapter, dispatch=dispatch)
    assert result == "Cancelled by user."
    tool_steps = [s for s in working if s["role"] == "tool"]
    assert len(tool_steps) == 2
    assert "cancelled the questions" in tool_steps[0]["content"]
    assert tool_steps[1]["content"] == CANCELLED_RESULT


def test_ask_user_call_label():
    from dostuff.helpers.agent.tool_output import format_call_label

    three = {"questions": [{"question": "Which database should I use?"}, {"question": "Which auth?"}, {"question": "Name?"}]}
    assert format_call_label("ask_user", three) == 'ask_user("Which database should I use?" +2 more)'
    one = {"questions": [{"question": "Which database should I use?"}]}
    assert format_call_label("ask_user", one) == 'ask_user("Which database should I use?")'
    for garbage in ({}, {"questions": []}, {"questions": "nope"}, {"questions": [{}]}):
        label = format_call_label("ask_user", garbage)  # falls back to the generic shape
        assert label.startswith("ask_user(") and "more" not in label, label
        assert len(label) <= 200
    long = {"questions": [{"question": "Q" * 400}]}
    assert len(format_call_label("ask_user", long)) <= 200


def test_call_labels_are_names_only():
    from dostuff.helpers.agent.tool_output import format_call_label

    args = {"questions": [{"question": "Which db?"}]}
    mapped = format_call_label("ask_user", args)
    assert format_call_label("ask_user", args, labels={}) == f"ask_user(questions={json.dumps(args['questions'])})"
    assert mapped != format_call_label("ask_user", args, labels={})  # mapping disabled → the generic shape
    assert format_call_label("ask_user", args, labels={"ask_user": lambda a: "custom"}) == "ask_user(custom)"


def test_tool_output_imports_no_tools():
    """CON-010: the TUI's pre-first-frame path must not pull in a tool module (it once cost 27 s)."""
    import subprocess
    import sys
    code = (
        "import sys\n"
        "from dostuff.helpers.agent.tool_output import format_call_label\n"
        "assert format_call_label('ask_user', {'questions': [{'question': 'Which db?'}]})\n"
        "assert not [m for m in sys.modules if m == 'dostuff.tools' or m.startswith('dostuff.tools.')], "
        "[m for m in sys.modules if m.startswith('dostuff.tools')]\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_ask_user_answers_are_in_the_next_call(monkeypatch):
    """T2.6 / WS-2 acceptance: the model calls ask_user, the adapter answers, the next call sees them."""
    from tests.conftest import FakeAdapter
    from dostuff.helpers.ui.ask_answers import format_answers

    pairs = [("Which database should I use?", "SQLite"), ("Which auth methods?", "JWT, OAuth")]
    adapter = FakeAdapter()
    adapter.ask_answers = pairs
    questions = [{"question": pairs[0][0], "options": [{"label": "SQLite"}, {"label": "Postgres"}]},
                 {"question": pairs[1][0], "options": [{"label": "JWT"}, {"label": "OAuth"}]}]
    calls = [_tc("ask_user", json.dumps({"questions": questions}), "c1")]
    _, working, calls_seen = _run(monkeypatch, [_msg(tool_calls=calls), _msg(content="done")], adapter=adapter)

    assistant = next(s for s in working if s["role"] == "assistant" and s.get("tool_calls"))
    assert assistant["tool_calls"][0]["function"]["name"] == "ask_user"
    tool_steps = [s for s in working if s["role"] == "tool"]
    assert len(tool_steps) == 1
    assert tool_steps[0]["content"] == format_answers(pairs)  # the D5 block, not a JSON blob
    assert adapter.asked and adapter.asked[0][0]["options"][0]["label"] == "SQLite"
    # the second model call is the one that sees the answers
    assert any(s["role"] == "tool" and s["content"] == format_answers(pairs) for s in calls_seen[1])


def test_ask_user_refusal_is_a_normal_step(monkeypatch):
    from tests.conftest import FakeAdapter

    adapter = FakeAdapter()
    calls = [_tc("ask_user", json.dumps({"questions": []}), "c1")]
    result, working, calls_seen = _run(monkeypatch, [_msg(tool_calls=calls), _msg(content="done")], adapter=adapter)
    tool_steps = [s for s in working if s["role"] == "tool"]
    assert len(tool_steps) == 1 and tool_steps[0]["content"].startswith("error: ")
    assert result == "done" and not adapter.asked  # the adapter was never touched
    assert "error: " in calls_seen[1][-1]["content"]


def test_ask_user_without_adapter_in_the_loop(monkeypatch):
    from dostuff.helpers.ui.ask_answers import UNAVAILABLE_ASK_RESULT

    calls = [_tc("ask_user", json.dumps({"questions": [{"question": "Which db?"}]}), "c1")]
    result, working, _ = _run(monkeypatch, [_msg(tool_calls=calls), _msg(content="done")], adapter=None)
    assert result == "done"
    tool_steps = [s for s in working if s["role"] == "tool"]
    assert len(tool_steps) == 1 and tool_steps[0]["content"] == UNAVAILABLE_ASK_RESULT
