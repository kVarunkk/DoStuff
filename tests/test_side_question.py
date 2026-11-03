"""`/btw` side questions: the call, the snapshot registry and the shared helpers (plan/2026-11-03-btw)."""
import asyncio
import dataclasses
import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dostuff.agent.permissions import PLAN_MODE_INSTRUCTION, SessionPermissions
from dostuff.helpers.agent import side_question as sq
from dostuff.helpers.agent.side_question import (
    BTW_MAX_EXCHANGES, SIDE_QUESTION_PROMPT, TOOL_CALL_NOTE, SideAnswer, add_exchange, answer_side_question,
    build_messages, compose_system_instruction, discard_snapshot, kept_prompt, plan_mode_on, publish_snapshot,
    published_snapshot, snapshot, usage_dict,
)
from tests.conftest import fake_response, fake_tool_call

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_registry():
    sq._reset_registry()
    yield
    sq._reset_registry()


def _ask(ctx, exchanges=(), question="where is retry?", plan_mode=False):
    return asyncio.run(answer_side_question(ctx, list(exchanges), question, plan_mode=plan_mode))


# --- T0.1 -------------------------------------------------------------------------------------------------------

def test_imports():
    assert BTW_MAX_EXCHANGES == 20
    assert "{question}" in SIDE_QUESTION_PROMPT
    assert {f.name for f in dataclasses.fields(SideAnswer)} == {
        "answer", "sent_question", "usage", "cost", "tool_call_note"}


# --- T1.1 the snapshot registry ---------------------------------------------------------------------------------

def test_publish_and_read(fork_ctx):
    ctx = fork_ctx()
    publish_snapshot("s1", ctx)
    got = snapshot("s1")
    assert got == ctx and got.history is not ctx.history
    ctx.history.append({"role": "user", "content": "later"})
    assert len(got.history) == 4
    assert snapshot("s2") is None


def test_publish_none_is_noop(fork_ctx):
    publish_snapshot("s1", fork_ctx())
    before = snapshot("s1")
    publish_snapshot("s1", None)
    assert snapshot("s1") is before


def test_demote_on_exit(fork_ctx):
    ctx = fork_ctx(compaction_notes="N", session_hook_context={"startup": "x"})
    with published_snapshot("s1"):
        publish_snapshot("s1", ctx)
        assert snapshot("s1") is not None
    assert snapshot("s1") is None
    kept = kept_prompt("s1")
    assert kept is not None and kept.history == []
    assert (kept.system_instruction, kept.compaction_notes, kept.tool_names, kept.session_hook_context) == (
        "SYS", "N", ["read_file", "bash"], {"startup": "x"})


def test_demote_on_raise(fork_ctx):
    with pytest.raises(RuntimeError):
        with published_snapshot("s1"):
            publish_snapshot("s1", fork_ctx())
            raise RuntimeError("boom")
    assert snapshot("s1") is None and kept_prompt("s1") is not None


def test_demote_disabled(fork_ctx):
    publish_snapshot("s1", fork_ctx())
    live = snapshot("s1")
    with published_snapshot("s1", enabled=False):
        pass
    assert snapshot("s1") is live


def test_discard(fork_ctx):
    with published_snapshot("s1"):
        publish_snapshot("s1", fork_ctx())
    publish_snapshot("s1", fork_ctx())
    discard_snapshot("s1")
    assert snapshot("s1") is None and kept_prompt("s1") is None
    discard_snapshot("unknown")


def test_one_entry_per_session(fork_ctx):
    publish_snapshot("s1", fork_ctx(system_instruction="A"))
    publish_snapshot("s1", fork_ctx(system_instruction="B"))
    live = snapshot("s1")
    assert live is not None and live.system_instruction == "B"
    assert list(sq._entries) == ["s1"]


def test_side_question_import_is_light():
    code = ("import sys; import dostuff.helpers.agent.side_question; "
            "print([m for m in ('litellm', 'chromadb', 'mcp') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert out.stdout.strip() == "[]"


# --- T1.2 the composer and plan mode ----------------------------------------------------------------------------

def _suffix(context):
    from dostuff.helpers.hooks.events import session_context_suffix
    return session_context_suffix(context)


# The T0.2 cases (tests/test_loop.py::SYSTEM_PIN_CASES), with the strings the unchanged loop sent
COMPOSE_CASES = [
    (True, "N", {"startup": "ctx"},
     "D\n\n" + PLAN_MODE_INSTRUCTION + "\n\n[Summary of earlier conversation]: N" + _suffix({"startup": "ctx"})),
    (False, "N", {"startup": "ctx"}, "D\n\n[Summary of earlier conversation]: N" + _suffix({"startup": "ctx"})),
    (True, "", {"startup": "ctx"}, "D\n\n" + PLAN_MODE_INSTRUCTION + _suffix({"startup": "ctx"})),
    (True, "N", None, "D\n\n" + PLAN_MODE_INSTRUCTION + "\n\n[Summary of earlier conversation]: N"),
]


@pytest.mark.parametrize("plan, notes, context, expected", COMPOSE_CASES)
def test_compose_matches_loop_pin(plan, notes, context, expected):
    assert compose_system_instruction("D", plan, notes, context) == expected


def test_plan_mode_on():
    ask = SimpleNamespace(permission_mode="ask")
    plan = SimpleNamespace(permission_mode="plan")
    assert plan_mode_on(SessionPermissions(mode="plan"), ask) is True
    assert plan_mode_on(SessionPermissions(mode=None), plan) is True
    assert plan_mode_on(None, ask) is False
    assert plan_mode_on(SessionPermissions(), ask) is False
    assert plan_mode_on(SessionPermissions(mode="auto"), plan) is False
    assert plan_mode_on(None, SimpleNamespace()) is False  # a config without the field: "ask"


# --- T1.3 messages and the exchange cap -------------------------------------------------------------------------

def test_build_messages_order(fork_ctx):
    ctx = fork_ctx()
    got = build_messages(ctx, [("q1s", "a1"), ("q2s", "a2")], "q3")
    assert got == ctx.history + [
        {"role": "user", "content": "q1s"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2s"}, {"role": "assistant", "content": "a2"},
        {"role": "user", "content": SIDE_QUESTION_PROMPT.format(question="q3")},
    ]


def test_build_messages_does_not_mutate(fork_ctx):
    ctx = fork_ctx()
    history, history_copy = ctx.history, [dict(s) for s in ctx.history]
    exchanges = [("q1s", "a1")]
    build_messages(ctx, exchanges, "q")
    assert ctx.history is history and ctx.history == history_copy
    assert exchanges == [("q1s", "a1")]


def test_cap_on_write():
    full = [(f"q{i}", f"a{i}") for i in range(20)]
    add_exchange(full, "new", "ans")
    assert len(full) == 20 and full[0] == ("q1", "a1") and full[-1] == ("new", "ans")
    short = [("q0", "a0")] * 3
    add_exchange(short, "n", "a")
    assert len(short) == 4 and short[-1] == ("n", "a")


def test_build_messages_never_truncates(fork_ctx):
    many = [(f"q{i}", f"a{i}") for i in range(30)]
    assert len(build_messages(fork_ctx(history=[]), many, "q")) == 61


def test_side_prompt_says_no_tools():
    assert SIDE_QUESTION_PROMPT.count("{question}") == 1
    for phrase in ("from the conversation above only", "in one response", "without calling tools"):
        assert phrase in SIDE_QUESTION_PROMPT


def test_question_with_braces_is_sent_verbatim():
    assert "what is {x}?" in SIDE_QUESTION_PROMPT.format(question="what is {x}?")


# --- T1.4 the call and the D7 note ------------------------------------------------------------------------------

def test_sends_turn_tools_not_empty(fork_ctx, side_call):
    ctx = fork_ctx(compaction_notes="N", session_hook_context={"startup": "c"})
    call = side_call()
    _ask(ctx)
    assert len(call.calls) == 1
    got = call.calls[0]
    assert got["tool_names"] == ["read_file", "bash"]
    assert got["stream"] is False
    assert got["steps_history"] == build_messages(ctx, [], "where is retry?")
    assert got["system_instruction"] == compose_system_instruction("SYS", False, "N", {"startup": "c"})


def test_plain_answer(fork_ctx, side_call):
    side_call(fake_response("In call_agent."))
    answer = _ask(fork_ctx())
    assert answer.text == answer.answer == "In call_agent."
    assert answer.tool_call_note is None
    assert answer.sent_question == SIDE_QUESTION_PROMPT.format(question="where is retry?")


def test_tool_calls_never_dispatched(fork_ctx, side_call, monkeypatch):
    import types

    async def boom(*a, **k):
        raise AssertionError("a side question dispatched a tool")
    # Whatever is loaded, nothing reachable from the side call may dispatch: a stub run_tool that fails if used
    stub = types.ModuleType("dostuff.agent.run_tool")
    stub.dispatch_tool = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dostuff.agent.run_tool", stub)
    side_call(fake_response("Let me look.", tool_calls=[fake_tool_call("read_file", {"path": "a.py"})]))
    answer = _ask(fork_ctx())
    assert answer.answer == "Let me look."
    assert answer.text == f"Let me look.\n\n{TOOL_CALL_NOTE} Asked for: read_file."
    assert answer.tool_call_note is not None and "read_file" in answer.tool_call_note


def test_tool_calls_without_text(fork_ctx, side_call):
    call = side_call(fake_response(None, tool_calls=[fake_tool_call("read_file", {}, "c1"),
                                                     fake_tool_call("bash", {}, "c2")]))
    answer = _ask(fork_ctx())
    assert answer.answer == ("The model asked to use read_file, bash instead of answering — press f to let a "
                             "background agent do it.")
    assert answer.text.endswith(f"{TOOL_CALL_NOTE} Asked for: read_file, bash.")
    assert len(call.calls) == 1


@pytest.mark.parametrize("text, noted", [
    ('I would run:\n```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```', True),
    ('{"name": "x", "parameters": {}}', True),
    ('{"name": "x"}', False),
    ("The retry loop is in call_agent.", False),
    ("{'name': 'x', 'arguments': {}}", False),
])
def test_json_in_text_note(fork_ctx, side_call, text, noted):
    side_call(fake_response(text))
    answer = _ask(fork_ctx())
    assert (answer.tool_call_note == TOOL_CALL_NOTE) is noted
    assert answer.answer == text


def test_answer_excludes_note_for_replay(fork_ctx, side_call):
    side_call(fake_response('{"name": "x", "arguments": {}}'))
    answer = _ask(fork_ctx())
    assert TOOL_CALL_NOTE not in answer.answer and TOOL_CALL_NOTE in answer.text


def test_empty_answer(fork_ctx, side_call):
    side_call(fake_response(None))
    answer = _ask(fork_ctx())
    assert answer.answer == sq.EMPTY_ANSWER and answer.tool_call_note is None


def test_replayed_exchange_matches_request(fork_ctx, side_call):
    ctx = fork_ctx()
    call = side_call(fake_response("first answer"), fake_response("second"))
    exchanges: list = []
    first = _ask(ctx, exchanges, "q1")
    add_exchange(exchanges, first.sent_question, first.answer)
    _ask(ctx, exchanges, "q2")
    first_request, second_request = call.calls[0]["steps_history"], call.calls[1]["steps_history"]
    assert second_request[:len(first_request)] == first_request  # the first request is a prefix of the second
    assert second_request[len(first_request)] == {"role": "assistant", "content": "first answer"}


def test_never_given_a_mutating_step(fork_ctx, side_call):
    ctx = fork_ctx()
    before = [dict(s) for s in ctx.history]
    call = side_call()
    _ask(ctx, [("q1s", "a1")])
    assert ctx.history == before
    sent = call.calls[0]["steps_history"]
    assert sent[:len(before)] == before  # the snapshot, unchanged, then only text steps
    assert all(set(s) == {"role", "content"} and s["role"] in ("user", "assistant") for s in sent[len(before):])


# --- T1.5 span, failure, usage ----------------------------------------------------------------------------------

def test_span_ended_on_success(fork_ctx, side_call):
    call = side_call()
    _ask(fork_ctx())
    assert call.spans[0].ended == 1
    call = side_call(recording=False)
    _ask(fork_ctx())
    assert call.spans[0].ended == 0


def test_failure_propagates_without_ending_span(fork_ctx, side_call):
    call = side_call(raise_exc=RuntimeError("429"))
    with pytest.raises(RuntimeError, match="429"):
        _ask(fork_ctx())
    assert call.spans == []  # call_agent ends its own span when it raises


def test_usage_dict_shared(fork_ctx, side_call, monkeypatch):
    from dostuff.helpers.agent import model_cost
    monkeypatch.setattr(model_cost, "call_cost", lambda model, usage: 0.5)
    details = SimpleNamespace(cached_tokens=80)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120, prompt_tokens_details=details)
    expected = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cached_tokens": 80, "cost": 0.5}
    assert usage_dict(fake_response("x", usage=usage)) == expected
    assert usage_dict(fake_response("x")) is None
    side_call(fake_response("x", usage=usage))
    answer = _ask(fork_ctx())
    assert answer.usage == expected and answer.cost == 0.5


def test_no_emit_no_store():
    params = set(inspect.signature(answer_side_question).parameters)
    assert params == {"snap", "exchanges", "question", "plan_mode"}


# --- T5.3 the README ---------------------------------------------------------------------------------------------

def test_readme_mentions_btw():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    features = text[text.index("## Features"):text.index("## Install")]
    assert any("`/btw`" in line for line in features.splitlines() if line.startswith("- "))
    controls = text[text.index("## TUI controls"):text.index("## Tools and permissions")]
    assert any(line.startswith("| `/btw <question>`") for line in controls.splitlines())
    coming = text[text.index("### Coming from Claude Code"):text.index("## Configuration")]
    assert any(line.startswith("| `/btw`") for line in coming.splitlines())
    section = text[text.index("## Side questions (`/btw`)"):text.index("## Subagents")]
    # the sketch is a real render: its footer is the panel's own footer text
    from dostuff.helpers.agent.side_question import BTW_MAX_EXCHANGES
    assert "Esc/Enter/Space close · f fork out · c copy" in section
    assert "No tools run." in section and "Nothing is written." in section and "No follow-up turns." in section
    assert f"the last {BTW_MAX_EXCHANGES}" in section and "Fork out (`f`)" in section and "dostuff -p" in section
