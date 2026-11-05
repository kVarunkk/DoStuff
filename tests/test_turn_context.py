"""The turn-context step: the marker, the composer, the plan step (plan/2026-11-05-cache-stable-context)."""
import asyncio
import json
import subprocess
import sys

import pytest

from dostuff.agent.call_agent import provider_messages
from dostuff.helpers.agent.turn_context import (
    HIDDEN_LIKE_PATTERN, PROMPT_CONTEXT_HEADER, TURN_CONTEXT_MARKER, compose_turn_context, delivered_plan_mode,
    is_turn_context, make_context_step, make_plan_mode_step,
)

ENV = "<environment>\ncwd: /x\n</environment>"


# --- T1.1 the marker --------------------------------------------------------------------------------------------

def test_marker_true_on_context_step():
    assert is_turn_context({"role": "user", "content": "x", "meta": {"hidden": TURN_CONTEXT_MARKER}})


def test_marker_false_for_plain_user_step():
    assert not is_turn_context({"role": "user", "content": "x"})


@pytest.mark.parametrize("step", [
    None, "text", {}, {"meta": None}, {"meta": "turn_context"}, {"meta": {"hidden": "other"}},
    {"meta": {"hidden": ["turn_context"]}},
])
def test_marker_false_for_garbage(step):
    assert is_turn_context(step) is False


def test_marker_survives_json_roundtrip():
    assert is_turn_context(json.loads(json.dumps(make_context_step("<turn-context>x</turn-context>"))))


# --- T1.2 the LIKE pattern --------------------------------------------------------------------------------------

def test_like_pattern_contains_marker():
    assert HIDDEN_LIKE_PATTERN == '%"hidden": "turn_context"%'
    assert TURN_CONTEXT_MARKER in HIDDEN_LIKE_PATTERN and "NOT LIKE" not in HIDDEN_LIKE_PATTERN


# --- T1.3 no heavy imports --------------------------------------------------------------------------------------

def test_turn_context_import_is_light():
    code = ("import sys; import dostuff.helpers.agent.turn_context; "
            "print([m for m in ('litellm', 'chromadb', 'mcp') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# --- T1.4 the composer ------------------------------------------------------------------------------------------

def test_bundle_section_order():
    text = compose_turn_context(ENV, "- k: v", "  • [EVENT] e", "hook says")
    order = [text.index(s) for s in ("<environment>", "<relevant_user_facts>", "<past_episodes>", PROMPT_CONTEXT_HEADER)]
    assert order == sorted(order)


def test_bundle_omits_empty_sections():
    text = compose_turn_context(ENV)
    assert text == f"<turn-context>\n{ENV}\n</turn-context>"


def test_bundle_all_empty_is_blank():
    assert compose_turn_context("", "", "", "") == ""


def test_bundle_wraps_in_turn_context():
    text = compose_turn_context(ENV, "- k: v")
    assert text.startswith("<turn-context>\n") and text.endswith("\n</turn-context>")


def test_bundle_keeps_inner_block_names():
    text = compose_turn_context(ENV, "- k: v", "  • e", "hook says")
    assert "<relevant_user_facts>\n- k: v\n</relevant_user_facts>" in text
    assert "<past_episodes>\n  • e\n</past_episodes>" in text
    assert "[Context from UserPromptSubmit hooks]\nhook says" in text
    assert ENV in text


# --- T1.5 the step ----------------------------------------------------------------------------------------------

def test_make_context_step_shape():
    step = make_context_step("bundle")
    assert step == {"role": "user", "content": "bundle", "meta": {"hidden": "turn_context"}}
    assert is_turn_context(step)


# --- T1.6 the plan step -----------------------------------------------------------------------------------------

def test_plan_step_is_a_context_step():
    step = make_plan_mode_step(True, "## Plan mode\nrules")
    assert step["role"] == "user"
    assert step["content"] == "<turn-context>\n## Plan mode\nrules\n</turn-context>"
    assert step["meta"] == {"hidden": "turn_context", "plan_mode": True}
    assert is_turn_context(step)


def test_delivered_plan_mode_newest_wins():
    history = [make_plan_mode_step(True, "on"), {"role": "user", "content": "x"}, make_plan_mode_step(False, "off")]
    assert delivered_plan_mode(history) is False
    assert delivered_plan_mode(history + [make_plan_mode_step(True, "on")]) is True


def test_delivered_plan_mode_defaults_off():
    assert delivered_plan_mode([]) is False
    assert delivered_plan_mode([{"role": "user", "content": "x"}]) is False


def test_bundle_does_not_change_delivered_plan_mode():
    assert delivered_plan_mode([make_plan_mode_step(True, "on"), make_context_step("bundle")]) is True


def test_plan_step_survives_json_roundtrip():
    history = json.loads(json.dumps([make_plan_mode_step(True, "on")]))
    assert is_turn_context(history[0]) and delivered_plan_mode(history) is True


# --- T3.7 single source for the marker --------------------------------------------------------------------------

def test_marker_literal_not_duplicated():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "dostuff"
    found = [p.relative_to(root).as_posix() for p in root.rglob("*.py")
             if '"turn_context"' in (text := p.read_text(encoding="utf-8")) or "'turn_context'" in text]
    assert found == ["helpers/agent/turn_context.py"]


# --- what a provider receives -----------------------------------------------------------------------------------

def _context_step(text="<turn-context>env</turn-context>"):
    return make_context_step(text)


def test_context_step_and_prompt_reach_the_provider_as_one_message():
    sent = provider_messages([_context_step(), {"role": "user", "content": "the prompt"}])
    assert sent == [{"role": "user", "content": "<turn-context>env</turn-context>\n\nthe prompt"}]


def test_no_two_user_messages_in_a_row_across_a_turn_with_tools():
    call = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    plan = make_plan_mode_step(True, "plan on")
    sent = provider_messages([
        _context_step(), {"role": "user", "content": "go"}, plan,
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "user", "content": "[hook ctx]"}, {"role": "user", "content": "steer"},
    ])
    assert [m["role"] for m in sent] == ["user", "assistant", "tool", "user"]
    assert sent[-1]["content"] == "[hook ctx]\n\nsteer"
    assert sent[2]["content"] == "result"  # a tool result never takes a note into itself


def test_merge_leaves_the_stored_steps_alone():
    steps = [_context_step(), {"role": "user", "content": "go"}]
    before = json.loads(json.dumps(steps))
    provider_messages(steps)
    assert steps == before


def test_meta_never_reaches_the_provider():
    sent = provider_messages([_context_step(), {"role": "assistant", "content": "a"}, _context_step()])
    assert all("meta" not in m for m in sent)


def test_merge_keeps_image_blocks():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
    sent = provider_messages([{"role": "user", "content": "look"}, {"role": "user", "content": [image]}])
    assert sent == [{"role": "user", "content": [{"type": "text", "text": "look"}, image]}]


def test_merge_skips_empty_content():
    sent = provider_messages([{"role": "user", "content": ""}, {"role": "user", "content": "go"}])
    assert sent == [{"role": "user", "content": "go"}]


def test_a_later_turn_only_extends_what_the_last_one_sent():
    """The cache rests on this: every message of a call is unchanged, in place, in the next call."""
    first = [_context_step("<turn-context>1</turn-context>"), {"role": "user", "content": "one"}]
    second = first + [{"role": "assistant", "content": "done"}, _context_step("<turn-context>2</turn-context>"),
                      {"role": "user", "content": "two"}]
    earlier = provider_messages(first)
    assert provider_messages(second)[:len(earlier)] == earlier


@pytest.mark.parametrize("explicit", [False, True])
def test_call_agent_merges_for_every_provider(fake_model, monkeypatch, explicit):
    from dostuff.agent import call_agent as module
    monkeypatch.setattr(module, "uses_explicit_prompt_caching", lambda: explicit)
    steps = [_context_step(), {"role": "user", "content": "go"}]
    asyncio.run(module.call_agent(steps, "system", tool_names=[]))
    assert [m["role"] for m in fake_model.calls[0]["messages"]] == ["system", "user"]
