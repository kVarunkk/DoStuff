"""The turn-context step: the marker, the composer, the plan step (plan/2026-11-05-cache-stable-context)."""
import json
import subprocess
import sys

import pytest

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
