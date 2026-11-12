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
    order = [text.index(s) for s in ("<environment>", "<other_relevant_facts>", "<past_episodes>", PROMPT_CONTEXT_HEADER)]
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
    assert "<other_relevant_facts>\n- k: v\n</other_relevant_facts>" in text
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


# --- memory recall T1.8 / T1.9: the facts step (plan/2026-11-11-memory-recall D2, D3) --------------------------

def _fact(key, value, scope="user", updated=0, working_dir=None):
    fact = {"key": key, "value": value, "source": "user_message", "scope": scope,
            "created_at_epoch": updated, "updated_at_epoch": updated}
    if scope == "project":
        fact["working_dir"] = working_dir or "/work/hooks-sandbox"
    return fact


def test_facts_fingerprint_sorted_and_hashes_value():
    import hashlib

    from dostuff.helpers.agent.turn_context import facts_fingerprint
    facts = [_fact("name", "Nemo"), _fact("linter", "ruff", "project")]
    fingerprint = facts_fingerprint(facts)
    assert fingerprint == tuple(sorted(fingerprint))
    assert f"user:name:{hashlib.sha1(b'Nemo').hexdigest()[:8]}" in fingerprint
    assert fingerprint[0].startswith("project:linter:")
    assert facts_fingerprint(list(reversed(facts))) == fingerprint
    assert facts_fingerprint([_fact("name", "Admiral Nemo"), facts[1]]) != fingerprint
    assert facts_fingerprint([]) == ()


def test_make_facts_step_shape():
    from dostuff.helpers.agent.turn_context import make_facts_step
    step = make_facts_step("<user_facts shown=\"1\" saved=\"1\">\n- name: Nemo\n</user_facts>", ("user:name:abcd1234",))
    assert step["role"] == "user" and is_turn_context(step)
    assert step["meta"]["facts"] == ["user:name:abcd1234"]
    assert step["content"].startswith("<turn-context>\n<user_facts")
    assert json.loads(json.dumps(step)) == step


def test_delivered_facts_newest_wins():
    from dostuff.helpers.agent.turn_context import delivered_facts, make_facts_step
    history = [make_facts_step("a", ("user:a:1",)), {"role": "user", "content": "hi"},
               make_context_step("<turn-context>x</turn-context>"), make_facts_step("b", ("user:b:2",)),
               make_plan_mode_step(True, "plan")]
    assert delivered_facts(history) == ("user:b:2",)


def test_delivered_facts_none_without_step():
    from dostuff.helpers.agent.turn_context import delivered_facts, make_facts_step
    assert delivered_facts([{"role": "user", "content": "hi"}, make_context_step("x")]) is None
    assert delivered_facts([make_facts_step("No facts", ())]) == ()


def test_facts_step_text():
    from dostuff.helpers.agent.turn_context import render_facts_step
    shown = [_fact("name", "Prefers to be called Admiral Nemo", updated=3),
             _fact("preferred_language", "Rust for code unless told otherwise", updated=2),
             _fact("linter", "ruff, not flake8 (flake8 clashed with black)", "project", updated=1)]
    text = render_facts_step(shown, saved_user=2, saved_project=1, project="hooks-sandbox")
    assert text == (
        '<user_facts shown="2" saved="2">\n'
        "- name: Prefers to be called Admiral Nemo\n"
        "- preferred_language: Rust for code unless told otherwise\n"
        "</user_facts>\n"
        '<project_facts project="hooks-sandbox" shown="1" saved="1">\n'
        "- linter: ruff, not flake8 (flake8 clashed with black)\n"
        "</project_facts>\n"
        "This replaces any earlier list of saved facts."
    )
    # A scope with nothing saved has no block
    assert "<project_facts" not in render_facts_step(shown[:1], saved_user=1, saved_project=0, project="p")


def test_facts_step_text_overflow_note():
    from dostuff.helpers.agent.turn_context import render_facts_step
    text = render_facts_step([_fact("name", "Nemo")], saved_user=51, saved_project=0, project="p")
    assert '<user_facts shown="1" saved="51">' in text
    assert text.endswith("This replaces any earlier list of saved facts.\n"
                         "Other saved facts appear in the turn context when relevant.")


def test_facts_step_text_when_everything_was_forgotten():
    from dostuff.helpers.agent.turn_context import render_facts_step
    assert render_facts_step([], saved_user=0, saved_project=0, project="p") == (
        "No facts are saved any more. This replaces any earlier list of saved facts.")


def test_select_newest_50_project_first_on_ties():
    from dostuff.helpers.agent.turn_context import FACT_INJECT_CAP, select_injected_facts
    user = [_fact(f"u{i:02d}", "v", updated=i) for i in range(60)]
    project = [_fact("tie", "v", "project", updated=59)]
    shown, overflow = select_injected_facts(user + project)
    assert len(shown) == FACT_INJECT_CAP == 50
    assert [f["key"] for f in shown[:2]] == ["tie", "u59"]  # same timestamp: project first
    assert {f["key"] for f in overflow} == {f"u{i:02d}" for i in range(11)}  # the 11 oldest user facts


def test_project_facts_capped_at_30_within_50():
    from dostuff.helpers.agent.turn_context import PROJECT_FACT_CAP, select_injected_facts
    project = [_fact(f"p{i:02d}", "v", "project", updated=100 + i) for i in range(40)]
    user = [_fact(f"u{i:02d}", "v", updated=i) for i in range(30)]
    shown, overflow = select_injected_facts(project + user)
    assert sum(f["scope"] == "project" for f in shown) == PROJECT_FACT_CAP == 30
    assert len(shown) == 50
    assert {f["key"] for f in overflow if f["scope"] == "project"} == {f"p{i:02d}" for i in range(10)}


def test_select_returns_overflow():
    from dostuff.helpers.agent.turn_context import select_injected_facts
    facts = [_fact("a", "v", updated=1), _fact("b", "v", "project", updated=2)]
    shown, overflow = select_injected_facts(facts)
    assert [f["key"] for f in shown] == ["b", "a"] and overflow == []
    assert select_injected_facts([]) == ([], [])
