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


@pytest.mark.parametrize("content", ["", "  \n", None, [{"type": "text", "text": ""}]])
def test_tool_only_assistant_step_goes_out_without_text(content):
    """An empty text beside tool calls becomes litellm's "[System: Empty message content sanitised ...]"
    block on Anthropic routes, which the model then echoes; send no text instead."""
    call = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    step = {"role": "assistant", "content": content, "tool_calls": [call]}
    [_, sent, _] = provider_messages([{"role": "user", "content": "go"}, step,
                                      {"role": "tool", "tool_call_id": "c1", "content": "r"}])
    assert sent == {"role": "assistant", "content": None, "tool_calls": [call]}
    assert step["content"] == content  # the stored step is left alone


def test_assistant_text_beside_tool_calls_is_kept():
    call = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    [sent] = provider_messages([{"role": "assistant", "content": "Reading it.", "tool_calls": [call]}])
    assert sent["content"] == "Reading it."


def test_no_sanitiser_placeholder_reaches_an_anthropic_request():
    """End to end through litellm's own Anthropic formatter: no placeholder, the tool_use block intact."""
    factory = pytest.importorskip("litellm.litellm_core_utils.prompt_templates.factory")
    call = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    sent = provider_messages([{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": [call]},
                              {"role": "tool", "tool_call_id": "c1", "content": "r"}])
    formatted = factory.anthropic_messages_pt(sent, model="claude-sonnet-4-5", llm_provider="anthropic")
    assert "sanitised" not in json.dumps(formatted)
    assert [b["type"] for b in formatted[1]["content"]] == ["tool_use"]


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
                         "Not every saved fact is listed: before writing code or committing, call search_memory for "
                         "the user's preferences and this project's conventions on that work.")


def test_facts_step_text_when_everything_was_forgotten():
    from dostuff.helpers.agent.turn_context import render_facts_step
    assert render_facts_step([], saved_user=0, saved_project=0, project="p") == (
        "No facts are saved any more. This replaces any earlier list of saved facts.")


def test_select_newest_first_until_the_budget_project_first_on_ties(monkeypatch):
    from dostuff.helpers.agent import turn_context
    from dostuff.helpers.agent.turn_context import fact_tokens, select_injected_facts
    user = [_fact(f"u{i:02d}", "v", updated=i) for i in range(60)]
    project = [_fact("tie", "v", "project", updated=59)]
    assert {fact_tokens(f) for f in user + project} == {3}  # "- u00: v\n"
    monkeypatch.setattr(turn_context, "FACT_TOKEN_BUDGET", 150)
    shown, overflow = select_injected_facts(user + project)
    assert len(shown) == 50
    assert [f["key"] for f in shown[:2]] == ["tie", "u59"]  # same timestamp: project first
    assert {f["key"] for f in overflow} == {f"u{i:02d}" for i in range(11)}  # the 11 oldest user facts


def test_default_budget_lists_hundreds_of_typical_facts():
    """The budget is sized for a few hundred facts of ~10 tokens, so a normal store is listed whole."""
    from dostuff.helpers.agent.turn_context import FACT_TOKEN_BUDGET, fact_tokens, select_injected_facts
    facts = [_fact(f"pref_{i:03d}", f"prefers option {i} for this", updated=i) for i in range(300)]
    assert 7 <= fact_tokens(facts[0]) <= 11
    shown, overflow = select_injected_facts(facts)
    assert len(shown) == 300 and overflow == []
    assert sum(fact_tokens(f) for f in shown) <= FACT_TOKEN_BUDGET


def test_budget_counts_tokens_not_facts(monkeypatch):
    """Long values use up the budget sooner: fewer of them fit than short ones."""
    from dostuff.helpers.agent import turn_context
    from dostuff.helpers.agent.turn_context import select_injected_facts
    monkeypatch.setattr(turn_context, "FACT_TOKEN_BUDGET", 300)
    short = [_fact(f"s{i:02d}", "v", updated=i) for i in range(100)]
    long = [_fact(f"l{i:02d}", "x" * 200, updated=i) for i in range(100)]
    assert len(select_injected_facts(short)[0]) == 100
    assert len(select_injected_facts(long)[0]) == 5  # 52 tokens each


def test_a_fact_too_long_for_what_is_left_is_skipped_not_the_end(monkeypatch):
    from dostuff.helpers.agent import turn_context
    from dostuff.helpers.agent.turn_context import select_injected_facts
    monkeypatch.setattr(turn_context, "FACT_TOKEN_BUDGET", 100)
    huge = _fact("huge", "x" * 2000, updated=100)  # newest, but bigger than the whole budget
    older = [_fact(f"o{i:02d}", "v", updated=i) for i in range(10)]
    shown, overflow = select_injected_facts([huge] + older)
    assert [f["key"] for f in overflow] == ["huge"]
    assert len(shown) == 10


def test_project_facts_get_at_most_their_share_of_the_budget(monkeypatch):
    from dostuff.helpers.agent import turn_context
    from dostuff.helpers.agent.turn_context import PROJECT_FACT_TOKEN_SHARE, fact_tokens, select_injected_facts
    monkeypatch.setattr(turn_context, "FACT_TOKEN_BUDGET", 150)
    project = [_fact(f"p{i:02d}", "v", "project", updated=100 + i) for i in range(40)]
    user = [_fact(f"u{i:02d}", "v", updated=i) for i in range(30)]
    shown, overflow = select_injected_facts(project + user)
    shown_project = [f for f in shown if f["scope"] == "project"]
    assert sum(fact_tokens(f) for f in shown_project) <= 150 * PROJECT_FACT_TOKEN_SHARE
    assert len(shown_project) == 30  # 90 of 150 tokens, the newest 30
    assert len(shown) == 50  # user facts fill the rest
    assert {f["key"] for f in overflow if f["scope"] == "project"} == {f"p{i:02d}" for i in range(10)}


def test_select_returns_overflow():
    from dostuff.helpers.agent.turn_context import select_injected_facts
    facts = [_fact("a", "v", updated=1), _fact("b", "v", "project", updated=2)]
    shown, overflow = select_injected_facts(facts)
    assert [f["key"] for f in shown] == ["b", "a"] and overflow == []
    assert select_injected_facts([]) == ([], [])


# --- memory recall T2.1: episodes already delivered in this session (D8) -------------------------------------------

def test_context_step_records_episode_ids():
    step = make_context_step("<turn-context>b</turn-context>", ("e1", "e2"))
    assert step["meta"] == {"hidden": TURN_CONTEXT_MARKER, "episode_ids": ["e1", "e2"]}
    # A bundle with no episodes is the step it always was
    assert make_context_step("<turn-context>b</turn-context>")["meta"] == {"hidden": TURN_CONTEXT_MARKER}


def test_delivered_episode_ids_every_bundle_in_history():
    from dostuff.helpers.agent.turn_context import delivered_episode_ids, make_facts_step
    history = [make_context_step("a", ("e1",)), {"role": "user", "content": "hi"},
               make_context_step("b", ("e2", "e3")), make_context_step("c"), make_facts_step("f", ()),
               {"role": "user", "content": "x", "meta": {"episode_ids": ["not-a-bundle"]}}]
    assert delivered_episode_ids(history) == {"e1", "e2", "e3"}
    assert delivered_episode_ids([]) == set()


# --- memory recall T2.2: the labelled episodes block (REQ-3.2, D8) -----------------------------------------------

def test_past_episodes_block_counts():
    from dostuff.helpers.agent.turn_context import EPISODES_MORE_NOTE, EpisodeCounts
    text = compose_turn_context(ENV, "", "  • [LEARNING] a", episode_counts=EpisodeCounts(3, "7", 24))
    assert '<past_episodes shown="3" matching="7" saved_in_project="24">\n  • [LEARNING] a\n' in text
    assert EPISODES_MORE_NOTE == "The closest to this prompt only; more are saved (search_memory finds them)."
    assert f"{EPISODES_MORE_NOTE}\n</past_episodes>" in text


def test_past_episodes_block_no_note_when_all_shown_and_unknown_counts_left_out():
    from dostuff.helpers.agent.turn_context import EPISODES_MORE_NOTE, EpisodeCounts
    text = compose_turn_context(ENV, "", "  • [LEARNING] a", episode_counts=EpisodeCounts(1, "1", 1))
    assert '<past_episodes shown="1" matching="1" saved_in_project="1">' in text and EPISODES_MORE_NOTE not in text
    # A store that can't count: only what is known
    text = compose_turn_context(ENV, "", "  • [LEARNING] a", episode_counts=EpisodeCounts(1, None, None))
    assert '<past_episodes shown="1">' in text and EPISODES_MORE_NOTE not in text
    assert "<past_episodes>" in compose_turn_context(ENV, "", "  • x")  # no counts given: the old tag
