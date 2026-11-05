"""Fork mode (subagents spec D11, REQ-7)."""
import asyncio

import pytest

from conftest import FakeAdapter, fake_response, fake_tool_call

FOOTER = "agent_id: {id} (send_message to continue)"

# Two earlier turns, then the turn that forks. The fork's prompt carries its own marker ([F]), so the fake
# model routes its calls apart from the main agent's ([MAIN]), though it sees the main agent's history.
EARLIER = [
    {"role": "user", "content": "[MAIN] the secret word is plum"},
    {"role": "assistant", "content": "noted"},
    {"role": "user", "content": "and the colour is teal"},
    {"role": "assistant", "content": "noted too"},
    {"role": "user", "content": "now fork"},
]


def _fork_call(prompt: str = "[F] X", call_id: str = "c1", **extra):
    return fake_tool_call("agent", {"description": "fork it", "prompt": prompt, "subagent_type": "fork", **extra},
                          call_id)


async def _main_turn(working, *, system="MAIN SYSTEM", notes="earlier notes", hook_ctx=None, tool_names=None,
                     adapter=None):
    from dostuff.agent.loop import loop
    return await loop(session_id="s1", turn_id="t1", user_text="now fork", dynamic_system_instruction=system,
                      mcp_client=object(), working_history=working, turn_type="interactive_loop",  # type: ignore[arg-type]
                      stream=False, adapter=adapter or FakeAdapter(), compaction_notes=notes,
                      session_hook_context=hook_ctx, tool_names=tool_names)


def _fork_calls(fake_model):
    return fake_model.calls_for("[F]")


def _tool_names(call) -> set[str]:
    return {t["function"]["name"] for t in call.get("tools") or []}


# --- T6.1 fork history, directive and prompt ----------------------------------------------------

def test_fork_first_call(agents, fake_model):
    from dostuff.agent.subagents.fork import FORK_DIRECTIVE
    fake_model.script("[MAIN]", fake_response(tool_calls=[_fork_call()]), fake_response("main done"))
    fake_model.script("[F]", fake_response("fork report"))
    working = [dict(s) for s in EARLIER]

    result = asyncio.run(_main_turn(working, hook_ctx={"startup": "hook context"}))

    assert result == "main done"
    main_first = fake_model.calls[0]
    [fork_first] = _fork_calls(fake_model)
    # (a) the system message is byte-identical: turn instructions + compaction notes + session context
    assert fork_first["messages"][0] == main_first["messages"][0]
    assert "earlier notes" in fork_first["messages"][0]["content"]
    assert "hook context" in fork_first["messages"][0]["content"]
    # (b) the main working history as of the call, without the in-flight assistant step
    assert fork_first["messages"][1:-1] == main_first["messages"][1:-1]
    # (c) then the directive and the task, which extend the last user message: it ended on a user step
    last = fork_first["messages"][-1]
    directive = FORK_DIRECTIVE.format(prompt="[F] X")
    assert last["role"] == "user" and last["content"] == main_first["messages"][-1]["content"] + "\n\n" + directive
    assert "Your task: [F] X" in last["content"]
    # the main history was copied, not shared
    assert not any("Your task" in str(s.get("content")) for s in working)
    tool_steps = [s for s in working if s["role"] == "tool"]
    assert tool_steps[0]["content"].startswith("fork report")
    [run] = agents.runs()
    assert run.agent_type == "fork" and run.status == "completed"


def test_fork_needs_the_main_conversation(agents, fake_model):
    """Called outside a main loop (no conversation to copy), a fork is an error, not an empty fork."""
    from dostuff.tools.agent import agent

    result = asyncio.run(agent(description="d", prompt="x", mcp_client=object(), session_id="s1",  # type: ignore[arg-type]
                               turn_id="t1", subagent_type="fork", adapter=FakeAdapter()))
    assert result.startswith("[error]") and "fork" in result
    assert agents.runs() == [] and fake_model.calls == []


# --- T6.2 fork model and tools ------------------------------------------------------------------

def test_fork_model_and_tools(agents, fake_model, monkeypatch, agents_home, tmp_path):
    from dostuff.agent.subagents import RESTRICTED_SUBAGENT_TOOLS
    from dostuff.agent.subagents.definitions import load_definitions
    from dostuff.config import get_config
    from dostuff.tools.definitions import default_tool_names
    cfg = get_config()
    monkeypatch.setattr(cfg, "_session_model", {"name": "openai/m-main", "provider": "openai"})
    monkeypatch.setattr(cfg, "agents_models", {"fast": "openai/m-fast"})
    fake_model.script("[MAIN]", fake_response(tool_calls=[_fork_call(model="fast")]), fake_response("main done"))
    fake_model.script("[F]", fake_response("fork report"))
    working = [dict(s) for s in EARLIER]

    asyncio.run(_main_turn(working))

    [fork_first] = _fork_calls(fake_model)
    assert fork_first["model"] == "openai/m-main"
    report = next(s for s in working if s["role"] == "tool")["content"]
    assert "[warning]" in report and "model" in report and "fast" in report and "ignored" in report
    assert report.rstrip().endswith(FOOTER.format(id=agents.runs()[0].id))
    main_tools = _tool_names(fake_model.calls[0])
    assert main_tools == set(default_tool_names())
    assert _tool_names(fork_first) == main_tools - RESTRICTED_SUBAGENT_TOOLS
    # a file can't take the fork name (T3.4 guard)
    project = tmp_path / "proj"
    (project / ".dostuff" / "agents").mkdir(parents=True)
    (project / ".dostuff" / "agents" / "fork.md").write_text("---\nname: fork\ndescription: mine\n---\nbody\n")
    loaded = load_definitions(project, agents_home)
    assert "fork" not in loaded.types and any("reserved" in w for w in loaded.warnings)


def test_fork_keeps_a_restricted_main_tool_list(agents, fake_model):
    """A main loop with its own tool list: the fork gets that list minus D12, not every tool."""
    fake_model.script("[MAIN]", fake_response(tool_calls=[_fork_call()]), fake_response("main done"))
    fake_model.script("[F]", fake_response("fork report"))
    main_tools = ["read_file", "glob", "agent", "send_message", "todo_write"]

    asyncio.run(_main_turn([dict(s) for s in EARLIER], tool_names=main_tools))

    [fork_first] = _fork_calls(fake_model)
    assert _tool_names(fork_first) == {"read_file", "glob"}


# --- T6.3 fork is an ordinary run otherwise -----------------------------------------------------

class _Store:
    """Records agent rows and steps like the session store does."""

    def __init__(self):
        self.rows: dict = {}
        self.steps: dict = {}

    async def upsert_agent_run(self, row):
        self.rows[row["id"]] = dict(row)

    async def append_agent_step(self, agent_id, step):
        self.steps.setdefault(agent_id, []).append(dict(step))

    async def compact_agent_steps(self, agent_id, keep_last, notes):
        pass


def test_fork_background_follow_up(agents, fake_model):
    from dostuff.agent.subagents.fork import FORK_DIRECTIVE
    from dostuff.helpers.agent.notices import get_notice_queue
    from dostuff.tools.send_message import send_message
    store = _Store()
    agents.attach_store(store)
    fake_model.script("[MAIN]", fake_response(tool_calls=[_fork_call(run_in_background=True)]),
                      fake_response("main done"))
    fake_model.script("[F]", fake_response("fork report 1"), fake_response("fork report 2"))
    working = [dict(s) for s in EARLIER]

    async def go():
        await _main_turn(working)
        [run] = agents.runs()
        await run.done
        notices = get_notice_queue().drain("s1")
        reply = await send_message(to=run.id, message="[F] and then?", mcp_client=object(),  # type: ignore[arg-type]
                                   session_id="s1", turn_id="t2", adapter=FakeAdapter())
        await agents.flush()
        return run, notices, reply

    run, notices, reply = asyncio.run(go())
    assert "Started agent" in next(s for s in working if s["role"] == "tool")["content"]
    assert [n.id for n in notices] == [run.id] and "fork report 1" in notices[0].report
    assert reply == f"fork report 2\n\n{FOOTER.format(id=run.id)}"
    second = _fork_calls(fake_model)[-1]["messages"]
    contents = [m.get("content") for m in second[1:]]
    # its own history: the copy it started from, its report, the follow-up; no fresh copy of the main one
    assert sum(FORK_DIRECTIVE.format(prompt="[F] X") in str(c) for c in contents) == 1
    assert contents[-2:] == ["fork report 1", "[F] and then?"]
    assert "main done" not in contents
    assert second[0] == fake_model.calls[0]["messages"][0]  # the same system message as when it started
    # its steps are stored under its id, never in the main history
    stored = [s.get("content") for s in store.steps[run.id]]
    assert stored[-4:] == [FORK_DIRECTIVE.format(prompt="[F] X"), "fork report 1", "[F] and then?", "fork report 2"]
    assert store.rows[run.id]["agent_type"] == "fork" and store.rows[run.id]["status"] == "completed"


def test_fork_worktree(agents, fake_model, tmp_path, monkeypatch):
    import subprocess

    from dostuff.helpers.tools.project_root import project_root
    root = tmp_path / "repo"
    root.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@example.com"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    (root / "a.txt").write_text("a\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True, capture_output=True)
    monkeypatch.chdir(root)
    seen: list = []
    fake = fake_model.acompletion

    async def spy(**kwargs):
        if fake_model._marker(kwargs["messages"]) == "[F]":
            seen.append(project_root())
        return await fake(**kwargs)
    from dostuff.agent import call_agent as call_agent_module
    monkeypatch.setattr(call_agent_module, "acompletion", spy)
    fake_model.script("[MAIN]", fake_response(tool_calls=[_fork_call(isolation="worktree")]),
                      fake_response("main done"))
    fake_model.script("[F]", fake_response("fork report"))
    working = [dict(s) for s in EARLIER]

    asyncio.run(_main_turn(working))

    assert seen and seen[0] != root.resolve() and ".dostuff" in str(seen[0])
    report = next(s for s in working if s["role"] == "tool")["content"]
    assert "worktree" in report and "removed" in report


def test_main_prompt_fork_guidance(monkeypatch):
    """The delegating section says to fork only when the task needs the conversation, and otherwise to use a
    typed agent with a self-contained prompt (D11, D13)."""
    from dostuff.agent.instructions import build_base_instructions
    from dostuff.config import get_config
    for wake in (True, False):
        monkeypatch.setattr(get_config(), "agents_wake_on_end", wake)
        for interactive in (True, False):
            base, _ = build_base_instructions(interactive=interactive)
            text = base.split("## Delegating to subagents", 1)[1].split("\n## ", 1)[0]
            assert 'subagent_type="fork"' in text
            assert "needs this conversation" in text
            assert "self-contained" in text


@pytest.mark.parametrize("name", ["fork"])
def test_fork_is_listed_as_a_type(name, agents_home):
    """`agent`'s error for an unknown type and the main prompt's type list name `fork` too."""
    from dostuff.agent.instructions import agent_types_list
    from dostuff.tools.agent import agent
    assert f"- {name}:" in agent_types_list()
    result = asyncio.run(agent(description="d", prompt="x", mcp_client=object(), session_id="s1",  # type: ignore[arg-type]
                               turn_id="t1", subagent_type="nope"))
    assert name in result


# --- Cache-stable context, T4.1 ----------------------------------------------------------------------------------

def test_fork_copies_context_steps(agents, fake_model):
    from dostuff.agent.subagents.fork import FORK_DIRECTIVE
    from dostuff.helpers.agent.turn_context import make_context_step
    fake_model.script("[MAIN]", fake_response(tool_calls=[_fork_call()]), fake_response("main done"))
    fake_model.script("[F]", fake_response("fork report"))
    bundle = "<turn-context>\n<environment>\ncwd: /x\n</environment>\n</turn-context>"
    working = [make_context_step(bundle), *[dict(s) for s in EARLIER]]

    asyncio.run(_main_turn(working))

    main_first = fake_model.calls[0]
    [fork_first] = _fork_calls(fake_model)
    assert fork_first["messages"][0] == main_first["messages"][0]  # the same system string
    kept = fork_first["messages"][1]
    assert kept["content"].startswith(bundle + "\n\n") and "meta" not in kept  # kept, and no `meta` on the wire
    assert fork_first["messages"][1:-1] == main_first["messages"][1:-1]  # so the copied prefix still matches
    assert fork_first["messages"][-1]["content"].endswith(FORK_DIRECTIVE.format(prompt="[F] X"))


# --- a /btw fork-out and the live snapshot -----------------------------------------------------------------------

def test_fork_out_leaves_the_live_snapshot_for_the_next_side_question(agents, fake_model):
    from dostuff.agent.subagents.fork import ForkContext
    from dostuff.helpers.agent.side_question import build_messages, discard_snapshot, publish_snapshot, snapshot
    fake_model.script("[F]", fake_response("fork report"))

    async def go():
        publish_snapshot("fork-out", ForkContext("sys", "", [dict(s) for s in EARLIER], ["read_file"], None))
        run = await agents.start(owner="fork-out", agent_type="fork", background=True, description="x",
                                 prompt="[F] X", fork=snapshot("fork-out"), adapter=FakeAdapter())
        while run.ended is None:
            await asyncio.sleep(0.01)
        return run

    run = asyncio.run(go())
    assert run.status == "completed" and fake_model.calls_for("[F]")  # the fork did take its copy
    messages = build_messages(snapshot("fork-out"), [], "what was the word?")
    discard_snapshot("fork-out")
    assert messages[:len(EARLIER)] == EARLIER
