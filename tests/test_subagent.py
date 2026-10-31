"""delegate_to_subagent wiring (REQ-2.1 – 2.4)."""
import asyncio
from types import SimpleNamespace

from dostuff.agent import call_agent as call_agent_module
from dostuff.tools.delegate_to_subagent import _impl_delegate_to_subagent, resolve_subagent_tools


def _fake_response(text: str):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _run(monkeypatch, requested_tools=None, adapter=None):
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _fake_response("done")

    monkeypatch.setattr(call_agent_module, "acompletion", fake_acompletion)
    result = asyncio.run(_impl_delegate_to_subagent(
        task="Summarize README.md",
        role="researcher",
        mcp_client=object(),  # type: ignore[arg-type]
        session_id="s",
        turn_id="t",
        requested_tools=requested_tools,
        adapter=adapter,
    ))
    return result, captured


def _tool_names(captured) -> set[str]:
    return {t["function"]["name"] for t in captured["tools"]}


def test_tools_exclude_restricted(monkeypatch):
    result, captured = _run(monkeypatch, ["read_file", "delegate_to_subagent", "delete_file", "bash_command"])
    # bash_command brings its job tools along (background shell spec D10)
    assert _tool_names(captured) == {"read_file", "bash_command", "bash_output", "kill_shell"}
    assert "delegate_to_subagent" in result and "delete_file" in result  # warning returned to the parent


def test_default_tools_all_exist(monkeypatch):
    allowed, dropped = resolve_subagent_tools(None)
    assert not dropped
    _, captured = _run(monkeypatch)
    assert _tool_names(captured) == set(allowed)
    assert "delegate_to_subagent" not in _tool_names(captured)
    assert "delete_file" not in _tool_names(captured)


def test_unknown_tools_dropped_with_warning(monkeypatch):
    result, captured = _run(monkeypatch, ["read_file", "run_code"])
    assert _tool_names(captured) == {"read_file"}
    assert "run_code" in result


def test_seed_message_is_valid_chat_message(monkeypatch):
    _, captured = _run(monkeypatch)
    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Summarize README.md"}
    assert all("role" in m for m in messages)


def test_output_routed_through_parent_adapter(monkeypatch):
    events: list = []

    class Parent:
        _cancelled = False
        _cancel_event = asyncio.Event()

        def emit(self, kind, data):
            events.append((kind, data))

        async def ask(self, message):
            return "no"

    _run(monkeypatch, adapter=Parent())
    # no tool calls in this fake run, but the cancel state was read through the wrapper without error
    assert all(not isinstance(d, str) or d.startswith("[subagent: researcher]") for k, d in events if k != "usage")


# --- background shell T5.3: a subagent's jobs die with it --------------------------------------

def test_subagent_jobs_killed(monkeypatch, registry, tmp_path, _py, _wait_dead):
    import json

    import pytest

    from dostuff.agent import loop as loop_module
    from dostuff.agent.permissions import SessionPermissions
    from dostuff.tools import delegate_to_subagent as delegate_module
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    monkeypatch.setattr(loop_module, "get_model_token_limit", lambda: 100_000)

    def start_call(job_id):
        args = json.dumps({"command": sleeper, "run_in_background": True, "background_id": job_id})
        tc = SimpleNamespace(id="c1", type="function", function=SimpleNamespace(name="bash_command", arguments=args))
        tc.model_dump = lambda: {"id": "c1", "type": "function", "function": {"name": "bash_command", "arguments": args}}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))], usage=None)

    sleeper = _py("import time; time.sleep(30)")
    hang = asyncio.Event()

    def script(job_id, then):
        responses = [start_call(job_id), then]

        async def fake_call_agent(steps_history, system_instruction, tool_names=None, stream=False):
            item = responses.pop(0)
            if item == "hang":
                await hang.wait()
            return item, None
        monkeypatch.setattr(loop_module, "call_agent", fake_call_agent)

    def delegate():
        return delegate_module._impl_delegate_to_subagent(
            task="t", role="r", mcp_client=object(), session_id="s", turn_id="t",  # type: ignore[arg-type]
            requested_tools=["bash_command"], permissions=SessionPermissions(mode="auto"),
        )

    def check(job_id):
        [job] = [j for j in registry.jobs() if j.id == job_id]
        assert job.owner.startswith("s/sub/")
        assert registry.owner_label(job.owner) == "subagent: r"  # what /jobs shows (T6.2)
        assert _wait_dead(job.pid, 5) and job.status == "killed" and job.killed_by == "owner_ended"
        assert registry.drain_notices(job.owner) == [] and registry.drain_notices("s") == []

    # returns normally
    script("ok", _fake_response("done"))
    assert "done" in asyncio.run(delegate())
    check("ok")

    # raises
    real = delegate_module._run_subagent_loop

    async def failing(*args, **kwargs):
        await real(*args, **kwargs)
        raise RuntimeError("subagent blew up")
    monkeypatch.setattr(delegate_module, "_run_subagent_loop", failing)
    script("err", _fake_response("done"))
    with pytest.raises(RuntimeError):
        asyncio.run(delegate())
    check("err")
    monkeypatch.setattr(delegate_module, "_run_subagent_loop", real)

    # cancelled while the model is thinking
    script("cancel", "hang")

    async def cancel_it():
        task = asyncio.create_task(delegate())
        for _ in range(100):
            await asyncio.sleep(0.05)
            if any(j.id == "cancel" for j in registry.jobs()):
                break
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(cancel_it())
    check("cancel")
