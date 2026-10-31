"""delegate_to_subagent wiring (REQ-2.1 – 2.4)."""
import asyncio
from types import SimpleNamespace

from dostuff.agent import call_agent as call_agent_module
from dostuff.tools.delegate_to_subagent import _impl_delegate_to_subagent, resolve_subagent_tools

from conftest import FakeStep, fake_response, fake_tool_call


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

def test_subagent_jobs_killed(fake_model, _start_subagent, monkeypatch, registry, tmp_path, _py, _wait_dead):
    import pytest

    from dostuff.agent import loop as loop_module
    from dostuff.agent.permissions import SessionPermissions
    from dostuff.tools.bash import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "_registry", registry)
    monkeypatch.setattr(loop_module, "retry_delay", lambda *a: 0.0)
    sleeper = _py("import time; time.sleep(30)")

    def start_call(job_id):
        args = {"command": sleeper, "run_in_background": True, "background_id": job_id}
        return fake_response(tool_calls=[fake_tool_call("bash_command", args)])

    def delegate(marker):
        return _start_subagent(f"[{marker}] t", tools=["bash_command"], permissions=SessionPermissions(mode="auto"))

    def check(job_id):
        [job] = [j for j in registry.jobs() if j.id == job_id]
        assert job.owner.startswith("s/sub/")
        assert registry.owner_label(job.owner) == "subagent: pin"  # what /jobs shows (T6.2)
        assert _wait_dead(job.pid, 5) and job.status == "killed" and job.killed_by == "owner_ended"
        assert registry.drain_notices(job.owner) == [] and registry.drain_notices("s") == []

    # returns normally
    fake_model.script("[ok]", start_call("ok"), fake_response("done"))
    assert "done" in asyncio.run(delegate("ok"))
    check("ok")

    # the model call fails
    fake_model.script("[err]", start_call("err"), ValueError("model blew up"), fake_response("wrap-up"))
    assert "model blew up" in asyncio.run(delegate("err"))
    check("err")

    # cancelled while the model is thinking
    hang = asyncio.Event()
    fake_model.script("[cancel]", start_call("cancel"))
    fake_model.script("[cancel]", FakeStep(fake_response("never"), gate=hang))

    async def cancel_it():
        task = asyncio.create_task(delegate("cancel"))
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


# --- subagents rebuild, Phase 1 pins (plan/2026-11-01-subagents/tasks.md T1.1) ----------------------
# Every pin starts the subagent through `_start_subagent` (conftest); T2.6 swaps only that helper.

def test_pin_final_report(fake_model, _start_subagent):
    fake_model.script("[A]", fake_response("done"))
    assert asyncio.run(_start_subagent("[A] summarize")) == "done"


def test_pin_partial_report_on_cap(fake_model, _start_subagent, monkeypatch):
    from dostuff.config import get_config
    monkeypatch.setattr(get_config(), "subagent_max_iterations", 1)
    fake_model.script("[A]", fake_response(tool_calls=[fake_tool_call("get_current_datetime", {})]),
                      fake_response("wrapped up"))
    result = asyncio.run(_start_subagent("[A] keep going", tools=["get_current_datetime"]))
    assert "did not finish" in result
    assert "Partial report:\nwrapped up" in result
    wrap_ups = [c for c in fake_model.calls if "run out of steps" in str(c["messages"][-1].get("content"))]
    assert len(wrap_ups) == 1  # one wrap-up call


def test_pin_dropped_tools_warning(fake_model, _start_subagent):
    fake_model.script("[A]", fake_response("the report"))
    result = asyncio.run(_start_subagent("[A] x", tools=["read_file", "delete_file", "run_code"]))
    first, _, rest = result.partition("\n\n")
    assert first.startswith("[warning]") and "delete_file" in first and "run_code" in first
    assert rest == "the report"
