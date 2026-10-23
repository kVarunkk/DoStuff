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
    assert _tool_names(captured) == {"read_file", "bash_command"}
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
