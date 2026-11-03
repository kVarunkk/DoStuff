"""Pin litellm's handling of a tool-call history with and without `tools=` (btw spec D2, tasks.md T1.6).

The side call sends the turn's tool schemas rather than `[]`. These tests record why, against the installed
litellm: if an upgrade changes either behaviour, they fail and D2 is re-read, not the test skipped. Kept out of
tests/test_side_question.py because importing litellm costs seconds.
"""
from litellm.llms.anthropic.chat.transformation import AnthropicConfig

from tests.conftest import make_fork_ctx

READ_FILE = {"type": "function", "function": {
    "name": "read_file", "description": "Read a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}


def _transform(optional_params):
    messages = [dict(s) for s in make_fork_ctx().history] + [{"role": "user", "content": "where is retry?"}]
    return AnthropicConfig().transform_request(
        model="claude-sonnet-4-5", messages=messages,  # type: ignore[arg-type]
        optional_params=optional_params, litellm_params={}, headers={})


def _block_types(request):
    return [[b.get("type") for b in m["content"]] if isinstance(m["content"], list) else ["text"]
            for m in request["messages"]]


def test_anthropic_tool_history_with_tools():
    request = _transform({"tools": [READ_FILE]})
    assert ["tool_use"] in _block_types(request)
    assert any("tool_result" in types for types in _block_types(request))
    assert [t.get("name") or t["function"]["name"] for t in request["tools"]] == ["read_file"]


def test_anthropic_tool_history_without_tools():
    # litellm 1.106.2: no error and the tool_use/tool_result blocks are kept, but a placeholder tool is injected.
    # So `tools=[]` would not fail; it would send a *different* tool list from the turn's, which is a different
    # cached prefix: the side question would pay full price (D2).
    request = _transform({})
    assert ["tool_use"] in _block_types(request)
    assert [t.get("name") or t["function"]["name"] for t in request["tools"]] == ["dummy_tool"]
