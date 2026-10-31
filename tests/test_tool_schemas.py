"""Schema snapshot + OpenAI strict-mode sanity checks (REQ-3.1 – 3.3, REQ-1.1)."""
import json
import os
import re
from pathlib import Path

import pytest

from dostuff.helpers.tools.generate_tool_schema import generate_litellm_tool_schema, parse_google_docstring
from dostuff.tools.definitions import TOOL_SCHEMAS

SNAPSHOT = Path(__file__).parent / "snapshots" / "tool_schemas.json"
APPROVAL_PARAMS = {"_confirmed", "overwrite", "confirmed"}
INJECTED = {"mcp_client", "session_id", "turn_id", "adapter", "permissions", "owner"}


def _normalized() -> dict:
    """Schemas with the machine-dependent shell name masked."""
    schemas = json.loads(json.dumps(TOOL_SCHEMAS))
    fn = schemas["bash_command"]["function"]
    fn["description"] = re.sub(r"Commands run in \S+\.", "Commands run in <shell>.", fn["description"])
    return schemas


def _walk(node, path=""):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}/{i}")


def test_snapshot_matches():
    current = _normalized()
    if os.environ.get("UPDATE_SNAPSHOTS") or not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert current == json.loads(SNAPSHOT.read_text(encoding="utf-8")), (
        "Tool schemas changed. If intended, rerun with UPDATE_SNAPSHOTS=1."
    )


@pytest.mark.parametrize("name", sorted(TOOL_SCHEMAS))
def test_no_approval_or_injected_params(name):
    props = TOOL_SCHEMAS[name]["function"]["parameters"]["properties"]
    assert not any(p.startswith("_") for p in props)
    assert not (set(props) & APPROVAL_PARAMS)
    assert not (set(props) & INJECTED)


@pytest.mark.parametrize("name", sorted(TOOL_SCHEMAS))
def test_strict_mode_shape(name):
    params = TOOL_SCHEMAS[name]["function"]["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) <= set(params["properties"])
    for path, node in _walk(params):
        if node.get("type") == "array":
            assert "items" in node, f"{name}{path}: array without items"
    for prop in params["properties"].values():
        assert prop.get("description")


def test_call_mcp_tool_arguments_is_object():
    props = TOOL_SCHEMAS["call_mcp_tool"]["function"]["parameters"]["properties"]
    assert props["arguments"]["type"] == "object"


def test_requested_tools_is_nullable_string_array():
    prop = TOOL_SCHEMAS["delegate_to_subagent"]["function"]["parameters"]["properties"]["requested_tools"]
    assert {"type": "array", "items": {"type": "string"}} in prop["anyOf"]
    assert {"type": "null"} in prop["anyOf"]


def test_docstring_args_with_continuations():
    doc = """Do a thing.

    Longer explanation.

    Args:
        first: The first line
            continues here.
        second (int): Second param.

    Returns:
        first: not a param.
    """
    import inspect
    summary, params = parse_google_docstring(inspect.cleandoc(doc))
    assert summary == "Do a thing.\n\nLonger explanation."
    assert params == {"first": "The first line continues here.", "second": "Second param."}


def test_types_and_exclusions():
    def sample(a: int, b: bool, c: dict[str, int], d: list[str] | None = None, mcp_client: object = None, _x: int = 0) -> str:
        """Sample.

        Args:
            a: A.
        """
        return ""

    schema = generate_litellm_tool_schema(sample, exclude={"mcp_client"})
    props = schema["function"]["parameters"]["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "boolean"
    assert props["c"]["type"] == "object"
    assert set(props) == {"a", "b", "c", "d"}
    assert schema["function"]["parameters"]["required"] == ["a", "b", "c"]
