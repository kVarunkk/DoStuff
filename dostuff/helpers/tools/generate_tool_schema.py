import inspect
import re
import typing
from typing import Any, Callable, Iterable
from pydantic import TypeAdapter

_SECTION_HEADERS = ("Args:", "Arguments:", "Returns:", "Return:", "Raises:", "Yields:", "Examples:", "Example:", "Note:", "Notes:")
_ARG_LINE = re.compile(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def parse_google_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, {param: description}).

    Only the `Args:` section is read; multi-line continuations (lines indented deeper
    than the param line) are joined into the param's description.
    """
    lines = docstring.splitlines()
    summary_lines: list[str] = []
    i = 0
    while i < len(lines) and lines[i].strip() not in _SECTION_HEADERS:
        summary_lines.append(lines[i])
        i += 1
    summary = "\n".join(summary_lines).strip()

    params: dict[str, str] = {}
    while i < len(lines):
        if lines[i].strip() in ("Args:", "Arguments:"):
            section_indent = _indent(lines[i])
            i += 1
            param_indent: int | None = None
            current: str | None = None
            while i < len(lines):
                line = lines[i]
                if not line.strip():
                    i += 1
                    continue
                ind = _indent(line)
                if ind <= section_indent:
                    break
                if param_indent is None:
                    param_indent = ind
                match = _ARG_LINE.match(line.strip())
                if ind == param_indent and match:
                    current = match.group(1).lstrip("*")
                    params[current] = match.group(2).strip()
                elif current is not None:
                    params[current] = f"{params[current]} {line.strip()}".strip()
                i += 1
            continue
        i += 1
    return summary, params


def _strip_titles(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


def _inline_refs(schema: Any, defs: dict[str, Any] | None = None) -> Any:
    """Replace `{"$ref": "#/$defs/X"}` with X's schema: a property's own `$defs` would not resolve
    once nested inside the tool's parameters."""
    if isinstance(schema, dict):
        if defs is None:
            defs = schema.get("$defs", {})
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            return _inline_refs(defs[ref.removeprefix("#/$defs/")], defs)
        return {k: _inline_refs(v, defs) for k, v in schema.items() if k != "$defs"}
    if isinstance(schema, list):
        return [_inline_refs(v, defs) for v in schema]
    return schema


def generate_litellm_tool_schema(
    func: Callable[..., Any],
    name: str | None = None,
    exclude: Iterable[str] = (),
    description_suffix: str = "",
) -> dict:
    """Convert a Python function into a LiteLLM / OpenAI tool schema.

    Parameter types come from `pydantic.TypeAdapter(annotation).json_schema()`, so
    `dict` → object, `list[str]` → array with items, `X | None` → nullable.
    Parameters named in `exclude` (harness-injected) or starting with `_` are skipped.
    """
    signature = inspect.signature(func)
    hints = typing.get_type_hints(func)
    summary, arg_docs = parse_google_docstring(inspect.getdoc(func) or "")
    excluded = set(exclude)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in signature.parameters.items():
        if pname in ("self", "cls") or pname in excluded or pname.startswith("_"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(pname, str)
        prop = _strip_titles(_inline_refs(TypeAdapter(annotation).json_schema()))
        prop["description"] = arg_docs.get(pname) or f"The {pname} parameter."
        properties[pname] = prop
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    description = summary
    if description_suffix:
        description = f"{description}\n\n{description_suffix}".strip()

    return {
        "type": "function",
        "function": {
            "name": name or func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
