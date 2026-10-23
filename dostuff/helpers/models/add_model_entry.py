"""Add a model to the `models:` list of a config.yaml, keeping the file's comments and layout."""
import json
import re
from pathlib import Path
from typing import Any

import yaml

# A top-level `models:` line; its body is the indented and blank lines after it
_HEADER = re.compile(r"(?m)^models:[^\n]*(?:\n|\Z)")


class ModelConfigError(ValueError):
    """The entry can't be added: a duplicate, or a config.yaml that isn't a mapping / list of entries."""


class _CantEdit(Exception):
    """The block has a shape the text edit doesn't handle; rewrite the block instead."""


def _load(text: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text) if text.strip() else {}
    except yaml.YAMLError as e:
        raise ModelConfigError(f"config.yaml is not valid YAML: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ModelConfigError("config.yaml is not a mapping")
    return data


def _entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("models")
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(e, dict) for e in raw):
        raise ModelConfigError("`models:` in config.yaml is not a list of entries")
    return raw


def _render(entry: dict[str, Any], indent: str) -> str:
    """One list entry, `name` first; JSON scalars are valid YAML."""
    lines = [f"{indent}- name: {json.dumps(entry['name'], ensure_ascii=False)}"]
    lines += [f"{indent}  {k}: {json.dumps(v, ensure_ascii=False)}" for k, v in entry.items() if k != "name"]
    return "\n".join(lines)


def _block_span(text: str) -> tuple[int, int, int] | None:
    """(start, body_start, end) of the top-level `models:` block, or None when there is none."""
    m = _HEADER.search(text)
    if not m:
        return None
    pos = m.end()
    for line in text[pos:].splitlines(keepends=True):
        if line.strip() and not line[0].isspace():
            break
        pos += len(line)
    return m.start(), m.end(), pos


def _append_block(text: str, block: str) -> str:
    return text.rstrip("\n") + ("\n\n" if text.strip() else "") + block.rstrip("\n") + "\n"


def _replace_block(text: str, entries: list[dict[str, Any]]) -> str:
    """`text` with its `models:` block (comments inside it included) rewritten from `entries`."""
    block = "models:\n" + "".join(_render(e, "  ") + "\n" for e in entries) + "\n"
    span = _block_span(text)
    if span is None:
        return _append_block(text, block)
    start, _, end = span
    return text[:start] + block + text[end:]


def _edit_block(text: str, entry: dict[str, Any], is_new: bool, active: bool) -> str:
    """Append `entry` (or mark the listed one active) by editing only the lines that change."""
    span = _block_span(text)
    if span is None:
        return _append_block(text, "models:\n" + _render(entry, "  "))
    start, body_start, end = span
    header, body = text[start:body_start], text[body_start:end]
    if re.sub(r"#.*", "", header[len("models:"):]).strip():
        raise _CantEdit  # flow style, e.g. `models: []`
    if active:
        body = re.sub(r"(?m)^([ \t]*(?:-[ \t]+)?active:[ \t]*)true\b", r"\1false", body)
    if is_new:
        first = re.search(r"(?m)^([ \t]*)-", body)
        content = body.rstrip()
        blank_lines = max(body[len(content):].count("\n") - 1, 0)
        body = (content + "\n" if content else "") + _render(entry, first.group(1) if first else "  ") + "\n" + "\n" * blank_lines
    else:
        name = re.escape(entry["name"])
        m = re.search(rf"""(?m)^([ \t]*)-[ \t]+name:[ \t]*["']?{name}["']?[ \t]*(?:#.*)?$""", body)
        if not m:
            raise _CantEdit
        nxt = re.compile(rf"(?m)^{re.escape(m.group(1))}-").search(body, m.end())
        seg_end = nxt.start() if nxt else len(body)
        seg, n = re.subn(r"(?m)^([ \t]*active:[ \t]*)\S+", r"\1true", body[m.end():seg_end], count=1)
        if not n:
            seg = f"\n{m.group(1)}  active: true" + seg
        body = body[:m.end()] + seg + body[seg_end:]
    return text[:start] + header + body + text[end:]


def add_model_entry(path: Path, name: str, provider: str, api_key_env: str = "", active: bool = False,
                    template: str = "") -> bool:
    """Add `name` to the `models:` list in `path`; `active` makes it the default (and clears the others).

    A missing file is created from `template` with this model as its only, active entry. A name that
    is already listed is only made active (`active=True`), else ModelConfigError. Returns True when a
    new entry was written. The edited text must parse to the expected config, or nothing is written.
    """
    exists = path.exists()
    text = path.read_text(encoding="utf-8") if exists else template
    data = _load(text)
    entries = _entries(data) if exists else []
    active = active or not exists
    is_new = name not in [str(e.get("name")) for e in entries]
    if not is_new and not active:
        raise ModelConfigError(f"'{name}' is already in {path}")
    entry: dict[str, Any] = {"name": name, "provider": provider, "active": active}
    if api_key_env:
        entry["api_key_env"] = api_key_env

    expected = [dict(e) for e in entries] + ([entry] if is_new else [])
    if active:
        for e in expected:
            if "active" in e or str(e.get("name")) == name:
                e["active"] = str(e.get("name")) == name
    others = {k: v for k, v in data.items() if k != "models"}

    def matches(candidate: str) -> bool:
        try:
            new = _load(candidate)
        except ModelConfigError:
            return False
        return new.get("models") == expected and {k: v for k, v in new.items() if k != "models"} == others

    result = None
    if exists:
        try:
            result = _edit_block(text, entry, is_new, active)
        except _CantEdit:
            pass
    if result is None or not matches(result):
        # A new file, or a block the line edit can't handle: rewrite just the `models:` block
        result = _replace_block(text, expected)
        if not matches(result):
            raise ModelConfigError(f"could not update the `models:` list in {path}; edit it by hand")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result, encoding="utf-8")
    return is_new
