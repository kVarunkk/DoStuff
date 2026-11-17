"""Merge overlapping saved facts on request (memory recall D10): one model call, a diff, a write only on confirm.

Never automatic and never in auto mode: a merge can lose information, so a person reads the diff first.
"""
import json
from dataclasses import dataclass, field
from typing import Any

from litellm import acompletion, ModelResponse
from dostuff.lib.model import completion_kwargs

SCOPES = ("user", "project")


class ConsolidateError(Exception):
    """The model's answer can't be used; nothing was written."""


@dataclass
class ScopeChanges:
    """What consolidating one scope's facts would do."""
    scope: str
    working_dir: str | None
    removed: list[dict] = field(default_factory=list)                    # saved facts that go
    changed: list[tuple[dict, dict]] = field(default_factory=list)       # (saved, new value) under the same key
    added: list[dict] = field(default_factory=list)                      # merged facts under a new key

    @property
    def empty(self) -> bool:
        return not (self.removed or self.changed or self.added)


def _prompt(current: dict[str, list[dict]]) -> str:
    return f"""You tidy a coding assistant's saved facts about its user ("user") and about one project ("project").
Merge facts that overlap or repeat into one, and drop a fact only when another fact already says it. Keep every
other fact exactly as it is, with its key. Never move a fact between "user" and "project", never invent
information, and keep a merged fact's key one of the keys it replaces where you can.

Return JSON only, the same shape as the input: {{"user": [{{"key": ..., "value": ...}}], "project": [...]}}

{json.dumps(current, indent=2, ensure_ascii=False)}
"""


def _parse(raw: str, current: dict[str, list[dict]]) -> dict[str, list[dict]]:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        answer = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as error:
        raise ConsolidateError(f"the model's answer is not JSON ({error})") from None
    if not isinstance(answer, dict) or set(answer) - set(current):
        raise ConsolidateError("the model's answer is not the expected {\"user\": [...], \"project\": [...]} object")
    merged: dict[str, list[dict]] = {}
    for scope, facts in current.items():
        new = answer.get(scope)
        if not isinstance(new, list) or (facts and not new):
            raise ConsolidateError(f"the model's answer has no {scope} facts")
        keys: set[str] = set()
        rows: list[dict] = []
        for item in new:
            key = str(item.get("key") or "").strip() if isinstance(item, dict) else ""
            value = str(item.get("value") or "").strip() if isinstance(item, dict) else ""
            if not key or not value or key in keys:
                raise ConsolidateError(f"the model's answer has a {scope} fact without a key or value, or a key twice")
            keys.add(key)
            rows.append({"key": key, "value": value})
        merged[scope] = rows
    return merged


def diff(scope: str, working_dir: str | None, saved: list[dict], merged: list[dict]) -> ScopeChanges:
    changes = ScopeChanges(scope, working_dir)
    by_key = {f["key"]: f for f in saved}
    merged_keys = {f["key"] for f in merged}
    changes.removed = [f for f in saved if f["key"] not in merged_keys]
    for fact in merged:
        old = by_key.get(fact["key"])
        if old is None:
            changes.added.append(fact)
        elif old["value"] != fact["value"]:
            changes.changed.append((old, fact))
    return changes


async def propose(store: Any, user_id: str, working_dir: str) -> list[ScopeChanges]:
    """The changes a merge of this user's facts and this project's would make, from one model call; [] when there
    is nothing to merge (fewer than two facts). Raises ConsolidateError on an answer that can't be trusted."""
    saved = {scope: await store.list_all(user_id, scope=scope, working_dir=working_dir) for scope in SCOPES}
    if sum(len(facts) for facts in saved.values()) < 2:
        return []
    current = {scope: [{"key": f["key"], "value": f["value"]} for f in facts] for scope, facts in saved.items()}
    try:
        response = await acompletion(**completion_kwargs(), messages=[{"role": "user", "content": _prompt(current)}])
        raw = (response.choices[0].message.content or "") if isinstance(response, ModelResponse) else ""
    except Exception as error:
        raise ConsolidateError(f"the model call failed ({error})") from None
    merged = _parse(raw, current)
    changes = [diff(scope, working_dir if scope == "project" else None, saved[scope], merged[scope]) for scope in SCOPES]
    return [c for c in changes if not c.empty]


def describe(changes: list[ScopeChanges], project_name: str) -> str:
    """The diff as text: `-` a fact that goes, `+` a new one, `~` a changed value."""
    sections = []
    for c in changes:
        title = "User facts" if c.scope == "user" else f"This project's facts ({project_name})"
        lines = ([f"- {f['key']}: {f['value']}" for f in c.removed]
                 + [f"~ {old['key']}: {old['value']}  →  {new['value']}" for old, new in c.changed]
                 + [f"+ {f['key']}: {f['value']}" for f in c.added])
        sections.append(f"{title}:\n" + "\n".join(lines))
    return "\n\n".join(sections)


def summary(changes: list[ScopeChanges]) -> str:
    """'2 removed, 1 changed, 1 added' (only the non-zero parts)."""
    counts = [(sum(len(c.removed) for c in changes), "removed"), (sum(len(c.changed) for c in changes), "changed"),
              (sum(len(c.added) for c in changes), "added")]
    return ", ".join(f"{n} {label}" for n, label in counts if n)


async def apply(store: Any, user_id: str, changes: list[ScopeChanges]) -> None:
    """Write the merge: new and changed facts first, then the deletes, so a failure part-way loses nothing.
    Through `upsert_fact`, so a kept key keeps its created date and gets a new updated one."""
    for c in changes:
        for old, new in c.changed:
            await store.upsert_fact(user_id, {**new, "source": old.get("source")}, scope=c.scope, working_dir=c.working_dir)
        for new in c.added:
            await store.upsert_fact(user_id, new, scope=c.scope, working_dir=c.working_dir)
    for c in changes:
        for old in c.removed:
            await store.delete(user_id, old["key"], scope=c.scope, working_dir=c.working_dir)
