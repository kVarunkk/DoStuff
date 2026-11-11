"""The turn-context step (plan/2026-11-05-cache-stable-context/spec.md).

Per-turn context (environment, recalled facts and episodes, UserPromptSubmit context) and the plan-mode text ride
the history as hidden `role:"user"` steps, because anything volatile in the system prompt invalidates the provider's
whole cached prefix. The model reads them; a user never sees them. Every reader of a history skips them through
`is_turn_context`, so the marker lives here and nowhere else (D7).

Light on import: the TUI loads this before its first frame.
"""
import hashlib
from contextvars import ContextVar
from typing import Any, NamedTuple

TURN_CONTEXT_MARKER = "turn_context"
# `step_data` is JSON text, so the title queries match the marker with this. A prompt quoting the same words can't
# false-match: its quotes are escaped in `step_data` (D8). A bare pattern, for binding to `NOT LIKE ?`.
HIDDEN_LIKE_PATTERN = f'%"hidden": "{TURN_CONTEXT_MARKER}"%'
PROMPT_CONTEXT_HEADER = "[Context from UserPromptSubmit hooks]"

# Saved facts delivered whole in the facts step; the rest are recalled per turn by similarity
# (plan/2026-11-11-memory-recall D3). Project facts are capped inside the total so one busy repo
# can't push every user fact out.
FACT_INJECT_CAP = 50
PROJECT_FACT_CAP = 30
FACTS_REPLACE_NOTE = "This replaces any earlier list of saved facts."


class FactsSource(NamedTuple):
    """Where the main session's facts step reads from: the semantic store, the user, the project root."""
    store: Any
    user_id: str
    project: str


# Set by the main session's loop; a typed subagent started from it (its task copies this context) reads the same
# facts, read-only, with the main project even when it runs in a worktree (memory recall D14)
facts_source: ContextVar[FactsSource | None] = ContextVar("dostuff_facts_source", default=None)


def is_turn_context(step: object) -> bool:
    """True for a harness-authored context step: a bundle or a plan step (`meta.hidden == "turn_context"`)."""
    if not isinstance(step, dict):
        return False
    meta = step.get("meta")
    return isinstance(meta, dict) and meta.get("hidden") == TURN_CONTEXT_MARKER


def _wrap(text: str) -> str:
    return f"<turn-context>\n{text}\n</turn-context>"


def compose_turn_context(environment: str = "", facts: str = "", episodes: str = "", prompt_context: str = "") -> str:
    """The bundle: the sections that have text, in a fixed order; "" when none has (D4). Does no I/O.

    `facts` are the saved facts past the facts step's cap that match the prompt (memory recall D3)."""
    sections = [
        environment,
        f"<other_relevant_facts>\n{facts}\n</other_relevant_facts>" if facts else "",
        f"<past_episodes>\n{episodes}\n</past_episodes>" if episodes else "",
        f"{PROMPT_CONTEXT_HEADER}\n{prompt_context}" if prompt_context else "",
    ]
    body = "\n\n".join(s for s in sections if s)
    return _wrap(body) if body else ""


def make_context_step(bundle: str) -> dict[str, Any]:
    """The hidden step that carries a bundle."""
    return {"role": "user", "content": bundle, "meta": {"hidden": TURN_CONTEXT_MARKER}}


def make_plan_mode_step(plan_mode: bool, text: str) -> dict[str, Any]:
    """The hidden step that tells the model plan mode changed (D11). `text` is passed in so this module stays light."""
    return {"role": "user", "content": _wrap(text), "meta": {"hidden": TURN_CONTEXT_MARKER, "plan_mode": plan_mode}}


def make_capability_step(text: str, capabilities: tuple[str, ...]) -> dict[str, Any]:
    """The hidden step that tells the model which skills and agent types changed since it was last told
    (plan/2026-11-11-mid-run-skill-agent-refresh D3). `capabilities` is the set it announced: the next belief."""
    return {"role": "user", "content": _wrap(text),
            "meta": {"hidden": TURN_CONTEXT_MARKER, "capabilities": list(capabilities)}}


def delivered_capabilities(history: list) -> tuple[str, ...] | None:
    """The capability set the newest capability step announced; None when the history has none (never told, or
    compaction dropped it), and the caller falls back to the prompt's baseline."""
    for step in reversed(history):
        if is_turn_context(step) and "capabilities" in step["meta"]:
            return tuple(step["meta"]["capabilities"])
    return None


def delivered_plan_mode(history: list) -> bool:
    """The newest plan step's state; False when the history has none."""
    for step in reversed(history):
        if is_turn_context(step) and "plan_mode" in step["meta"]:
            return bool(step["meta"]["plan_mode"])
    return False


def facts_fingerprint(facts: list[dict]) -> tuple[str, ...]:
    """What a facts step delivered: sorted `scope:key:sha1(value)[:8]`, so any save, update or forget changes it."""
    return tuple(sorted(
        f"{f.get('scope', 'user')}:{f['key']}:{hashlib.sha1(str(f['value']).encode('utf-8')).hexdigest()[:8]}"
        for f in facts))


def make_facts_step(text: str, fingerprint: tuple[str, ...]) -> dict[str, Any]:
    """The hidden step that lists the saved facts (D2). `fingerprint` is what it delivered: the next belief."""
    return {"role": "user", "content": _wrap(text), "meta": {"hidden": TURN_CONTEXT_MARKER, "facts": list(fingerprint)}}


def delivered_facts(history: list) -> tuple[str, ...] | None:
    """The fingerprint of the newest facts step; None when the history has none (never sent, or compacted away)."""
    for step in reversed(history):
        if is_turn_context(step) and "facts" in step["meta"]:
            return tuple(step["meta"]["facts"])
    return None


def select_injected_facts(facts: list[dict]) -> tuple[list[dict], list[dict]]:
    """(shown, overflow): the newest FACT_INJECT_CAP facts, project first on equal timestamps and at most
    PROJECT_FACT_CAP of them, newest first; overflow is everything else."""
    ordered = sorted(facts, key=lambda f: (-int(f.get("updated_at_epoch") or 0), f.get("scope") != "project", f["key"]))
    project_ids = [id(f) for f in ordered if f.get("scope") == "project"][:PROJECT_FACT_CAP]
    allowed = set(project_ids)
    candidates = [f for f in ordered if f.get("scope") != "project" or id(f) in allowed]
    shown = candidates[:FACT_INJECT_CAP]
    shown_ids = {id(f) for f in shown}
    return shown, [f for f in ordered if id(f) not in shown_ids]


def facts_step_content(facts: list[dict], project: str) -> tuple[str, tuple[str, ...]]:
    """(text, fingerprint) of the facts step for every saved fact `list_all` returned (user facts plus the
    project's); `project` is the project root, named by its last part."""
    shown, _ = select_injected_facts(facts)
    saved_project = sum(f.get("scope") == "project" for f in facts)
    name = project.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or project
    text = render_facts_step(shown, saved_user=len(facts) - saved_project, saved_project=saved_project, project=name)
    return text, facts_fingerprint(shown)


def render_facts_step(shown: list[dict], *, saved_user: int, saved_project: int, project: str) -> str:
    """The facts step's text (D2): a block per scope that has saved facts, then the replace note."""
    if not saved_user and not saved_project:
        return f"No facts are saved any more. {FACTS_REPLACE_NOTE}"
    blocks = []
    for scope, tag, saved, attrs in (("user", "user_facts", saved_user, ""),
                                     ("project", "project_facts", saved_project, f' project="{project}"')):
        if not saved:
            continue
        rows = [f for f in shown if f.get("scope", "user") == scope]
        lines = "".join(f"- {f['key']}: {f['value']}\n" for f in rows)
        blocks.append(f'<{tag}{attrs} shown="{len(rows)}" saved="{saved}">\n{lines}</{tag}>')
    text = "\n".join(blocks) + "\n" + FACTS_REPLACE_NOTE
    if len(shown) < saved_user + saved_project:
        text += "\nOther saved facts appear in the turn context when relevant."
    return text
