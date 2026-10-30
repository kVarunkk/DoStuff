"""Run every hook matching an event and merge their results into one HookOutcome (spec REQ-1.2, D2, D7)."""
from dataclasses import dataclass, field
from typing import Any

from dostuff.helpers.agent.loop_emit import loop_emit
from dostuff.helpers.hooks import trust
from dostuff.helpers.hooks.load_hooks import TOOL_EVENTS, HookEntry
from dostuff.helpers.hooks.run_hook import FAILED_REASON, Decision, HookResult, run_hook

SHELL_TOOLS = ("bash_command",)  # the tools an `if` pattern is searched in (tool_input.command)
_RANK: dict[Decision, int] = {"none": 0, "allow": 1, "ask": 2, "deny": 3}
UNTRUSTED_WARNING = "project hooks not trusted; skipped ({count} from {origin})"

_untrusted_warned: set[str] = set()  # session ids already told their project hooks were skipped


def untrusted_notice_given(session_id: str) -> None:
    """The caller already said project hooks are skipped (headless does, up front)."""
    _untrusted_warned.add(session_id)


@dataclass
class HookOutcome:
    """What callers see: the merged decision of every hook that ran for one event."""
    decision: Decision = "none"
    reason: str = ""
    decided_by: str = ""  # name of the hook whose decision won
    updated_input: dict[str, Any] | None = None
    contexts: list[tuple[str, str]] = field(default_factory=list)  # (hook name, text), config order
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False

    @property
    def context(self) -> str:
        return "\n".join(text for _, text in self.contexts)


def hook_line(data: dict[str, Any]) -> str:
    """How a `hook` event reads in the TUI and on headless stderr: `⚓ PreToolUse bash_command → deny (reason)`."""
    prefix = f"[subagent: {data['subagent']}] " if data.get("subagent") else ""
    target = f" {data['tool']}" if data.get("tool") else ""
    reason = f" ({data['reason']})" if data.get("reason") else ""
    return f"{prefix}⚓ {data.get('event', '')}{target} → {data.get('decision', '')}{reason}"


async def report_outcome(
    event: str, outcome: "HookOutcome", *, adapter: Any, target: str = "", show_block: bool = True,
) -> None:
    """Show what the hooks did: one `hook` line for a decision, rewrite or context, one `error` line per failure.

    `target` is the tool name (tool events) or empty; a deny on a non-tool event reads as "block".
    `show_block=False` when the caller prints its own message for a block (a rejected prompt).
    """
    for error in outcome.errors:
        await loop_emit("error", error, adapter=adapter)
    decision: str
    if outcome.decision != "none":
        if outcome.decision == "deny" and not show_block:
            return
        decision, reason = outcome.decision, outcome.reason
        if decision == "deny" and event not in TOOL_EVENTS:
            decision = "block"
    elif outcome.updated_input is not None:
        decision, reason = "rewrite", ""
    elif outcome.contexts:
        decision, reason = "context", ", ".join(name for name, _ in outcome.contexts)
    else:
        return
    await loop_emit("hook", {"event": event, "tool": target, "decision": decision, "reason": reason}, adapter=adapter)


def _matches(entry: HookEntry, payload: dict[str, Any]) -> bool:
    event = entry.event
    if event in TOOL_EVENTS:
        target = payload.get("tool_name")
    elif event == "SessionStart":
        target = payload.get("source")
    else:
        return True  # UserPromptSubmit / Stop / SubagentStop have nothing to match against
    if entry.matcher_re is not None and not entry.matcher_re.fullmatch(str(target)):
        return False
    if entry.if_re is not None and payload.get("tool_name") in SHELL_TOOLS:
        tool_input = payload.get("tool_input")
        command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
        return entry.if_re.search(str(command)) is not None
    return True


def _merge(outcome: HookOutcome, entry: HookEntry, result: HookResult) -> None:
    """Fold one hook's result in: strictest decision wins, contexts concatenate, last rewrite wins."""
    decision, reason = result.decision, result.reason
    if result.error:
        outcome.errors.append(result.error)
        if entry.fail_closed:
            decision, reason = "deny", FAILED_REASON.format(name=entry.name)
    if result.warning:
        outcome.errors.append(result.warning)
    if _RANK[decision] > _RANK[outcome.decision]:
        outcome.decision, outcome.reason, outcome.decided_by = decision, reason, entry.name
    if result.context:
        outcome.contexts.append((entry.name, result.context))
    if result.updated_input is not None:
        outcome.updated_input = result.updated_input


async def run_hooks(event: str, payload: dict[str, Any], *, adapter: Any = None) -> HookOutcome:
    """Run the hooks configured for `event`, sequentially in config order, each on the same payload."""
    # Read per dispatch, not cached here, so Config.reload() (edit_config) applies without a restart (REQ-1.5).
    # Lazy import: config.py imports this package to load hooks.
    from dostuff.config import get_config
    config = get_config()
    entries = config.hooks.get(event)
    if not entries:
        return HookOutcome()  # the cost of having no hooks: one attribute read and one dict lookup (REQ-1.4)

    runnable = [e for e in entries if _matches(e, payload)]
    skipped: list[HookEntry] = []
    if any(e.source == "project" for e in runnable) and not await trust.project_allowed(
            config.hooks_project_trust, config.hooks, adapter):
        skipped = [e for e in runnable if e.source == "project"]
        runnable = [e for e in runnable if e.source != "project"]
    session_id = str(payload.get("session_id", ""))
    # project_trust: deny is the user's standing choice, not news worth a line every session
    if skipped and config.hooks_project_trust != "deny" and session_id not in _untrusted_warned:
        _untrusted_warned.add(session_id)
        await loop_emit("system", UNTRUSTED_WARNING.format(count=len(skipped), origin=skipped[0].origin), adapter)

    outcome = HookOutcome()
    for entry in runnable:
        result = await run_hook(entry, payload, adapter=adapter, max_output_chars=config.max_tool_output_chars)
        if result.cancelled:
            # Not promoted to deny even when fail_closed: the caller handles cancellation (D7)
            outcome.cancelled = True
            break
        _merge(outcome, entry, result)
    if outcome.updated_input is not None and outcome.decision == "allow":
        # An allow judged the input the hook saw; the rewritten args face the permission layer (D2 step 5)
        outcome.decision, outcome.reason, outcome.decided_by = "none", "", ""
    return outcome
