"""Fork mode: a subagent that inherits the main conversation (subagents spec D11)."""
from dataclasses import dataclass, field
from typing import Any

from dostuff.agent.subagents import RESTRICTED_SUBAGENT_TOOLS

FORK_TYPE = "fork"
FORK_DESCRIPTION = ("A copy of you that inherits this whole conversation and your tools; the prompt is just the "
                    "task. Costly: use it only when the task needs this conversation.")
FORK_DIRECTIVE = (
    "You are a fork of the main agent: everything above is the main agent's conversation, copied when it started "
    "you. You are not the main agent: if the conversation asks for a fork or an agent, that request was the main "
    "agent's, and you are the result. Work on the task below with your tools, and don't continue anything else from "
    "the conversation. You can see the main agent's whole tool list, but these stay the main agent's and return an "
    "error if you call them: {restricted}.\n\n"
    "Your task: {prompt}\n\n"
    "Report back as your final answer; don't address the user."
)
MODEL_IGNORED = "[warning] model '{model}' was ignored: a fork always runs on the main model ({main})."
NO_CONVERSATION = ("[error] A fork copies the main agent's conversation, and there is none here. Use a typed agent "
                   "with a self-contained prompt.")


@dataclass
class ForkContext:
    """What a fork inherits from the main agent's loop at the moment of the call (D11). `loop()` builds it and
    `dispatch_tool` injects it into `agent` and `send_message` as the `fork_context` harness param."""
    system_instruction: str  # the main turn's dynamic system instruction (before the notes)
    compaction_notes: str
    history: list[dict[str, Any]]  # a copy of the main working history, without the in-flight assistant step
    tool_names: list[str]  # the main agent's tools
    session_hook_context: dict[str, str] | None = field(default=None)  # SessionStart context, as the main sees it
    # The capability set the copied system prompt lists (mid-run refresh D10): the fork announces against the main's
    # baseline and history, so it never re-announces what the main agent was already told. None: never announces.
    capabilities_baseline: tuple[str, ...] | None = field(default=None)

    def fork_tools(self) -> list[str]:
        """The tools a fork may run: the main agent's minus D12. It is still sent all of `tool_names`, since the
        tool schemas are the head of the cached prefix; the loop refuses a call to the rest."""
        return [name for name in self.tool_names if name not in RESTRICTED_SUBAGENT_TOOLS]


def fork_directive(prompt: str) -> str:
    """The directive that opens a fork's first step, naming the tools it sees but may not run (D12)."""
    return FORK_DIRECTIVE.format(prompt=prompt, restricted=", ".join(sorted(RESTRICTED_SUBAGENT_TOOLS)))


def fork_first_step(prompt: str, worktree_text: str = "") -> dict[str, Any]:
    """The user step after the copied history. A worktree note goes here, not in the system prompt, so the
    system prompt stays byte-identical to the main agent's (prompt cache)."""
    content = fork_directive(prompt)
    if worktree_text:
        content = f"{content}\n\n{worktree_text}"
    return {"role": "user", "content": content}
