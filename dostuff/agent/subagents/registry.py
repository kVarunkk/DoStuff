"""Every subagent run, foreground or background, is an asyncio task owned by the registry (spec D1, D4)."""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

from dostuff.lib.worktree import Worktree

AgentStatus = Literal["queued", "running", "waiting_approval", "completed", "failed", "stopped", "interrupted"]


@dataclass
class AgentRun:
    id: str  # "a3f2c1", what the model and the user see
    owner: str  # session id
    agent_type: str  # "explore", "code-reviewer", "fork", …
    description: str  # 3–5 words from the call, shown in the UI
    model: str  # resolved litellm model string
    status: AgentStatus = "queued"
    background: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)  # kept across runs
    compaction_notes: str = ""
    worktree: Worktree | None = None
    started: float = 0.0
    ended: float | None = None
    tokens: int = 0
    cost_usd: float = 0.0
    result: str | None = None  # the last run's final report
    inbox: list[str] = field(default_factory=list)  # send_message to a running agent (D10)
    cancel: asyncio.Event = field(default_factory=asyncio.Event)  # its own, not the main agent's (D5)
    task: asyncio.Task[Any] | None = None
    done: asyncio.Future[str] | None = None  # what a foreground tool call awaits; made inside the loop


class AgentRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}

    def runs(self, owner: str | None = None) -> list[AgentRun]:
        return [r for r in self._runs.values() if owner is None or r.owner == owner]

    def get(self, agent_id: str) -> AgentRun | None:
        return self._runs.get(agent_id)

    async def start(self, **kwargs: Any) -> AgentRun:
        raise NotImplementedError

    async def stop(self, run: AgentRun, by: str) -> None:
        raise NotImplementedError

    async def stop_all(self, by: str) -> None:
        raise NotImplementedError


_agent_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry
