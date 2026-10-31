"""Subagent registry, foreground runs, adapter, prompt, the `agent` tool and concurrency (subagents spec)."""


def test_imports():
    from dostuff.agent.subagents.registry import AgentRegistry, AgentRun, get_agent_registry
    assert isinstance(get_agent_registry(), AgentRegistry)
    assert AgentRun.__dataclass_fields__["status"].default == "queued"
