"""Agent types: built-in ones, and (from T3.1) Markdown files with YAML frontmatter (spec D2, D3)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentDef:
    name: str
    description: str
    prompt: str = ""  # the type's system prompt (a file's body)
    tools: tuple[str, ...] | None = None  # None: every tool a subagent of this type may have
    model: str = "inherit"
    max_iterations: int | None = None  # None: the configured subagent cap
    source: str = "built-in"  # a file path, or "built-in"


BUILTIN_TYPES: dict[str, AgentDef] = {
    "general-purpose": AgentDef(
        name="general-purpose",
        description="Research and multi-step tasks; reports what it did and found.",
        prompt=(
            "Research and multi-step tasks: investigate, make the changes the task asks for, and check your "
            "work. Report what you did and what you found."
        ),
    ),
}


class UnknownAgentType(ValueError):
    def __init__(self, name: str, available: list[str]):
        super().__init__(f"Unknown subagent_type '{name}'. Available types: {', '.join(available)}.")
        self.available = available


def get_definitions() -> dict[str, AgentDef]:
    """Every agent type this session can start, by name."""
    return dict(BUILTIN_TYPES)
