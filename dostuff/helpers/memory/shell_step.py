"""User steps that record a `!command` typed in the TUI, and their parsing (TUI, session list, export)."""


def format_shell_step(command: str, output: str) -> str:
    """User step recording a `!command` and its output, so the model sees it next turn."""
    return f"<bash-input>{command}</bash-input>\n<bash-output>\n{output}\n</bash-output>"


def parse_shell_step(content: str) -> tuple[str, str] | None:
    """Inverse of format_shell_step: (command, output), or None for a normal message."""
    if not content.startswith("<bash-input>") or "</bash-input>" not in content:
        return None
    command, _, rest = content[len("<bash-input>"):].partition("</bash-input>")
    output = rest.strip().removeprefix("<bash-output>").removesuffix("</bash-output>").strip("\n")
    return command, output
