"""/tools: every tool the main agent has, how each one asks, whether subagents get it, and the MCP tools."""
from typing import Any

from dostuff.agent.permissions import ToolKind

# How a call of each kind is approved in `ask` mode (README "Tools and permissions")
ASKS: dict[ToolKind, str] = {
    "read": "never",
    "write": "yes",
    "shell": "unless read-only",
    "network": "yes",
    "mcp": "yes",
    "settings": "always",
}


def _summary(description: str | None) -> str:
    """The first sentence of a tool description, on one line."""
    text = " ".join((description or "").split())
    end = text.find(". ")
    return text[:end + 1] if end != -1 else text


def tools_listing(mcp_tools: dict[str, dict[str, Any]] | None = None) -> str:
    """The built-in tools (name, when it asks, subagents, what it does), then each connected MCP server's
    tools as `call_mcp_tool` names them. `mcp_tools` is MCPClient.mcp_tools."""
    from dostuff.agent.subagents import RESTRICTED_SUBAGENT_TOOLS
    from dostuff.tools.definitions import MEMORY_TOOLS, TOOL_MAP, default_tool_names

    offered = set(default_tool_names())
    names = [n for n in TOOL_MAP if n in offered]
    width = max(len(n) for n in names) + 2
    lines = [f"Tools ({len(names)})", f"  {'name'.ljust(width)}{'asks?'.ljust(18)}{'subagents'.ljust(11)}what it does"]
    for name in names:
        spec = TOOL_MAP[name]
        sub = "no" if name in RESTRICTED_SUBAGENT_TOOLS else "yes"
        lines.append(f"  {name.ljust(width)}{ASKS[spec.kind].ljust(18)}{sub.ljust(11)}"
                     f"{_summary(spec.schema['function'].get('description'))}")
    off = [n for n in TOOL_MAP if n not in offered and n in MEMORY_TOOLS]
    if off:
        lines.append(f"  Off: {', '.join(off)} (need the memory extra: pip install \"dostuff[memory]\")")

    servers: dict[str, list[dict[str, Any]]] = {}
    for info in (mcp_tools or {}).values():
        servers.setdefault(info["server_name"], []).append(info)
    total = sum(len(tools) for tools in servers.values())
    lines.append("")
    if not servers:
        lines.append("MCP tools: none (no server connected; /status shows the servers, /config mcp adds one)")
    else:
        lines.append(f"MCP tools ({total}): the agent reaches them with call_mcp_tool, which asks; subagents too")
        for server in sorted(servers):
            lines.append(f"  {server}")
            for info in sorted(servers[server], key=lambda i: i["tool_name"]):
                lines.append(f"    {info['tool_name']}  {_summary(info['schema'].get('description'))}".rstrip())
    lines += [
        "",
        "asks? is for the ask mode: /permissions changes it, and settings tools ask in every mode.",
        "ask_user is the one tool that talks to the user: subagents never get it, and it never needs approval.",
        "In an agent file, tools: takes these names or Claude Code's (Read, Bash, mcp__*...); /agents types shows each type's.",
    ]
    return "\n".join(lines)
