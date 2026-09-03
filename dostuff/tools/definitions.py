from dostuff.tools.bash.bash_command import bash_command
from dostuff.tools.files.read_file import read_file
from dostuff.tools.files.write_file import write_file
from dostuff.tools.files.delete_file import delete_file
from dostuff.tools.files.list_files import list_files
from dostuff.tools.get_current_datetime import get_current_datetime
from dostuff.tools.mcp.call_mcp_tool import call_mcp_tool
from dostuff.tools.mcp.get_mcp_tool_details import get_mcp_tool_details
from dostuff.tools.mcp.search_mcp_tools import search_mcp_tools
from dostuff.tools.delegate_to_subagent import delegate_to_subagent
from dostuff.helpers.tools.generate_tool_schema import generate_litellm_tool_schema
from typing import Any

write_file_schema = generate_litellm_tool_schema(write_file)
read_file_schema = generate_litellm_tool_schema(read_file)
delete_file_schema = generate_litellm_tool_schema(delete_file)
list_files_schema = generate_litellm_tool_schema(list_files)
bash_command_schema = generate_litellm_tool_schema(bash_command)
get_current_datetime_schema = generate_litellm_tool_schema(get_current_datetime)

search_mcp_tools_schema = generate_litellm_tool_schema(search_mcp_tools)
get_mcp_tool_details_schema = generate_litellm_tool_schema(get_mcp_tool_details)
call_mcp_tool_schema = generate_litellm_tool_schema(call_mcp_tool)


delegate_to_subagent_schema = generate_litellm_tool_schema(delegate_to_subagent)

tool_schemas = [write_file_schema, read_file_schema, delete_file_schema, list_files_schema, get_current_datetime_schema, search_mcp_tools_schema, get_mcp_tool_details_schema, call_mcp_tool_schema, delegate_to_subagent_schema, bash_command_schema]

TOOL_MAP = { "read_file": read_file,"write_file": write_file, "delete_file": delete_file, "list_files": list_files, "get_current_datetime": get_current_datetime, "search_mcp_tools": search_mcp_tools, "get_mcp_tool_details": get_mcp_tool_details, "call_mcp_tool": call_mcp_tool, "delegate_to_subagent": delegate_to_subagent, "bash_command": bash_command}

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    str(schema["function"]["name"]): schema 
    for schema in tool_schemas
}
