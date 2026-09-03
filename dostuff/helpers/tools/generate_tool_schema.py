import inspect
from typing import Callable, Any
from pydantic import TypeAdapter

def generate_litellm_tool_schema(func: Callable[..., Any]) -> dict:
    """Automating conversion of a Python function into a LiteLLM tool schema."""
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""
    
    # Extract the main function description from the docstring text
    description = docstring.split("Args:")[0].split("Returns:")[0].strip()
    
    properties = {}
    required = []
    
    for name, param in signature.parameters.items():
        # Skip self or cls references if part of a class structure
        if name in ("self", "cls"):
            continue
            
        # Infer the JSON schema type safely using Pydantic's TypeAdapter matrix
        try:
            type_schema = TypeAdapter(param.annotation).core_schema
            # Map standard Python typing categories to standard JSON fields
            json_type = "string" if param.annotation == str else "number"
            if param.annotation == int: json_type = "integer"
            elif param.annotation == bool: json_type = "boolean"
            elif param.annotation == list or getattr(param.annotation, "__origin__", None) == list: json_type = "array"
        except Exception:
            json_type = "string" # Smart fallback text baseline

        # Parse parameter-specific description strings straight from docstring args layout
        param_desc = ""
        if f"{name}:" in docstring:
            try:
                # Basic string slicing to isolate lines corresponding to parameter definitions
                param_desc = docstring.split(f"{name}:")[1].split("\n")[0].strip()
            except Exception:
                pass

        properties[name] = {
            "type": json_type,
            "description": param_desc or f"The {name} parameter."
        }
        
        # Track whether parameter lacks a default assignment value
        if param.default == inspect.Parameter.empty:
            required.append(name)

    # Return the clean nested payload layout that LiteLLM accepts
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }
