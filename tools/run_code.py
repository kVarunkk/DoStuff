import json
import subprocess
import sys
from typing import List, Optional
from helpers.agent.constants import PROJECT_ROOT
from pathlib import Path
from helpers.tools.resolve_safe_path import resolve_safe_path

IS_WINDOWS = sys.platform.startswith("win")
PYTHON_BINARY = "py" if IS_WINDOWS else "python3"

RUNTIMES = {
    ".py": PYTHON_BINARY,
    ".js": "node",
    ".ts": "npx tsx",
}

LANG_MAP = {
    "python": PYTHON_BINARY,
    "js": "node", "javascript": "node",
    "ts": "npx tsx", "typescript": "npx tsx",
}

def run_code(
    script_path: str,
    args: Optional[List[str]] = None,
    language: Optional[str] = None,
    timeout: int = 30,
) -> str:
    """Executes a script (Python, JavaScript, or TypeScript) and returns its output.
    No sandboxing is applied — the script runs with the same permissions as this
    process. Only use with trusted, reviewed scripts.

    Args:
        script_path: Path relative to the project root, e.g.
            'skills/../..' or 'agent_workspace/../..'.
        args: A list of command-line arguments to pass to the script. Any argument that is
            itself a file path must also be given relative to the project root
            (e.g. ['agent_workspace/data.json']) — it is not resolved further.
        language: Explicit language override ('python', 'javascript', 'typescript').
            Auto-detected from the file extension if omitted.
        timeout: Maximum execution time in seconds before the process is killed.
    """
    args = args or []

    safe_script_path = resolve_safe_path(script_path) 
    if not safe_script_path.exists():
        return json.dumps({"success": False, "error": f"Script not found: {script_path}"})

    ext = safe_script_path.suffix.lower()
    runtime = LANG_MAP.get(language.lower()) if language else RUNTIMES.get(ext)
    if not runtime:
        return json.dumps({"success": False, "error": f"Unsupported file type '{ext}'."})

    command = runtime.split() + [str(safe_script_path)] + [str(a) for a in args]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(PROJECT_ROOT).resolve()),
        )
        return json.dumps({
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout.strip()[:5000],   
            "stderr": result.stderr.strip()[:2000],
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": f"Timed out after {timeout}s."})
    except FileNotFoundError:
        return json.dumps({"success": False, "error": f"Runtime '{runtime}' not found on host."})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Execution error: {e}"})