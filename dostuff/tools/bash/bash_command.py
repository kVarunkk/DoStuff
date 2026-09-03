import subprocess
from pathlib import Path
from dostuff.lib.exceptions import ConfirmationRequired

DANGEROUS_PATTERNS = [
    "rm -rf", "rm -r -f", "del /f", "del /s /f",
    "format ", "format.c", ">", ">>"
    "> /dev", "> /etc", "> /usr", "> /sys",
    "pip uninstall", "pip remove", "pip install --force",
    "git reset --hard", "git push --force", "git push -f",
    "mv ", "shred ",
]

def _is_dangerous(cmd: str) -> bool:
    lowered = cmd.lower()
    # Check destructive patterns
    for pat in DANGEROUS_PATTERNS:
        if pat.lower() in lowered:
            return True
    # Redirection to system paths
    if ">" in cmd and ("/etc" in cmd or "/usr" in cmd or "/sys" in cmd or "/dev" in cmd):
        return True
    return False

def bash_command(command: str, cwd: str = "", _confirmed: bool = False) -> str:
    """Run a shell command and return its output as a single string.

    Args:
        command: Shell command to run, e.g. 'ls -la'.
        cwd: Optional working directory (defaults to current).
        _confirmed: Internal flag used when resuming after user approval.
            Leave as False initially; harness retries with True after HITL.

    Returns:
        Combined stdout, stderr, and exit code as a formatted string.

    Raises:
        ConfirmationRequired: If command looks destructive (rm -rf, format,
            forced git push, etc.) and has not been confirmed yet.
    """
    if _is_dangerous(command) and not _confirmed:
        raise ConfirmationRequired(
            f"Dangerous command detected: '{command}'. Allow execution?",
            resume_args={"_confirmed": True},
        )

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=cwd or str(Path.cwd()),
        )
        return (
            f"$ {command}\n"
            f"exit: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    except Exception as e:
        return f"$ {command}\nerror: {e}"
