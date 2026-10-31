from dostuff.config import get_config
from dostuff.tools.bash.jobs import JobError, clamp_wait_timeout, format_status, get_registry, indent_block

_WAIT_OUTCOMES = {"matched": "matched", "exited": "the job ended before a match"}


async def bash_output(background_id: str, filter: str = "", wait_for: str = "", timeout: int = 30, owner: str = "") -> str:
    """Read a background job's new output (since the previous bash_output call) and its status.

    wait_for blocks until the job prints a line you expect, e.g. a server's "ready" line.
    Cancelling the wait does not stop the job.

    Args:
        background_id: The job's id, as returned by bash_command(run_in_background=true).
        filter: Optional regex; only matching lines are returned (the others are still consumed).
        wait_for: Optional regex; wait until a new line matches it, the job ends, or timeout passes.
        timeout: Seconds to wait for wait_for (default 30, 1-110).

    Returns:
        The job's status, then its new stdout and stderr lines.
    """
    from dostuff.lib.model import current_session_id

    registry = get_registry()
    cap = get_config().max_tool_output_chars
    try:
        job = registry.get(owner or current_session_id.get(), background_id)
        waited = ""
        if wait_for:
            result = await registry.wait_for(job, wait_for, timeout, filter, cap)
            outcome = _WAIT_OUTCOMES.get(result.reason, f"timed out after {clamp_wait_timeout(timeout):.0f}s")
            waited = f"wait_for: {outcome}\n"
            output = result.output
        else:
            output = registry.read_new(job, filter, cap)
    except JobError as e:
        return f"error: {e}"
    return (
        f'job "{job.id}": {format_status(job)}\n{waited}'
        f"stdout (new):\n{indent_block(output.stdout.splitlines())}\n"
        f"stderr (new):\n{indent_block(output.stderr.splitlines())}"
    )
