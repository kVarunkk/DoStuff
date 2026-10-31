from dostuff.tools.bash.jobs import NOTICE_TAIL_LINES, JobError, _tail, format_status, get_registry, indent_block


async def kill_shell(background_id: str, owner: str = "") -> str:
    """Stop a background job: kill its process and everything it started.

    Killing a job that has already ended just reports how it ended.

    Args:
        background_id: The job's id, as returned by bash_command(run_in_background=true).

    Returns:
        The job's final status and the last lines of its stdout and stderr.
    """
    from dostuff.lib.model import current_session_id

    registry = get_registry()
    try:
        job = registry.get(owner or current_session_id.get(), background_id)
    except JobError as e:
        return f"error: {e}"
    await registry.kill(job, by="model")
    n = NOTICE_TAIL_LINES
    return (
        f'job "{job.id}": {format_status(job)}\n'
        f"stdout (last {n} lines):\n{indent_block(_tail(job.out_path, n))}\n"
        f"stderr (last {n} lines):\n{indent_block(_tail(job.err_path, n))}"
    )
