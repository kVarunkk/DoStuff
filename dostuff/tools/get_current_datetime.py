import datetime


def get_current_datetime() -> str:
    """Returns the current date and time.

    Use this whenever a task depends on knowing today's date or the current time —
    for example, resolving relative dates like 'tomorrow' or 'next Friday' before
    calling another tool that requires an exact date.

    Args:
        None.

    Returns:
        The current date and time as an ISO 8601 string, e.g. '2026-07-30T14:32:10'.
    """
    return datetime.datetime.now().isoformat(timespec="seconds")