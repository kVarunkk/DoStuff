from dostuff.lib.exceptions import ConfirmationRequired
import os
from dostuff.helpers.tools.resolve_safe_path import resolve_safe_path

def delete_file(path: str,  _confirmed: bool = False) -> str:
    """Deletes a file inside the agent's workspace. Always requires user confirmation
    before proceeding, since deletion is irreversible.

    Args:
        path: Path relative to the project root, e.g. './blog_post.md'
            or 'skills/my-skill/SKILL.md'. Always include the top-level folder
            ('./' or 'skills/') as part of the path — do not omit it.
        _confirmed: Internal flag used by the harness when resuming after user approval.
            Do not set this manually — leave it as the default (False) when initially
            requesting a deletion; the harness will retry with this set to True only
            after the user has explicitly confirmed. 
    Returns:
        A confirmation message once the file has been deleted.

    Raises:
        ValueError: If the resolved path is outside the workspace directory.
        FileNotFoundError: If the file does not exist.
        ConfirmationRequired: Always raised on first call (no prior approval) —
            the harness should catch this, prompt the user, and retry with
            resume_args merged in to actually perform the deletion.
    """
    safe_path = resolve_safe_path(path)    

    if safe_path.is_file() and not _confirmed:
        raise ConfirmationRequired(f"Delete file '{path}'? This cannot be undone.", resume_args={"_confirmed": True})

    os.remove(safe_path)
    return f"File deleted: {path}"