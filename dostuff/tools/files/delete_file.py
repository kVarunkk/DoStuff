import os
from dostuff.helpers.tools.resolve_safe_path import resolve_safe_path

def delete_file(path: str) -> str:
    """Deletes a file inside the project directory. Deletion is irreversible and
    requires user approval.

    Args:
        path: Path relative to the project root, e.g. './blog_post.md'
            or 'skills/my-skill/SKILL.md'.

    Returns:
        A confirmation message once the file has been deleted.

    Raises:
        PathOutsideProject: If the resolved path is outside the project root.
        FileNotFoundError: If the file does not exist.
    """
    safe_path = resolve_safe_path(path)
    if not safe_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    os.remove(safe_path)
    return f"File deleted: {path}"
