from dostuff.lib.exceptions import ConfirmationRequired
from dostuff.helpers.tools.resolve_safe_path import resolve_safe_path

def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Writes content to a file inside the project directory. Creates the file
    if it does not exist. Use this when the user asks to write to a file or
    create one — including creating or updating a skill's SKILL.md or bundled
    scripts as part of the learning loop.

    Args:
        path: Path relative to the project root, e.g. './blog_post.md'
            or 'skills/my-skill/SKILL.md'. Always include the top-level folder
            ('./' or 'skills/') as part of the path — do not omit it.
        content: The text content to write to the file.
        overwrite: Internal flag used by the harness when resuming after user
            approval. Do not set this manually — leave it as the default (False);
            the harness will retry with this set to True only after the user has
            explicitly confirmed.
    """
    safe_path = resolve_safe_path(path)

    if safe_path.is_file() and not overwrite:
        raise ConfirmationRequired(
            f"File '{path}' already exists. Overwrite it?",
            resume_args={"overwrite": True},
        )

    safe_path.parent.mkdir(parents=True, exist_ok=True)
    safe_path.write_text(content, encoding="utf-8")

    return f"File written successfully: {path}"