from dostuff.lib.exceptions import ConfirmationRequired
from dostuff.helpers.tools.resolve_safe_path import resolve_safe_path
import difflib
from pathlib import Path

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

    # Compute diff if file existed (before overwrite/write)
    previous_content = ""
    if safe_path.is_file():
        previous_content = safe_path.read_text(encoding="utf-8")
    else:
        previous_content = ""  # new file

    safe_path.write_text(content, encoding="utf-8")

    diff_lines = []
    if previous_content != content:
        diff_lines = list(difflib.unified_diff(
            previous_content.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        ))

    result = f"File written: {path}"
    if diff_lines:
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))

        colored_lines = []
        old_line = new_line = 0
        for ln in diff_lines:                    
            s = ln.rstrip("\n")
            if s.startswith("@@"):
                parts = s.split()
                try:
                    old_line = int(parts[1].split(",")[0].lstrip("-"))
                    new_line = int(parts[2].split(",")[0].lstrip("+"))
                except Exception:
                    pass
                continue
            if s.startswith("---") or s.startswith("+++"):
                continue
            if s.startswith("\\ No newline at end of file"):
                continue
            if ln.startswith("-"):
                colored_lines.append(f"  {old_line:>3}    \033[48;5;52m{s}\033[0m")
                old_line += 1
            elif ln.startswith("+"):
                colored_lines.append(f"  {new_line:>3}    \033[48;5;22m{s}\033[0m")
                new_line += 1
            else:
                colored_lines.append(f"  {new_line:>3}    {s}")
                old_line += 1
                new_line += 1
        colored_diff = "\n".join(colored_lines)
        result += f" (Δ +{added} / -{removed} lines)\n{colored_diff}"
    return result