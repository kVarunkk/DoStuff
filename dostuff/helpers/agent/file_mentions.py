"""`@path` in a prompt attaches that file (or a folder listing) to the message the model gets.

Mentions that aren't an existing path inside the project (e.g. `@someone`) are left as plain text.
A path with spaces is written `@"my file.txt"`.
"""
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from dostuff.helpers.agent.tool_output import truncate_tool_output
from dostuff.helpers.tools.resolve_safe_path import is_denied_for_read, resolve_safe_path
from dostuff.lib.exceptions import PathAccessDenied, PathOutsideProject
from dostuff.tools.files.read_file import read_file
from dostuff.tools.search.walk import display_path, is_excluded_dir

MENTION_RE = re.compile(r'(?<!\S)@("[^"\n]+"|[^\s"]+)')
# The mention being typed at the cursor (for completion)
PARTIAL_MENTION_RE = re.compile(r'(?:^|\s)(@"?[^\s"]*)$')
ATTACHMENT_RE = re.compile(r'\n*<attached_(file|dir) path="([^"]*)">.*?</attached_\1>', re.S)
TRAILING_PUNCTUATION = ".,;:!?)]}'"
MAX_DIR_ENTRIES = 200
MAX_INDEXED_PATHS = 20000
INDEX_TTL_S = 30.0


@dataclass
class Expansion:
    content: str  # prompt + attached blocks: what the model gets
    attached: list[str] = field(default_factory=list)  # display paths ("dir/" for folders)
    skipped: list[str] = field(default_factory=list)  # "path: reason", for paths that exist but can't be attached


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return b"\0" in f.read(8192)
    except OSError:
        return False


def _dir_listing(path: Path) -> str:
    entries = []
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.is_dir():
            if not is_excluded_dir(child.name):
                entries.append(child.name + "/")
        elif not is_denied_for_read(child):
            entries.append(child.name)
    extra = len(entries) - MAX_DIR_ENTRIES
    lines = entries[:MAX_DIR_ENTRIES] + ([f"[… {extra} more entries]"] if extra > 0 else [])
    return "\n".join(lines) or "(empty folder)"


def _resolve_mention(raw: str, root: Path) -> Path | None:
    """The existing path inside the project that `raw` names; trailing punctuation is tried
    without it ("see @a.py." → a.py). None if it isn't a path here."""
    if raw.startswith('"'):
        candidates = [raw.strip('"')]
    else:
        candidates = [raw]
        stripped = raw.rstrip(TRAILING_PUNCTUATION)
        if stripped and stripped != raw:
            candidates.append(stripped)
    for text in candidates:
        try:
            path = resolve_safe_path(text, root)
            if path.exists():
                return path
        except PathOutsideProject:
            return None
        except (OSError, ValueError):  # not a valid path on this OS
            continue
    return None


def expand_file_mentions(text: str, max_chars: int = 0, root: Path | None = None) -> Expansion:
    """Append an <attached_file> / <attached_dir> block for each `@path` in `text`.

    Files come numbered like read_file output and capped at `max_chars` each (0 = no cap).
    """
    root = (root or Path.cwd()).resolve()
    result = Expansion(content=text)
    blocks: list[str] = []
    seen: set[Path] = set()
    for match in MENTION_RE.finditer(text):
        path = _resolve_mention(match.group(1), root)
        if path is None or path in seen:
            continue
        seen.add(path)
        shown = display_path(path, root)
        if path.is_dir():
            blocks.append(f'<attached_dir path="{shown}/">\n{_dir_listing(path)}\n</attached_dir>')
            result.attached.append(shown.rstrip("/") + "/")
            continue
        if _is_binary(path):
            result.skipped.append(f"{shown}: binary file")
            continue
        try:
            body = read_file(str(path))
        except PathAccessDenied:
            result.skipped.append(f"{shown}: may contain secrets")
            continue
        except (OSError, ValueError) as e:
            result.skipped.append(f"{shown}: {e}")
            continue
        blocks.append(f'<attached_file path="{shown}">\n{truncate_tool_output(body, max_chars)}\n</attached_file>')
        result.attached.append(shown)
    if blocks:
        result.content = text + "\n\n" + "\n\n".join(blocks)
    return result


def strip_attachments(content: str) -> str:
    """The prompt as the user typed it (attached blocks removed), for display and export."""
    return ATTACHMENT_RE.sub("", content)


def attached_paths(content: str) -> list[str]:
    return [m.group(2) for m in ATTACHMENT_RE.finditer(content)]


# ── Completion ────────────────────────────────────────────────────────────────

_index: dict[str, tuple[float, list[str]]] = {}


def _project_paths(root: Path) -> list[str]:
    """Files and folders under root (posix, folders end in "/"); cached for INDEX_TTL_S."""
    key = str(root)
    cached = _index.get(key)
    if cached and time.monotonic() - cached[0] < INDEX_TTL_S:
        return cached[1]
    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not is_excluded_dir(d))
        base = Path(dirpath)
        for d in dirnames:
            paths.append(display_path(base / d, root) + "/")
        for name in sorted(filenames):
            path = base / name
            if not is_denied_for_read(path):
                paths.append(display_path(path, root))
        if len(paths) >= MAX_INDEXED_PATHS:
            break
    _index[key] = (time.monotonic(), paths)
    return paths


def mention_candidates(partial: str, root: Path | None = None, limit: int = 10) -> list[str]:
    """Project paths for a partly typed mention (without the "@"), best first: paths that start
    with it, then names that start with it, then paths that contain it."""
    root = (root or Path.cwd()).resolve()
    query = partial.strip('"').replace("\\", "/").lower()
    paths = _project_paths(root)
    if not query:
        # Top level first
        return sorted((p for p in paths if p.rstrip("/").count("/") == 0), key=lambda p: (not p.endswith("/"), p.lower()))[:limit]
    ranked: list[tuple[int, int, str]] = []
    for path in paths:
        lower = path.lower()
        name = lower.rstrip("/").rsplit("/", 1)[-1]
        if lower.startswith(query):
            rank = 0
        elif name.startswith(query):
            rank = 1
        elif query in lower:
            rank = 2
        else:
            continue
        ranked.append((rank, len(path), path))
    ranked.sort()
    return [path for _, _, path in ranked[:limit]]


def format_mention(path: str) -> str:
    return f'@"{path}"' if " " in path else f"@{path}"
