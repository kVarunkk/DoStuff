"""The directory file, search and shell tools work in: the cwd, unless a subagent runs in its own worktree.

The cwd is process-wide, so a worktree subagent can't chdir; it sets this context variable for its
loop instead. Context variables follow `await`, new tasks and `asyncio.to_thread`.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

_override: ContextVar[Path | None] = ContextVar("dostuff_project_root", default=None)


def project_root() -> Path:
    return _override.get() or Path.cwd()


@contextmanager
def use_project_root(path: Path) -> Iterator[None]:
    token = _override.set(path)
    try:
        yield
    finally:
        _override.reset(token)
