"""Shared fixtures."""
import itertools
import sys
import textwrap
from pathlib import Path

import pytest

_hook_ids = itertools.count()


@pytest.fixture
def hook_cmd(tmp_path):
    """`hook_cmd(body)`: write a hook script and return a shell command running it with this interpreter.

    Same as `_script` in tests/test_hooks.py, for the TUI and headless tests that configure hooks.
    """
    from dostuff.tools.bash import bash_command

    def make(body: str) -> str:
        path = tmp_path / f"hook_{next(_hook_ids)}.py"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        cmd = f'"{Path(sys.executable).as_posix()}" "{path.as_posix()}"'
        # PowerShell treats a leading quoted string as a value, not a command
        return f"& {cmd}" if bash_command.select_shell()[0] == "PowerShell" else cmd
    return make
