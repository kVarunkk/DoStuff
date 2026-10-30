"""Lifecycle hooks: user- and project-defined commands the harness runs at fixed points (plan/2026-10-29-hooks)."""
from dostuff.helpers.hooks.dispatch import HookOutcome, run_hooks
from dostuff.helpers.hooks.events import session_start, user_prompt_submit
from dostuff.helpers.hooks.load_hooks import HookEntry, load_hooks
from dostuff.helpers.hooks.run_hook import HookResult, run_hook

__all__ = [
    "load_hooks", "run_hook", "run_hooks", "HookOutcome", "HookResult", "HookEntry",
    "user_prompt_submit", "session_start",
]
