"""Prompts typed in the TUI, saved per project directory for Up/Down recall.

One JSON line per prompt ({"cwd": ..., "text": ...}) in ~/.dostuff/data/prompt_history.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

MAX_RECALLED = 500  # prompts loaded for one directory
MAX_FILE_LINES = 5000  # past this, the file is cut back to its newest KEEP_LINES lines
KEEP_LINES = 2000


def load_history(path: Path, cwd: str) -> list[str]:
    """This directory's prompts, oldest first; [] if the file is missing or unreadable."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    if len(lines) > MAX_FILE_LINES:
        lines = lines[-KEEP_LINES:]
        try:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass
    prompts = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get("cwd") == cwd and isinstance(entry.get("text"), str):
            prompts.append(entry["text"])
    return prompts[-MAX_RECALLED:]


def append_history(path: Path, cwd: str, text: str) -> None:
    """Save one prompt; errors are ignored (history is a convenience)."""
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"cwd": cwd, "text": text}) + "\n")
    except OSError:
        pass
