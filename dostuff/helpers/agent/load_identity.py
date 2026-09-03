import os
from pathlib import Path

def load_identity(path: str = "DOSTUFF.md") -> str:
    # Local override first, then global, then default
    local_path = Path(".dostuff") / "DOSTUFF.md"
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            return f.read()
    global_path = Path.home() / ".dostuff" / "DOSTUFF.md"
    if global_path.exists():
        with open(global_path, "r", encoding="utf-8") as f:
            return f.read()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "You are a general-purpose personal assistant agent built by Varun. Your name is 'DoStuff'."
