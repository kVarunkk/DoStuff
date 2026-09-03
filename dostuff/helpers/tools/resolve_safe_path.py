from pathlib import Path
from dostuff.helpers.agent.constants import PROJECT_ROOT

def resolve_safe_path(path: str) -> Path:
    base_root = Path(PROJECT_ROOT).resolve()
    clean_path_str = path.replace("\\", "/").strip().lstrip("/")

    target_path = (base_root / clean_path_str).resolve()

    try:
        target_path.relative_to(base_root)
    except ValueError:
        raise ValueError(f"Access denied: Path '{path}' resolves outside the allowed project root.")

    return target_path