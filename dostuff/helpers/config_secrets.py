"""Keys belong in `.dostuff/.env`: find what looks like one in a parsed config.yaml."""
import re
from typing import Any

_SECRET_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|apikey|secret|token|password|passwd|credential|bearer|authorization)s?$", re.IGNORECASE,
)
# Well-known key shapes, so a key inside a hook command or a URL is caught too
_SECRET_VALUE = re.compile(
    r"\b(?:sk-[\w-]{20,}|gh[pousr]_\w{20,}|github_pat_\w{20,}|xox[abprs]-[\w-]{10,}|AIza[\w-]{30,}|AKIA[0-9A-Z]{16}"
    r"|gsk_\w{20,}|Bearer\s+[\w.~+/-]{16,})"
)
_ENV_REFERENCE = re.compile(r"^\$\{?\w+\}?$")

SECRET_HINT = "put the value in .dostuff/.env and name the variable instead (a model's `api_key_env`)"


def find_secrets(data: Any, path: str = "") -> list[str]:
    """The dotted paths of values that look like secrets. Never the values: this text reaches the model."""
    found: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            where = f"{path}.{key}" if path else str(key)
            named = isinstance(value, str) and value.strip() and _SECRET_KEY.search(str(key))
            if named and not _ENV_REFERENCE.match(value.strip()):
                found.append(where)
            else:
                found += find_secrets(value, where)
    elif isinstance(data, list):
        for i, value in enumerate(data):
            found += find_secrets(value, f"{path}[{i}]")
    elif isinstance(data, str) and _SECRET_VALUE.search(data):
        found.append(path)
    return found
