import os
import re
from pathlib import Path
from typing import Any

import yaml

from dostuff.helpers.tools.resolve_safe_path import set_skill_roots

# Search order is precedence order: the first skill found with a given name wins,
# so project skills override same-named user (home) skills.
SKILLS_DIRS = [
    ".dostuff/skills",
    ".agents/skills",
    str(Path.home() / ".dostuff" / "skills"),
    str(Path.home() / ".agents" / "skills"),
    "./skills",
]


_DISABLE_KEY = "disable-model-invocation"
_NOT_DISABLED = frozenset({"", "false", "no", "off", "0"})


def _parse_frontmatter(frontmatter: str) -> dict[str, Any]:
    """`name` and `description` as written, when they are strings, and `disable-model-invocation` whatever it
    is. BaseLoader keeps scalars as spelled (`name: 1.10`, `name: yes`). Invalid YAML falls back to one regex
    per field: a malformed third-party SKILL.md degrades, it never drops out of the index."""
    try:
        parsed = yaml.load(frontmatter, Loader=yaml.BaseLoader)
    except yaml.YAMLError:
        parsed = None
    if isinstance(parsed, dict):
        fields = {key: parsed[key] for key in ("name", "description") if isinstance(parsed.get(key), str)}
        return {**fields, _DISABLE_KEY: parsed[_DISABLE_KEY]} if _DISABLE_KEY in parsed else fields
    fields = {}
    for key in ("name", "description", _DISABLE_KEY):
        match = re.search(rf"^{key}:[ \t]*(.+)$", frontmatter, re.MULTILINE)
        if match:
            fields[key] = match.group(1)
    return fields


def _disabled(value: Any) -> bool:
    """Only a clear "no" leaves the skill visible to the model: one someone tried to protect must not lose that
    to a typo, and running it by hand works either way."""
    if value is None:
        return False
    return not (isinstance(value, str) and value.strip().lower() in _NOT_DISABLED)


def discover_skills() -> list[dict]:
    skills: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_dirs: set[str] = set()
    for SKILLS_DIR in SKILLS_DIRS:
        if not os.path.isdir(SKILLS_DIR):
            continue
        # The same directory can be reached twice (e.g. cwd is home, so
        # .dostuff/skills is ~/.dostuff/skills); scan it once.
        real_dir = os.path.normcase(os.path.realpath(SKILLS_DIR))
        if real_dir in seen_dirs:
            continue
        seen_dirs.add(real_dir)
        for entry in sorted(os.listdir(SKILLS_DIR)):
            skill_md_path = os.path.join(SKILLS_DIR, entry, "SKILL.md")
            if not os.path.isfile(skill_md_path):
                continue
            with open(skill_md_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            if not match:
                continue
            fields = _parse_frontmatter(match.group(1))
            name = fields.get("name", "").strip() or entry
            if name in seen_names:
                continue
            seen_names.add(name)
            # The entry folder, never the file's parent: a symlinked SKILL.md must not move `dir`
            skill_dir = Path(SKILLS_DIR, entry).resolve()
            skills.append({
                "name": name,
                "description": " ".join(fields.get("description", "").split()),
                "dir": str(skill_dir),
                "location": str(skill_dir / "SKILL.md"),
                "disable_model_invocation": _disabled(fields.get(_DISABLE_KEY)),
            })
    set_skill_roots(s["dir"] for s in skills)
    return skills
