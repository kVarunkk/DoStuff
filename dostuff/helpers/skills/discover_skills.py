import os
import re
import stat
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


def model_visible_skills(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The skills the model may be told about: every list it sees goes through here, so a manual-only skill
    (`disable-model-invocation`) is left out in this one place (skills-and-multi-edit CON-005)."""
    return [s for s in skills if not s.get("disable_model_invocation")]


def _skill_folders() -> list[str]:
    """SKILLS_DIRS that exist, each once: the same directory can be reached twice (e.g. cwd is home, so
    .dostuff/skills is ~/.dostuff/skills)."""
    folders: list[str] = []
    seen_dirs: set[str] = set()
    for skills_dir in SKILLS_DIRS:
        if not os.path.isdir(skills_dir):
            continue
        real_dir = os.path.normcase(os.path.realpath(skills_dir))
        if real_dir not in seen_dirs:
            seen_dirs.add(real_dir)
            folders.append(skills_dir)
    return folders


def skills_signature() -> tuple[tuple[str, int, int], ...]:
    """(SKILL.md path, mtime_ns, size) for every candidate skill, in a stable order: a stat pass that opens and parses
    nothing (a few ms per 100 skills, one stat each), so it can run before every model call. It only says *whether*
    discovery would find something different. Paths are absolute, so a `cd` moves it too."""
    sig: list[tuple[str, int, int]] = []
    for skills_dir in _skill_folders():
        try:
            entries = sorted(os.listdir(skills_dir))
        except OSError:
            continue
        base = os.path.abspath(skills_dir)
        for entry in entries:
            path = os.path.join(base, entry, "SKILL.md")
            try:
                st = os.stat(path)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                sig.append((path, st.st_mtime_ns, st.st_size))
    return tuple(sig)


def discover_skills() -> list[dict]:
    skills: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for SKILLS_DIR in _skill_folders():
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
