import os
import re
from pathlib import Path

# Search order is precedence order: the first skill found with a given name wins,
# so project skills override same-named user (home) skills.
SKILLS_DIRS = [
    ".dostuff/skills",
    ".agents/skills",
    str(Path.home() / ".dostuff" / "skills"),
    str(Path.home() / ".agents" / "skills"),
    "./skills",
]

def discover_skills() -> list[dict]:
    skills = []
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
            with open(skill_md_path, "r", encoding="utf-8") as f:
                content = f.read()
            match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            if not match:
                continue
            frontmatter = match.group(1)
            name_match = re.search(r"name:\s*(.+)", frontmatter)
            desc_match = re.search(r"description:\s*(.+)", frontmatter)
            name = name_match.group(1).strip() if name_match else entry
            if name in seen_names:
                continue
            seen_names.add(name)
            skills.append({
                "name": name,
                "description": desc_match.group(1).strip() if desc_match else "",
                "location": skill_md_path,
            })
    return skills
