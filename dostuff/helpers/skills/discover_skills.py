import os
import re
from pathlib import Path

SKILLS_DIRS = [
    ".dostuff/skills",
    ".agents/skills",
    str(Path.home() / ".dostuff" / "skills"),
    str(Path.home() / ".agents" / "skills"),
    "./skills",
]

def discover_skills() -> list[dict]:
    skills = []
    for SKILLS_DIR in SKILLS_DIRS:
        if not os.path.isdir(SKILLS_DIR):
            continue
        for entry in os.listdir(SKILLS_DIR):
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
            skills.append({
                "name": name_match.group(1).strip() if name_match else entry,
                "description": desc_match.group(1).strip() if desc_match else "",
                "location": skill_md_path,
            })
    return skills