"""chromadb is optional (`dostuff[memory]`); AGENTS.md / CLAUDE.md files are combined; `--version`."""
import asyncio
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dostuff.helpers.agent.load_project_instructions import instruction_files, load_project_instructions


@pytest.fixture
def no_chroma(monkeypatch, tmp_path):
    """memory.backend: chroma without chromadb installed: the one setup where long-term memory is off."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    (tmp_path / "home" / ".dostuff").mkdir(parents=True)
    (tmp_path / "home" / ".dostuff" / "config.yaml").write_text("memory:\n  backend: chroma\n")
    monkeypatch.chdir(tmp_path)
    from dostuff import config as config_module
    from dostuff.lib.memory import chroma
    config_module.reset_config()
    monkeypatch.setattr(chroma, "chroma_available", lambda: False)
    yield
    config_module.reset_config()


def test_without_chromadb_memory_is_off(no_chroma):
    from dostuff.agent.instructions import build_base_instructions, build_turn_context
    from dostuff.lib.memory.stores import get_episodic_store, get_semantic_store, make_tool_registry_store
    from dostuff.tools.definitions import MEMORY_TOOLS, TOOL_SCHEMAS, default_tool_names

    assert get_semantic_store() is None and get_episodic_store() is None and make_tool_registry_store() is None
    names = default_tool_names()
    assert not MEMORY_TOOLS & set(names)
    assert set(names) | MEMORY_TOOLS == set(TOOL_SCHEMAS)

    base, _, _ = build_base_instructions()
    assert "save_memory" not in base and "## Long-term memory" not in base
    turn = asyncio.run(build_turn_context("hi", memory_store=None, episodic_store=None, user_id="user")).text
    assert "<environment>" in turn and "<other_relevant_facts>" not in turn

    from dostuff.tools.memory.save_memory import save_memory
    with pytest.raises(RuntimeError, match=r"dostuff\[memory\]"):
        asyncio.run(save_memory("name", "Ada"))


def test_without_chromadb_mcp_search_matches_keywords(no_chroma):
    from dostuff.lib.mcp.mcp_client import MCPClient
    from dostuff.tools.mcp.search_mcp_tools import _impl_search_mcp_tools

    client = MCPClient(registration_store=None, tool_registry_store=None)  # type: ignore[arg-type]
    client.servers["gh"] = object()
    client.mcp_tools["gh__create_issue"] = {"server_name": "gh", "tool_name": "create_issue",
                                            "schema": {"description": "Create a GitHub issue"}}
    assert "gh__create_issue" in asyncio.run(_impl_search_mcp_tools("open an issue", client))


def test_default_backend_without_chromadb_keeps_memory_on(no_chroma, tmp_path):
    from dostuff import config as config_module
    from dostuff.lib.memory import stores
    from dostuff.lib.memory.sqlite_memory_store import SQLiteEpisodeStore, SQLiteFactStore
    from dostuff.tools.definitions import MEMORY_TOOLS, default_tool_names
    (tmp_path / "home" / ".dostuff" / "config.yaml").write_text("")  # the default backend: hybrid
    config_module.reset_config()
    assert isinstance(stores.get_semantic_store(), SQLiteFactStore)  # hybrid without chromadb = keywords only
    assert isinstance(stores.get_episodic_store(), SQLiteEpisodeStore)
    assert MEMORY_TOOLS <= set(default_tool_names())
    from dostuff.cli import app
    result = CliRunner().invoke(app, ["doctor"])
    assert "Long-term memory: on, keyword search only" in result.output


def test_doctor_reports_memory_off(no_chroma):
    from dostuff.cli import app
    result = CliRunner().invoke(app, ["doctor"])
    assert "Long-term memory: off" in result.output and 'pip install "dostuff[memory]"' in result.output


def test_version_flag():
    from dostuff import __version__
    from dostuff.cli import app
    for flag in ("--version", "-V"):
        result = CliRunner().invoke(app, [flag])
        assert result.exit_code == 0 and result.output.strip() == f"dostuff {__version__}"


def test_instruction_files_are_combined_general_first(tmp_path):
    home = tmp_path / "home"
    (home / ".dostuff").mkdir(parents=True)
    (home / ".dostuff" / "AGENTS.md").write_text("global rule", encoding="utf-8")
    repo = tmp_path / "repo"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)
    (repo / "AGENTS.md").write_text("repo rule", encoding="utf-8")
    (repo / "CLAUDE.md").write_text("ignored: AGENTS.md wins in the same folder", encoding="utf-8")
    (pkg / "CLAUDE.md").write_text("package rule", encoding="utf-8")

    assert instruction_files(pkg, home) == [home / ".dostuff" / "AGENTS.md", repo / "AGENTS.md", pkg / "CLAUDE.md"]
    text = load_project_instructions(pkg, home)
    assert text.index("global rule") < text.index("repo rule") < text.index("package rule")
    assert f"### From {repo / 'AGENTS.md'}" in text and "ignored" not in text

    # A single file is used as-is
    assert load_project_instructions(repo / "pkg" / ".." , tmp_path / "nohome") == "repo rule"


# --- Cache-stable context (plan/2026-11-05-cache-stable-context/tasks.md) ---------------------------------------

class _Store:
    def __init__(self, rows=None, error=None):
        self.rows, self.error, self.calls = rows or [], error, []

    async def query(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.rows

    async def list_all(self, *args, **kwargs):
        self.calls.append(("list_all", kwargs))
        if self.error:
            raise self.error
        return self.rows


FACT = {"key": "name", "value": "Ada", "scope": "user", "updated_at_epoch": 0}
EPISODE = {"event_type": "EVENT", "anchor_event": "shipped", "summary": "it went out", "created_at_iso": "2026-11-05"}


@pytest.mark.parametrize("base", ["base", "base\n\n  "])
def test_system_instructions_is_base_verbatim(base):
    from dostuff.agent.instructions import build_system_instructions
    assert build_system_instructions(base) == base


def test_no_stale_importer():
    root = Path(__file__).resolve().parents[1] / "dostuff"
    assert not [p for p in root.rglob("*.py") if "build_turn_instructions" in p.read_text(encoding="utf-8")]


def test_turn_context_recalls_episodes_not_facts_below_the_cap():
    """Below the facts step's cap every fact is in the facts step already (memory recall D3): no fact query."""
    from dostuff.agent.instructions import build_turn_context
    facts, episodes = _Store([FACT]), _Store([EPISODE])
    text = asyncio.run(build_turn_context("hi", memory_store=facts, episodic_store=episodes, user_id="u")).text
    assert "facts>" not in text and "- name: Ada" not in text
    assert "  • [EVENT] shipped\n    Summary: it went out\n    Date: 2026-11-05" in text
    assert [c[0] for c in facts.calls] == ["list_all"]
    assert episodes.calls[0][1]["top_k"] == 3 and episodes.calls[0][1]["query_text"] == "hi"


def test_turn_context_recall_failure_is_reported(monkeypatch):
    from dostuff.agent import instructions
    said = []
    monkeypatch.setattr(instructions, "emit", lambda text, kind: said.append((text, kind)))
    text = asyncio.run(instructions.build_turn_context(
        "hi", memory_store=_Store(error=RuntimeError("no model")), episodic_store=_Store([EPISODE]), user_id="u")).text
    assert said == [("Memory recall failed, continuing without it: no model", "error")]
    assert "<environment>" in text and "<past_episodes>" not in text


def test_turn_context_without_memory_stores():
    from dostuff.agent.instructions import build_turn_context
    text = asyncio.run(build_turn_context("hi", memory_store=None, episodic_store=None, user_id="u",
                                          prompt_context="from a hook")).text
    assert "<environment>" in text and "from a hook" in text
    assert "<other_relevant_facts>" not in text and "<past_episodes>" not in text


def test_hook_context_rides_the_bundle():
    from dostuff.agent.instructions import build_turn_context
    text = asyncio.run(build_turn_context("hi", memory_store=None, episodic_store=None, user_id="u",
                                          prompt_context="from a hook")).text
    assert text.startswith("<turn-context>\n") and text.endswith("[Context from UserPromptSubmit hooks]\nfrom a hook\n</turn-context>")


def test_with_prompt_context_is_gone():
    with pytest.raises(ImportError):
        from dostuff.helpers.hooks.events import with_prompt_context  # noqa: F401


def _index_block(skill):
    """One skill's rows in the index: the shape lives here, in one place."""
    return f"- name: {skill['name']}\n  description: {skill['description']}\n  dir: {skill['dir']}"


def test_skill_headers_point_at_the_dir(monkeypatch):
    from dostuff.agent.subagents.definitions import BUILTIN_TYPES
    from dostuff.agent.subagents.prompt import _sections
    from dostuff.helpers.agent.constants import SYSTEM_INSTRUCTIONS
    from dostuff.helpers.skills import discover_skills as ds
    monkeypatch.setattr(ds, "SKILLS_DIRS", [])
    subagent, _ = _sections(BUILTIN_TYPES["general-purpose"], None)
    for text in (" ".join(SYSTEM_INSTRUCTIONS.split()), " ".join(subagent.split())):
        assert "read the SKILL.md in its dir" in text
        assert "the files it refers to are relative to that dir" in text
        assert "at the given path" not in text


def test_system_prompt_names_the_skill_dir_exception():
    from dostuff.helpers.agent.constants import SYSTEM_INSTRUCTIONS
    section = " ".join(SYSTEM_INSTRUCTIONS.split("## Responding")[0].split())
    assert "File tools cannot reach outside it, except that they can read (not write) `.dostuff`" in section
    assert "the skill dirs listed under Available skills" in section


def test_pin_skills_summary_shape(tmp_path, monkeypatch):
    from dostuff.agent.instructions import skills_summary
    from dostuff.helpers.skills import discover_skills as ds
    for name in ("alpha", "beta"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: the {name}\n---\nbody\n",
                                                  encoding="utf-8")
    monkeypatch.setattr(ds, "SKILLS_DIRS", [str(tmp_path)])

    skills = ds.discover_skills()

    assert [s["name"] for s in skills] == ["alpha", "beta"]
    assert skills_summary(skills) == "\n\n".join(_index_block(s) for s in skills)
    assert all(Path(s["dir"]).is_absolute() and Path(s["dir"]).is_dir() for s in skills)
    assert skills_summary([]) == "(none available)"


def _visible_and_manual(root):
    for name, extra in (("visible", ""), ("manual", "disable-model-invocation: true\n")):
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: the {name} one\n{extra}---\nbody\n",
                                         encoding="utf-8")


def test_manual_skill_not_in_prompt(tmp_path, monkeypatch):
    from dostuff.agent.instructions import build_base_instructions
    from dostuff.helpers.skills import discover_skills as ds
    _visible_and_manual(tmp_path)
    monkeypatch.setattr(ds, "SKILLS_DIRS", [str(tmp_path)])
    prompt, skills, _ = build_base_instructions()
    assert "the visible one" in prompt and "the manual one" not in prompt and str(tmp_path / "manual") not in prompt
    assert sorted(s["name"] for s in skills) == ["manual", "visible"]  # the / menu and /help are built from this


def test_all_skills_manual_shows_none_available():
    from dostuff.agent.instructions import skills_summary
    assert skills_summary([{"name": "a", "description": "d", "dir": "x", "disable_model_invocation": True}]) \
        == "(none available)"
