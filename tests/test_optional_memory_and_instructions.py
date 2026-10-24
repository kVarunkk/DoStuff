"""chromadb is optional (`dostuff[memory]`); AGENTS.md / CLAUDE.md files are combined; `--version`."""
import asyncio
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dostuff.helpers.agent.load_project_instructions import instruction_files, load_project_instructions


@pytest.fixture
def no_chroma(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    from dostuff import config as config_module
    from dostuff.lib.memory import chroma
    config_module.reset_config()
    monkeypatch.setattr(chroma, "chroma_available", lambda: False)
    yield
    config_module.reset_config()


def test_without_chromadb_memory_is_off(no_chroma):
    from dostuff.agent.instructions import build_base_instructions, build_turn_instructions
    from dostuff.lib.memory.stores import get_episodic_store, get_semantic_store, make_tool_registry_store
    from dostuff.tools.definitions import MEMORY_TOOLS, TOOL_SCHEMAS, default_tool_names

    assert get_semantic_store() is None and get_episodic_store() is None and make_tool_registry_store() is None
    names = default_tool_names()
    assert not MEMORY_TOOLS & set(names)
    assert set(names) | MEMORY_TOOLS == set(TOOL_SCHEMAS)

    base, _ = build_base_instructions()
    assert "save_memory" not in base and "## Long-term memory" not in base
    turn = asyncio.run(build_turn_instructions(base, "hi", None, None, "user"))
    assert "<environment>" in turn and "<relevant_user_facts>" not in turn

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
