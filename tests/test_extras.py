"""The optional extras are named where a plain `pip install dostuff` user looks: doctor, setup, /tools, the TUI."""
import asyncio
import importlib.util
from types import SimpleNamespace

import httpx
import pytest

from dostuff.helpers import extras
from dostuff.helpers.extras import EXTRAS, install_hint, missing_extras
from dostuff.tools import web_search as web_search_module

REAL_FIND_SPEC = importlib.util.find_spec
REAL_CLIENT = httpx.AsyncClient


@pytest.fixture
def without(monkeypatch):
    """Pretend the given top-level modules are not installed (and every other module is)."""
    def setup(*modules: str) -> None:
        def find_spec(name, *args, **kwargs):
            return None if name in modules else REAL_FIND_SPEC("json")
        monkeypatch.setattr(extras.importlib.util, "find_spec", find_spec)
    return setup


def test_missing_extras_and_install_hint(without):
    without("ddgs", "openpyxl")
    assert [e.name for e in missing_extras()] == ["search", "docs"]  # one missing module is enough
    assert install_hint("search", "docs") == 'pip install "dostuff[search,docs]"'


def test_doctor_lists_each_extra_with_its_install_command(without):
    from dostuff.helpers.ui.doctor import extras_checks

    without("PIL")
    checks = {c.label.split(":")[0]: c for c in extras_checks()}
    assert set(checks) == {f"Extra {e.name}" for e in EXTRAS}
    assert all(c.ok is None for c in checks.values())  # information, never a failure
    assert checks["Extra images"].label.startswith("Extra images: not installed")
    assert 'pip install "dostuff[images]"' in checks["Extra images"].label
    assert checks["Extra search"].label.startswith("Extra search: installed")


def test_setup_ends_with_the_missing_extras(without, monkeypatch):
    from dostuff.setup_wizard import extras_note

    monkeypatch.setattr("dostuff.lib.memory.chroma.chroma_available", lambda: False)
    without("ddgs")
    note = extras_note()
    assert 'pip install "dostuff[full]"' in note
    assert "  memory:" in note and "  search:" in note and "  docs:" not in note

    monkeypatch.setattr("dostuff.lib.memory.chroma.chroma_available", lambda: True)
    without()
    assert extras_note() == ""


def test_tools_listing_says_what_missing_extras_change(without, monkeypatch):
    from dostuff.helpers.ui.tools_listing import tools_listing

    monkeypatch.setattr("dostuff.config.get_config", lambda: SimpleNamespace(searxng_url=""))
    without("ddgs", "pypdf")
    text = tools_listing({})
    assert "Without the search extra: web_search reads DuckDuckGo's HTML page" in text
    assert "Without the docs extra:" in text and "Without the images extra" not in text
    assert 'pip install "dostuff[search,docs]" adds them' in text

    monkeypatch.setattr("dostuff.config.get_config", lambda: SimpleNamespace(searxng_url="http://localhost:8080"))
    assert "Without the search extra" not in tools_listing({})  # SearXNG is in use, ddgs wouldn't be

    without()
    assert "Without the" not in tools_listing({})


def _search(monkeypatch, *, ddgs: bool) -> None:
    page = '<a class="result__a" href="https://a.example">A</a>'
    monkeypatch.setattr(web_search_module.httpx, "AsyncClient", lambda *a, **k: REAL_CLIENT(
        *a, transport=httpx.MockTransport(lambda r: httpx.Response(200, html=page)), **k))
    monkeypatch.setattr(web_search_module, "_ddgs_available", lambda: ddgs)
    monkeypatch.setattr(web_search_module, "_search_ddgs", lambda q, n: [])
    monkeypatch.setattr("dostuff.config.get_config", lambda: SimpleNamespace(searxng_url=""))
    asyncio.run(web_search_module.web_search("q"))


def test_tui_says_once_that_web_search_fell_back_to_the_html_page(monkeypatch):
    from dostuff.cli_tui import DostuffTUI

    monkeypatch.setattr(web_search_module, "_html_fallback_used", False)
    shown: list[str] = []
    app = SimpleNamespace(_search_extra_noted=False, _append=lambda line, msg_type: shown.append(line))

    _search(monkeypatch, ddgs=True)
    DostuffTUI._note_search_fallback(app)  # type: ignore[arg-type]
    assert shown == []  # ddgs did the search: nothing to say

    _search(monkeypatch, ddgs=False)
    DostuffTUI._note_search_fallback(app)  # type: ignore[arg-type]
    DostuffTUI._note_search_fallback(app)  # type: ignore[arg-type]
    assert len(shown) == 1 and 'pip install "dostuff[search]"' in shown[0]
