"""web_search tool: DuckDuckGo HTML parsing, backend choice, and approval."""
import asyncio
import subprocess
import sys

import httpx
import pytest

from dostuff.agent.permissions import PermissionConfig, SessionPermissions, check, describe_approval_keys
from dostuff.tools import web_search as web_search_module
from dostuff.tools.web_search import format_results, parse_ddg_html, web_search

DDG_PAGE = """<html><body><div id="links">
<div class="result results_links web-result"><h2 class="result__title">
<a rel="nofollow" class="result__a" href="https://supabase.com/docs/guides/ai-tools/mcp">Supabase MCP Server | Supabase Docs</a></h2>
<a class="result__snippet" href="https://supabase.com/docs/guides/ai-tools/mcp">Connect your AI tools to <b>Supabase</b> using <b>MCP</b></a></div>
<div class="result"><h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fsupabase-community%2Fsupabase-mcp&amp;rut=x">supabase-mcp</a></h2></div>
<div class="result result--ad"><h2><a class="result__a" href="https://duckduckgo.com/y.js?ad_domain=x">Ad</a></h2></div>
</div></body></html>"""


class _Cfg:
    def __init__(self, searxng_url: str = "") -> None:
        self.searxng_url = searxng_url


@pytest.fixture
def backend(monkeypatch):
    """Route httpx to a handler, with no ddgs and the given SearXNG url."""
    def setup(handler, searxng_url: str = ""):
        real_client = httpx.AsyncClient

        def client(*args, **kwargs):
            return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)
        monkeypatch.setattr(web_search_module.httpx, "AsyncClient", client)
        monkeypatch.setattr(web_search_module, "_ddgs_available", lambda: False)
        monkeypatch.setattr("dostuff.config.get_config", lambda: _Cfg(searxng_url))
    return setup


def test_parse_ddg_html_titles_links_snippets():
    results = parse_ddg_html(DDG_PAGE)
    assert results == [
        {"title": "Supabase MCP Server | Supabase Docs", "url": "https://supabase.com/docs/guides/ai-tools/mcp",
         "snippet": "Connect your AI tools to Supabase using MCP"},
        # redirect links are unwrapped; the ad is dropped
        {"title": "supabase-mcp", "url": "https://github.com/supabase-community/supabase-mcp", "snippet": ""},
    ]


def test_web_search_duckduckgo_by_default(backend):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, html=DDG_PAGE)
    backend(handler)
    out = asyncio.run(web_search("  supabase   mcp ", max_results=1))
    assert seen[0].url.host == "html.duckduckgo.com" and b"q=supabase+mcp" in seen[0].content
    assert out.startswith("Results for 'supabase mcp' (DuckDuckGo):")
    assert "1. Supabase MCP Server | Supabase Docs\n   https://supabase.com/docs/guides/ai-tools/mcp" in out
    assert "supabase-mcp" not in out  # max_results=1


def test_web_search_reports_a_challenge_page(backend):
    backend(lambda request: httpx.Response(200, html="<html><body>Please verify you are human</body></html>"))
    with pytest.raises(RuntimeError, match="rate limiting"):
        asyncio.run(web_search("anything"))


def test_web_search_uses_searxng_when_configured(backend):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith("http://localhost:8080/search?")
        assert request.url.params["format"] == "json"
        return httpx.Response(200, json={"results": [{"title": "T", "url": "https://a.example", "content": "S"}]})
    backend(handler, searxng_url="http://localhost:8080/")
    assert asyncio.run(web_search("q")) == "Results for 'q' (SearXNG):\n\n1. T\n   https://a.example\n   S"


def test_web_search_rejects_empty_query():
    with pytest.raises(ValueError):
        asyncio.run(web_search("   "))
    assert format_results("x", "DuckDuckGo", []) == "No results for 'x' (DuckDuckGo)."


def test_web_search_asks_and_always_covers_every_search():
    ask = PermissionConfig(mode="ask")
    decision = check("web_search", "network", {"query": "supabase mcp"}, None, ask)
    assert decision.action == "ask"
    assert "Search the web:\n  supabase mcp" in decision.prompt
    assert decision.approval_keys == [("web_search", "web_search")]
    assert describe_approval_keys(decision.approval_keys) == "web searches"

    session = SessionPermissions()
    session.approve(decision.approval_keys)
    assert check("web_search", "network", {"query": "other"}, session, ask).action == "allow"
    assert check("web_search", "network", {"query": "q"}, None, PermissionConfig(mode="allowlist", allow=["web_search"])).action == "allow"


@pytest.mark.parametrize("module", ["dostuff", "dostuff.cli"])
def test_python_dash_m_runs_the_cli(module):
    out = subprocess.run([sys.executable, "-m", module, "--version"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.startswith("dostuff ")
