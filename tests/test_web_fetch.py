"""web_fetch tool: HTML → text, content types, and per-domain approval."""
import asyncio

import httpx
import pytest

from dostuff.agent.permissions import PermissionConfig, SessionPermissions, check, describe_approval_keys
from dostuff.tools import web_fetch as web_fetch_module
from dostuff.tools.web_fetch import html_to_text, web_fetch

PAGE = """<html><head><title>Docs  Page</title><style>body{color:red}</style>
<script>alert(1)</script></head><body>
<nav><a href="#top">skip</a></nav>
<h1>Install</h1><p>Run   the <a href="/setup">setup guide</a> first.</p>
<ul><li>one</li><li>two</li></ul>
<pre>pip install  dostuff
dostuff doctor</pre>
</body></html>"""


def test_html_to_text_keeps_structure_and_drops_scripts():
    title, text = html_to_text(PAGE, "https://example.com/docs/")
    assert title == "Docs Page"
    assert "# Install" in text
    assert "Run the [setup guide](https://example.com/setup) first." in text
    assert "- one\n- two" in text
    assert "pip install  dostuff\ndostuff doctor" in text  # <pre> keeps its spacing
    assert "alert" not in text and "color:red" not in text
    assert "skip" in text and "#top" not in text  # in-page anchors keep their text, not the link


def _patch_transport(monkeypatch, handler):
    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(web_fetch_module.httpx, "AsyncClient", client)


def test_web_fetch_html_json_binary_and_cut(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/page":
            return httpx.Response(200, html=PAGE)
        if request.url.path == "/api":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/img":
            return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})
        if request.url.path == "/missing":
            return httpx.Response(404, html="<html><head><title>Not Found</title></head><body><nav>menu</nav></body></html>")
        if request.url.path == "/api-error":
            return httpx.Response(400, json={"error": "bad " + "x" * 5000})
        return httpx.Response(200, text="x" * 500)
    _patch_transport(monkeypatch, handler)

    page = asyncio.run(web_fetch("https://example.com/page"))
    assert page.startswith("URL: https://example.com/page\nStatus: 200\nContent-Type: text/html\nTitle: Docs Page")
    assert "# Install" in page
    assert '{"ok":true}' in asyncio.run(web_fetch("https://example.com/api")).replace(" ", "")
    assert "(binary content, not shown)" in asyncio.run(web_fetch("https://example.com/img"))
    cut = asyncio.run(web_fetch("https://example.com/long", max_chars=100))
    assert "[… cut at 100 of 500 chars" in cut
    missing = asyncio.run(web_fetch("https://example.com/missing"))
    assert "Status: 404" in missing and "Title: Not Found" in missing
    assert "page not shown" in missing and "menu" not in missing
    api_error = asyncio.run(web_fetch("https://example.com/api-error"))
    assert '"error"' in api_error and "[… cut at 2000 of" in api_error


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "example.com"])
def test_web_fetch_rejects_non_http(url):
    with pytest.raises(ValueError):
        asyncio.run(web_fetch(url))


def test_web_fetch_asks_per_domain():
    ask = PermissionConfig(mode="ask")
    decision = check("web_fetch", "network", {"url": "https://Docs.Python.org/3/"}, None, ask)
    assert decision.action == "ask"
    assert decision.approval_keys == [("web_fetch", "web_fetch:docs.python.org")]
    assert describe_approval_keys(decision.approval_keys) == "fetches from docs.python.org"
    assert "Fetch URL:\n  https://Docs.Python.org/3/" in decision.prompt

    session = SessionPermissions()
    session.approve(decision.approval_keys)
    assert check("web_fetch", "network", {"url": "https://docs.python.org/other"}, session, ask).action == "allow"
    assert check("web_fetch", "network", {"url": "https://evil.example/?q=secret"}, session, ask).action == "ask"

    allow = PermissionConfig(mode="allowlist", allow=["web_fetch:docs.python.org"])
    assert check("web_fetch", "network", {"url": "https://docs.python.org/x"}, None, allow).action == "allow"
    assert check("web_fetch", "network", {"url": "https://pypi.org/x"}, None, allow).action == "ask"
    assert check("web_fetch", "network", {"url": "https://pypi.org/x"}, None, PermissionConfig(mode="auto")).action == "allow"
