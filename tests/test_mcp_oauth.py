"""MCP OAuth: confidential clients from dynamic registration (Supabase), scopes, error details."""
import asyncio
import base64
import json
import urllib.parse

import httpx
import pytest

from dostuff.helpers.mcp import mcp_oauth
from dostuff.helpers.mcp.mcp_oauth import OAuthError, authenticate_via_oauth, refresh_access_token, with_client_auth
from dostuff.lib.mcp.mcp_client_registration_store import MCPClientRegistrationStore

MCP_URL = "https://mcp.example.com/mcp"
AS = "https://auth.example.com"
SCOPES = ["projects:read", "database:read", "database:write"]


def supabase_like(auth_methods: list[str], seen: dict) -> httpx.MockTransport:
    """A server that, like Supabase, always issues a client_secret and requires it at the token endpoint."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://mcp.example.com/.well-known/oauth-protected-resource/mcp":
            return httpx.Response(200, json={"authorization_servers": [AS], "scopes_supported": SCOPES})
        if url == f"{AS}/.well-known/oauth-authorization-server":
            return httpx.Response(200, json={
                "authorization_endpoint": f"{AS}/authorize", "token_endpoint": f"{AS}/token",
                "registration_endpoint": f"{AS}/register", "token_endpoint_auth_methods_supported": auth_methods,
            })
        if url == f"{AS}/register":
            seen["registration"] = request.read()
            return httpx.Response(201, json={"client_id": "new-client", "client_secret": "s3cret"})
        if url == f"{AS}/token":
            form = dict(urllib.parse.parse_qsl(request.read().decode()))
            basic = request.headers.get("authorization", "")
            if basic:
                form["client_id"], _, form["client_secret"] = base64.b64decode(basic.split()[1]).decode().partition(":")
            seen.setdefault("token_forms", []).append(form)
            if form.get("client_secret") != "s3cret" or form.get("client_id") != "new-client":
                return httpx.Response(422, json={"message": "Required parameter: client_secret"})
            return httpx.Response(200, json={"access_token": "tok", "refresh_token": "ref", "expires_in": 3600})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


@pytest.fixture
def oauth_env(monkeypatch, tmp_path):
    """Mock the network, the browser and the loopback callback."""
    def setup(auth_methods: list[str]):
        seen: dict = {}
        transport = supabase_like(auth_methods, seen)
        real_client = httpx.AsyncClient
        monkeypatch.setattr(mcp_oauth.httpx, "AsyncClient", lambda *a, **kw: real_client(*a, transport=transport, **kw))

        class FakeServer:
            def __init__(self, *args):
                pass

            def server_close(self):
                pass
        monkeypatch.setattr(mcp_oauth, "_CallbackServer", FakeServer)
        monkeypatch.setattr(mcp_oauth.webbrowser, "open", lambda url: seen.__setitem__("auth_url", url))

        def callback(server, timeout_s):
            state = urllib.parse.parse_qs(urllib.parse.urlparse(seen["auth_url"]).query)["state"][0]
            return {"code": "the-code", "state": state}
        monkeypatch.setattr(mcp_oauth, "_wait_for_callback", callback)
        store = MCPClientRegistrationStore(path=str(tmp_path / "registrations.json"))
        return seen, store
    return setup


www_auth = 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource/mcp"'


@pytest.mark.parametrize("methods, expected", [
    (["client_secret_basic", "client_secret_post"], "client_secret_post"),
    (["client_secret_basic"], "client_secret_basic"),
])
def test_confidential_client_sends_its_secret(oauth_env, methods, expected):
    seen, store = oauth_env(methods)
    token = asyncio.run(authenticate_via_oauth(MCP_URL, www_auth, store, "supabase"))
    assert token["access_token"] == "tok" and token["client_id"] == "new-client"
    assert json.loads(seen["registration"])["token_endpoint_auth_method"] == expected
    saved = asyncio.run(store.get("supabase"))
    assert saved["client_secret"] == "s3cret" and saved["token_endpoint_auth_method"] == expected
    # every scope the protected resource lists, not the first two
    scope = urllib.parse.parse_qs(urllib.parse.urlparse(seen["auth_url"]).query)["scope"][0]
    assert scope == " ".join(SCOPES)


def test_registration_saved_without_a_secret_is_replaced(oauth_env):
    seen, store = oauth_env(["client_secret_post"])
    asyncio.run(store.save("supabase", {"client_id": "old-client-without-secret"}))
    asyncio.run(authenticate_via_oauth(MCP_URL, www_auth, store, "supabase"))
    assert "registration" in seen
    assert asyncio.run(store.get("supabase"))["client_id"] == "new-client"


def test_token_error_includes_the_server_message(oauth_env, monkeypatch):
    seen, store = oauth_env(["client_secret_post"])
    monkeypatch.setattr(mcp_oauth, "with_client_auth", lambda payload, secret, method: (payload, httpx.Auth()))  # drop the secret
    with pytest.raises(OAuthError, match=r"HTTP 422 .*Required parameter: client_secret"):
        asyncio.run(authenticate_via_oauth(MCP_URL, www_auth, store, "supabase"))


def test_refresh_sends_the_secret(oauth_env):
    seen, _ = oauth_env(["client_secret_post"])
    asyncio.run(refresh_access_token(f"{AS}/token", "ref", client_id="new-client", client_secret="s3cret", auth_method="client_secret_post"))
    assert seen["token_forms"][-1]["client_secret"] == "s3cret"


def test_public_client_sends_no_secret():
    body, auth = with_client_auth({"client_id": "c"}, None, None)
    assert body == {"client_id": "c"} and type(auth) is httpx.Auth
