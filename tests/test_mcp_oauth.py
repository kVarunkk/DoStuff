"""MCP OAuth: confidential clients from dynamic registration (Supabase), scopes, error details."""
import asyncio
import base64
import json
import time
import urllib.parse

import httpx
import httpx2
import pytest

from dostuff.helpers.mcp import mcp_oauth, token_store
from dostuff.helpers.mcp.mcp_oauth import OAuthError, authenticate_via_oauth, refresh_access_token, with_client_auth
from dostuff.lib.mcp import mcp_client as mcp_client_mod
from dostuff.lib.mcp.mcp_client import MCPClient, _OAuthBearer
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


# --- token persistence and refresh ---------------------------------------------------------

class FakeKeyring:
    """Like Windows Credential Manager: refuses secrets over 1280 chars."""
    def __init__(self):
        self.items: dict[tuple[str, str], str] = {}

    def get_password(self, service, name):
        return self.items.get((service, name))

    def set_password(self, service, name, value):
        if len(value) > 1280:
            raise OSError("The stub received bad data.")
        self.items[(service, name)] = value

    def delete_password(self, service, name):
        if (service, name) not in self.items:
            raise KeyError(name)
        del self.items[(service, name)]


@pytest.fixture
def fake_keyring(monkeypatch):
    ring = FakeKeyring()
    monkeypatch.setattr(token_store, "_keyring", ring)
    return ring


def test_long_token_is_split_across_entries(fake_keyring):
    token = {"access_token": "a" * 3000, "refresh_token": "r" * 1500}
    assert token_store.save_token(MCP_URL, token)
    assert token_store.load_token(MCP_URL) == token
    assert len(fake_keyring.items) > 1
    token_store.save_token(MCP_URL, {"access_token": "short"})  # replaces the chunks
    assert token_store.load_token(MCP_URL) == {"access_token": "short"} and len(fake_keyring.items) == 1
    token_store.delete_token(MCP_URL)
    assert fake_keyring.items == {}


def make_client(tmp_path, monkeypatch, access_token="old", expires_in=3600, refresh=None):
    monkeypatch.setattr(token_store, "_keyring", FakeKeyring())
    client = MCPClient(MCPClientRegistrationStore(path=str(tmp_path / "reg.json")), None)
    client._tokens["srv"] = {
        "access_token": access_token, "refresh_token": "ref", "expires_at": time.time() + expires_in,
        "token_endpoint": f"{AS}/token", "client_id": "c", "resource": None,
    }
    calls = []

    async def fake_refresh(endpoint, refresh_token, **kw):
        calls.append(refresh_token)
        if refresh is None:
            return {"access_token": "new", "expires_in": 3600}
        raise refresh

    async def no_browser(*a, **kw):
        raise AssertionError("the browser flow ran")
    monkeypatch.setattr(mcp_client_mod, "refresh_access_token", fake_refresh)
    monkeypatch.setattr(mcp_client_mod, "authenticate_via_oauth", no_browser)
    return client, calls


def test_expired_token_is_refreshed_without_a_browser(tmp_path, monkeypatch):
    client, calls = make_client(tmp_path, monkeypatch, expires_in=-10)
    assert asyncio.run(client._get_or_refresh_token("srv", MCP_URL)) == "new"
    assert calls == ["ref"] and token_store.load_token(MCP_URL)["refresh_token"] == "ref"


def test_401_mid_session_refreshes_and_retries(tmp_path, monkeypatch):
    client, calls = make_client(tmp_path, monkeypatch)
    seen = []

    def handler(request):
        seen.append(request.headers["authorization"])
        return httpx2.Response(200 if request.headers["authorization"] == "Bearer new" else 401)

    async def go():
        async with httpx2.AsyncClient(auth=_OAuthBearer(client, "srv", MCP_URL), transport=httpx2.MockTransport(handler)) as http:
            return (await http.post(MCP_URL, json={})).status_code
    assert asyncio.run(go()) == 200
    assert seen == ["Bearer old", "Bearer new"] and calls == ["ref"]


def test_reauthenticate_refreshes_before_signing_in_again(tmp_path, monkeypatch):
    client, calls = make_client(tmp_path, monkeypatch)
    client.server_meta["srv"] = {"url": MCP_URL, "transport": "streamable-http", "headers": {}}
    reconnects = []

    async def fake_connect(**kw):
        reconnects.append(kw["server_name"])
    monkeypatch.setattr(client, "connect_to_server", fake_connect)
    asyncio.run(client.reauthenticate("srv"))
    assert calls == ["ref"] and client._tokens["srv"]["access_token"] == "new" and reconnects == ["srv"]


def test_reauthenticate_drops_a_dead_refresh_token(tmp_path, monkeypatch):
    client, calls = make_client(tmp_path, monkeypatch, refresh=httpx.HTTPError("invalid_grant"))
    client.server_meta["srv"] = {"url": MCP_URL, "transport": "streamable-http", "headers": {}}

    async def fake_connect(**kw):
        pass
    monkeypatch.setattr(client, "connect_to_server", fake_connect)
    asyncio.run(client.reauthenticate("srv"))
    assert "srv" not in client._tokens
