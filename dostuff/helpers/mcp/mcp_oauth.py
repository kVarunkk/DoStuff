import http.server
import re
import secrets
import time
import urllib.parse
import webbrowser
import httpx
import pkce
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from dostuff.lib.mcp.mcp_client_registration_store import MCPClientRegistrationStore

CLIENT_ID = "mcp-generic-client"
REDIRECT_PORT = 8085
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/callback"
CLIENT_ID_METADATA_DOCUMENT="https://d1b7jdanqdrk6e.cloudfront.net/DoStuff/client-metadata.json"
CALLBACK_TIMEOUT_S = 120.0

# The loopback callback port is shared, so only one browser flow may run at a time
_oauth_lock = asyncio.Lock()


class OAuthError(Exception):
    """The authorization flow failed (denied, timed out, state mismatch, ...)."""


class _CallbackServer(http.server.HTTPServer):
    """Loopback server that records the first OAuth callback it receives."""
    result: dict[str, str] | None = None


class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    server: _CallbackServer

    def do_GET(self):
        self._handle_callback()

    def do_POST(self):
        self._handle_callback()

    def _handle_callback(self):
        parsed_url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed_url.query)

        if self.command == "POST":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length:
                body = self.rfile.read(content_length).decode("utf-8")
                content_type = self.headers.get("Content-Type", "")
                try:
                    if "application/json" in content_type:
                        body_params = {k: [str(v)] for k, v in json.loads(body).items()}
                    else:
                        body_params = urllib.parse.parse_qs(body)
                except (json.JSONDecodeError, AttributeError):
                    body_params = {}
                for key, value in body_params.items():
                    query.setdefault(key, value)

        if "code" not in query and "error" not in query:
            # e.g. a favicon request; keep waiting for the real callback
            self.send_response(404)
            self.end_headers()
            return

        self.server.result = {k: v[0] for k, v in query.items() if v}
        ok = "code" in query
        self.send_response(200 if ok else 400)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        if ok:
            self.wfile.write(b"<h1>Authentication Successful!</h1><p>You may close this window and return to your terminal/agent.</p>")
        else:
            self.wfile.write(b"<h1>Authentication failed</h1><p>You may close this window.</p>")

    def log_message(self, format, *args):
        return


def parse_www_authenticate(header_value: str) -> dict[str, str]:
    """Extracts parameters from a WWW-Authenticate header (resource_metadata, scope, realm, etc.)."""
    params: dict[str, str] = {}
    if not header_value:
        return params

    matches = re.findall(r'(\w+)=(?:"([^"]+)"|([^\s,]+))', header_value)
    for key, val_quoted, val_raw in matches:
        params[key.lower()] = val_quoted or val_raw
    return params


def canonical_resource(mcp_server_url: str) -> str:
    """RFC 8707 resource indicator: the server URL without query or fragment."""
    parsed = urllib.parse.urlparse(mcp_server_url)
    return urllib.parse.urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", "", ""))


@dataclass
class OAuthServerInfo:
    authorization_endpoint: str
    token_endpoint: str
    scope: str
    registration_endpoint: str | None
    client_id_metadata_document_supported: bool
    # token_endpoint_auth_methods_supported; empty when the server doesn't say
    token_auth_methods: list[str] = field(default_factory=list)

    @property
    def needs_client_secret(self) -> bool:
        """The token endpoint takes no public clients (Supabase: client_secret_basic / _post only)."""
        return bool(self.token_auth_methods) and "none" not in self.token_auth_methods


def pick_secret_auth_method(methods: list[str]) -> str:
    """client_secret_post when offered, else client_secret_basic (the RFC 6749 default)."""
    return "client_secret_post" if "client_secret_post" in methods else "client_secret_basic"


def with_client_auth(
    payload: dict[str, str], client_secret: str | None, auth_method: str | None
) -> tuple[dict[str, str], httpx.Auth]:
    """The token request body and HTTP auth for a client: public (client_id only), secret in
    the body (client_secret_post), or HTTP Basic (client_secret_basic)."""
    if not client_secret:
        return payload, httpx.Auth()  # no-op: public client
    if auth_method == "client_secret_basic":
        body = {k: v for k, v in payload.items() if k != "client_id"}
        # RFC 6749 §2.3.1: id and secret are form-urlencoded before Basic encoding
        quote = urllib.parse.quote_plus
        return body, httpx.BasicAuth(quote(payload["client_id"]), quote(client_secret))
    return {**payload, "client_secret": client_secret}, httpx.Auth()


def _http_error(what: str, res: httpx.Response) -> OAuthError:
    """The server's own message (e.g. "Required parameter: client_secret"), not just the status."""
    detail = " ".join(res.text.split())[:300] or res.reason_phrase
    return OAuthError(f"{what} failed (HTTP {res.status_code} from {res.request.url}): {detail}")


async def discover_oauth_endpoints(mcp_server_url: str, www_auth_header: str) -> OAuthServerInfo:
    """Vendor-agnostic OAuth discovery, per RFC 9728 (protected resource metadata)
    and RFC 8414 (authorization server metadata), with an OIDC discovery fallback.

    Scope, as the MCP spec says: the WWW-Authenticate `scope`, else the protected resource's
    `scopes_supported`, else OIDC-style scopes from the authorization server metadata.
    """
    www_auth_params = parse_www_authenticate(www_auth_header)
    requested_scope = www_auth_params.get("scope", "")

    resource_metadata_url = www_auth_params.get("resource_metadata") or (
        f"{mcp_server_url.rstrip('/')}/.well-known/oauth-protected-resource"
    )

    async with httpx.AsyncClient(follow_redirects=True) as client:
        res = await client.get(resource_metadata_url)
        res.raise_for_status()
        res_meta = res.json()

        if not requested_scope and res_meta.get("scopes_supported"):
            requested_scope = " ".join(res_meta["scopes_supported"])
        auth_servers = res_meta.get("authorization_servers", [])
        if not auth_servers:
            raise ValueError(f"No 'authorization_servers' array found in {resource_metadata_url}")

        issuer = auth_servers[0].rstrip("/")
        parsed_issuer = urllib.parse.urlparse(issuer)

        rfc8414_url = f"{parsed_issuer.scheme}://{parsed_issuer.netloc}/.well-known/oauth-authorization-server{parsed_issuer.path}"
        oidc_url = f"{issuer}/.well-known/openid-configuration"

        meta_data = None
        for discovery_url in [rfc8414_url, oidc_url, f"{issuer}/.well-known/oauth-authorization-server"]:
            try:
                disc_res = await client.get(discovery_url, timeout=4.0)
                if disc_res.status_code == 200:
                    meta_data = disc_res.json()
                    break
            except Exception:
                continue

        if not meta_data:
            raise ValueError(f"Could not resolve OAuth 2.1 metadata for issuer '{issuer}'")

        auth_endpoint = meta_data.get("authorization_endpoint")
        token_endpoint = meta_data.get("token_endpoint")
        registration_endpoint = meta_data.get("registration_endpoint")
        client_id_metadata_document_supported = meta_data.get("client_id_metadata_document_supported") or False

        if not auth_endpoint or not token_endpoint:
            raise ValueError("Discovered metadata is missing 'authorization_endpoint' or 'token_endpoint'")

        if not requested_scope and meta_data.get("scopes_supported"):
            supported = meta_data["scopes_supported"]
            requested_scope = " ".join([s for s in supported if s in ["openid", "profile", "email"]] or supported[:2])

        return OAuthServerInfo(
            authorization_endpoint=auth_endpoint,
            token_endpoint=token_endpoint,
            scope=requested_scope or "openid",
            registration_endpoint=registration_endpoint,
            client_id_metadata_document_supported=client_id_metadata_document_supported,
            token_auth_methods=[str(m) for m in meta_data.get("token_endpoint_auth_methods_supported") or []],
        )


def _wait_for_callback(server: _CallbackServer, timeout_s: float) -> dict[str, str] | None:
    """Serve requests until the callback arrives or the timeout passes (blocking)."""
    deadline = time.monotonic() + timeout_s
    server.timeout = 1.0  # handle_request() returns after this many idle seconds
    while server.result is None and time.monotonic() < deadline:
        server.handle_request()
    return server.result


async def authenticate_via_oauth(mcp_server_url: str, www_auth_header: str, registration_store: MCPClientRegistrationStore, server_name: str) -> dict:
    """Discovers OAuth provider details, runs the PKCE authorization-code flow via a
    local browser + loopback listener, and returns the token response plus
    `token_endpoint`, `client_id` and `resource` (needed for refresh).
    """
    info = await discover_oauth_endpoints(mcp_server_url, www_auth_header)

    existing_registration = await registration_store.get(server_name) or {}
    client_secret: str | None = existing_registration.get("client_secret")
    auth_method: str | None = existing_registration.get("token_endpoint_auth_method")
    # A registration saved without its secret (before dostuff kept it) can't get a token from a
    # server that requires one, so register again
    reusable = bool(existing_registration.get("client_id")) and (client_secret or not info.needs_client_secret)

    if reusable:
        client_id = existing_registration["client_id"]
    elif info.client_id_metadata_document_supported and not info.needs_client_secret:
        client_id = CLIENT_ID_METADATA_DOCUMENT
    else:
        if not info.registration_endpoint:
            raise ValueError(
                f"No cached client_id for '{server_name}' and server does not "
                "support dynamic client registration."
            )
        requested_method = pick_secret_auth_method(info.token_auth_methods) if info.needs_client_secret else "none"
        client_info = await register_oauth_client(info.registration_endpoint, REDIRECT_URI, requested_method)
        client_id = client_info["client_id"]
        # Servers may issue a secret whatever was asked for (Supabase always does); it must be sent back
        client_secret = client_info.get("client_secret") or None
        auth_method = client_info.get("token_endpoint_auth_method") or requested_method
        if client_secret and auth_method == "none":
            auth_method = pick_secret_auth_method(info.token_auth_methods)
        registration = {
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "registration_endpoint": info.registration_endpoint,
            "token_endpoint_auth_method": auth_method,
        }
        if client_secret:
            registration["client_secret"] = client_secret
        await registration_store.save(server_name, registration)

    resource = canonical_resource(mcp_server_url)
    code_verifier, code_challenge = pkce.generate_pkce_pair()
    state = secrets.token_urlsafe(32)

    params: dict[str, str] = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "resource": resource,
    }
    if info.scope:
        params["scope"] = info.scope

    auth_url = f"{info.authorization_endpoint}?{urllib.parse.urlencode(params)}"

    async with _oauth_lock:
        server = _CallbackServer(("localhost", REDIRECT_PORT), OAuthCallbackHandler)
        try:
            webbrowser.open(auth_url)
            callback = await asyncio.to_thread(_wait_for_callback, server, CALLBACK_TIMEOUT_S)
        finally:
            server.server_close()  # always release the port

    if callback is None:
        raise OAuthError(f"OAuth for '{server_name}' timed out after {CALLBACK_TIMEOUT_S:.0f}s waiting for the browser callback.")
    if "error" in callback:
        detail = callback.get("error_description") or callback["error"]
        raise OAuthError(f"OAuth for '{server_name}' failed: {detail}")
    if not secrets.compare_digest(callback.get("state", ""), state):
        raise OAuthError(f"OAuth for '{server_name}' failed: state mismatch (possible CSRF); try again.")

    async with httpx.AsyncClient() as client:
        token_payload = {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": callback["code"],
            "redirect_uri": REDIRECT_URI,
            "code_verifier": code_verifier,
            "resource": resource,
        }
        body, basic_auth = with_client_auth(token_payload, client_secret, auth_method)
        res = await client.post(info.token_endpoint, data=body, auth=basic_auth)
        if res.is_error:
            raise _http_error(f"OAuth for '{server_name}': token exchange", res)

        token_data: dict[str, Any] = res.json()
        token_data["token_endpoint"] = info.token_endpoint
        token_data["client_id"] = client_id
        token_data["resource"] = resource
        return token_data


async def register_oauth_client(registration_endpoint: str, redirect_uri: str, auth_method: str = "none") -> dict:
    """RFC 7591 dynamic registration. The reply may hold a `client_secret` even when "none" was asked for."""
    async with httpx.AsyncClient() as client:
        res = await client.post(
            registration_endpoint,
            json={
                "client_name": "DoStuff Agent",
                "redirect_uris": [redirect_uri],
                "token_endpoint_auth_method": auth_method,
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"]
            },
        )
        if res.is_error:
            raise _http_error("OAuth client registration", res)
        return res.json()


async def refresh_access_token(
    token_endpoint: str,
    refresh_token: str,
    client_id: str | None = None,
    resource: str | None = None,
    client_secret: str | None = None,
    auth_method: str | None = None,
) -> dict:
    """Refresh with the client the token was issued to (falls back to the generic id), and its
    secret when the registration has one."""
    async with httpx.AsyncClient() as client:
        payload = {
            "grant_type": "refresh_token",
            "client_id": client_id or CLIENT_ID,
            "refresh_token": refresh_token,
        }
        if resource:
            payload["resource"] = resource
        body, basic_auth = with_client_auth(payload, client_secret, auth_method)
        res = await client.post(token_endpoint, data=body, auth=basic_auth)
        res.raise_for_status()
        return res.json()
