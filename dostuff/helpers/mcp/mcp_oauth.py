import http.server
import re
import urllib.parse
import webbrowser
import httpx
import pkce
import asyncio
import json

from dostuff.lib.mcp.mcp_client_registration_store import MCPClientRegistrationStore

CLIENT_ID = "mcp-generic-client"
REDIRECT_PORT = 8085
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/callback"
CLIENT_ID_METADATA_DOCUMENT="https://d1b7jdanqdrk6e.cloudfront.net/DoStuff/client-metadata.json"


class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    auth_code = None

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
                # print(f"RAW POST BODY: {body}")
    
                content_type = self.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    body_params = {k: [v] for k, v in json.loads(body).items()}
                else:
                    body_params = urllib.parse.parse_qs(body)
    
                for key, value in body_params.items():
                    query.setdefault(key, value)  
    
        if "code" in query:
            OAuthCallbackHandler.auth_code = query["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<h1>Authentication Successful!</h1><p>You may close this window and return to your terminal/agent.</p>"
            )
        else:
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args):
        return


def parse_www_authenticate(header_value: str) -> dict:
    """Extracts parameters from a WWW-Authenticate header (resource_metadata, scope, realm, etc.)."""
    params = {}
    if not header_value:
        return params

    matches = re.findall(r'(\w+)=(?:"([^"]+)"|([^\s,]+))', header_value)
    for key, val_quoted, val_raw in matches:
        params[key.lower()] = val_quoted or val_raw
    return params


async def discover_oauth_endpoints(mcp_server_url: str, www_auth_header: str) -> tuple[str, str, str, str, bool]:
    """Vendor-agnostic OAuth discovery, per RFC 9728 (protected resource metadata)
    and RFC 8414 (authorization server metadata), with an OIDC discovery fallback.

    Returns:
        (authorization_endpoint, token_endpoint, requested_scope, registration_endpoint)
    """
    www_auth_params = parse_www_authenticate(www_auth_header)
    requested_scope = www_auth_params.get("scope", "openid")

    resource_metadata_url = www_auth_params.get("resource_metadata") or (
        f"{mcp_server_url.rstrip('/')}/.well-known/oauth-protected-resource"
    )

    async with httpx.AsyncClient(follow_redirects=True) as client:
        res = await client.get(resource_metadata_url)
        res.raise_for_status()
        res_meta = res.json()

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

        if "scope" not in www_auth_params and "scopes_supported" in meta_data:
            supported = meta_data["scopes_supported"]
            requested_scope = " ".join([s for s in supported if s in ["openid", "profile", "email"]] or supported[:2])

        return auth_endpoint, token_endpoint, requested_scope, registration_endpoint, client_id_metadata_document_supported


async def authenticate_via_oauth(mcp_server_url: str, www_auth_header: str, registration_store: MCPClientRegistrationStore, server_name: str) -> dict:
    """Discovers OAuth provider details, runs the PKCE authorization-code flow via a
    local browser + loopback listener, and returns an access token.
    """

    auth_endpoint, token_endpoint, scope, registration_endpoint, client_id_metadata_document_supported = await discover_oauth_endpoints(mcp_server_url, www_auth_header)

    existing_registration = await registration_store.get(server_name)

    if existing_registration and existing_registration.get("client_id"):
        client_id = existing_registration["client_id"]
    elif client_id_metadata_document_supported:
        client_id = CLIENT_ID_METADATA_DOCUMENT
    else:
        if not registration_endpoint:
            raise ValueError(
                f"No cached client_id for '{server_name}' and server does not "
                "support dynamic client registration."
            )
        client_info = await register_oauth_client(registration_endpoint, REDIRECT_URI)
        client_id = client_info["client_id"]
        await registration_store.save(
            server_name,
            {
                "client_id": client_id,
                "redirect_uri": REDIRECT_URI,
                "registration_endpoint": registration_endpoint,
            },
        )

    code_verifier, code_challenge = pkce.generate_pkce_pair()

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if scope:
        params["scope"] = scope

    auth_url = f"{auth_endpoint}?{urllib.parse.urlencode(params)}"

    # print(f"\nOpening browser for OAuth authentication ({mcp_server_url})...")
    webbrowser.open(auth_url)

    server = http.server.HTTPServer(("localhost", REDIRECT_PORT), OAuthCallbackHandler)
    OAuthCallbackHandler.auth_code = None

    await asyncio.to_thread(lambda: _wait_for_callback(server))

    code = OAuthCallbackHandler.auth_code

    async with httpx.AsyncClient() as client:
        token_payload = {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": code_verifier,
        }
        res = await client.post(token_endpoint, data=token_payload)
        res.raise_for_status()

        token_data = res.json()
        token_data["token_endpoint"] = token_endpoint
        return token_data


async def register_oauth_client(registration_endpoint: str, redirect_uri: str) -> dict:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            registration_endpoint,
            json={
                "client_name": "DoStuff Agent",
                "redirect_uris": [redirect_uri],
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"]
            },
        )
        res.raise_for_status()
        return res.json()

async def refresh_access_token(token_endpoint: str, refresh_token: str) -> dict:
    async with httpx.AsyncClient() as client:
        payload = {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
        }
        res = await client.post(token_endpoint, data=payload)
        res.raise_for_status()
        return res.json()

def _wait_for_callback(server: http.server.HTTPServer):
    while OAuthCallbackHandler.auth_code is None:
        server.handle_request()   