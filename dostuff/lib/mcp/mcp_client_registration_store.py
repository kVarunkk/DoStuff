import json
import os
from typing import Any


class MCPClientRegistrationStore:
    """Persists OAuth client registration info (client_id, and client_secret if
    ever issued) to a local JSON file, keyed by server_name. This lets us reuse
    the same registered client across process restarts instead of calling the
    server's registration_endpoint every time we connect.

    Not thread/process-safe for concurrent writers — fine for a single local
    agent process, not appropriate as-is if multiple processes could write to
    this file at once.
    """

    def __init__(self, path: str = ".dostuff/data/mcp_client_registrations.json"):
        self.path = path

    def _load_all(self) -> dict[str, Any]:
        if not os.path.isfile(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_all(self, data: dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    async def get(self, server_name: str) -> dict[str, Any] | None:
        """Returns the stored registration for a server, or None if not registered yet."""
        return self._load_all().get(server_name)

    async def save(self, server_name: str, registration: dict[str, Any]) -> None:
        """Stores registration info for a server, overwriting any existing entry.

        Args:
            server_name: The logical name used to identify this MCP server in
                your config (not necessarily its URL).
            registration: Dict containing at least "client_id", and optionally
                "client_secret", "redirect_uri", "registration_endpoint".
        """
        data = self._load_all()
        data[server_name] = registration
        self._save_all(data)