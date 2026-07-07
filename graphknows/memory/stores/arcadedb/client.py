"""Thin async httpx client for the ArcadeDB REST API.

Covers the subset of endpoints needed by graphknows:
- Database lifecycle (create / drop / exists)
- Read-only queries  (POST /api/v1/query/{db})
- Read-write commands (POST /api/v1/command/{db})
- Explicit transactions (begin / commit / rollback)

Authentication uses HTTP Basic Auth (root + server password).
Transaction state is passed via the ``arcadedb-session-id`` header.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _raise_for_status(r: httpx.Response) -> None:
    """Like ``Response.raise_for_status`` but include the response body.

    ArcadeDB returns the real error (``"detail"`` / ``"exception"``) in the
    body; the default httpx message only carries the status code, which hides
    idempotent signals like "already exists" from callers that match on them.
    """
    if r.is_success:
        return
    try:
        body = r.text
    except Exception:  # pragma: no cover - body already consumed/streamed
        body = ""
    raise httpx.HTTPStatusError(
        f"{r.status_code} {r.reason_phrase} for {r.request.url}: {body}",
        request=r.request,
        response=r,
    )


class ArcadeDBClient:
    """Async REST client for ArcadeDB.

    Args:
        base_url: HTTP base URL, e.g. ``http://arcadedb:2480``.
        user: ArcadeDB server user (default ``root``).
        password: ArcadeDB server password.
    """

    def __init__(
        self,
        base_url: str,
        user: str = "root",
        password: str = "changeme",  # nosec B107 — dev default, overridden by settings
    ) -> None:
        self._base = base_url.rstrip("/")
        self._auth = (user, password)
        self._client: httpx.AsyncClient | None = None

    @property
    def connected(self) -> bool:
        """True when the underlying httpx pool is open."""
        return self._client is not None

    async def connect(self) -> None:
        """Open the underlying httpx connection pool.

        Idempotent: calling connect() on an already-connected client is a no-op,
        so stores that share one client can each connect() safely without leaking
        a second httpx pool.
        """
        if self._client is not None:
            return
        transport = httpx.AsyncHTTPTransport(retries=3)
        self._client = httpx.AsyncClient(
            auth=self._auth,
            # Separate connect vs. read timeouts: fast-fail on connection issues,
            # allow longer reads for large Cypher result sets.
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0),
            headers={"Content-Type": "application/json"},
            transport=transport,
        )

    async def close(self) -> None:
        """Close the connection pool. Idempotent."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _c(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("ArcadeDBClient not connected — call connect() first")
        return self._client

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------

    async def _server_command(self, command: str) -> dict:
        """Execute a server-level SQL command via /api/v1/server."""
        r = await self._c().post(
            f"{self._base}/api/v1/server",
            json={"command": command},
        )
        _raise_for_status(r)
        return r.json()

    async def create_database(self, name: str) -> None:
        """Create a database; silently succeed if it already exists."""
        if not await self.database_exists(name):
            await self._server_command(f"create database {name}")

    async def drop_database(self, name: str) -> None:
        """Drop a database; silently succeed if it does not exist."""
        if await self.database_exists(name):
            await self._server_command(f"drop database {name}")

    async def database_exists(self, name: str) -> bool:
        """Return True if the database exists."""
        r = await self._c().get(f"{self._base}/api/v1/exists/{name}")
        if r.status_code == 200:
            return bool(r.json().get("result", False))
        return False

    # ------------------------------------------------------------------
    # Query / command
    # ------------------------------------------------------------------

    async def command(
        self,
        db: str,
        command: str,
        language: str = "cypher",
        params: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a read-write command and return the result list.

        Args:
            db: Target database name.
            command: Cypher or SQL command string.
            language: ``"cypher"`` (default) or ``"sql"``.
            params: Named parameters referenced with ``$name`` in the command.
            session_id: Active transaction session ID (from :meth:`begin`).
        """
        headers: dict[str, str] = {}
        if session_id:
            headers["arcadedb-session-id"] = session_id
        body: dict[str, Any] = {"language": language, "command": command}
        if params:
            body["params"] = params
        r = await self._c().post(
            f"{self._base}/api/v1/command/{db}",
            json=body,
            headers=headers,
        )
        _raise_for_status(r)
        return r.json().get("result", [])

    async def query(
        self,
        db: str,
        command: str,
        language: str = "cypher",
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a read-only query and return the result list.

        Args:
            db: Target database name.
            command: Cypher or SQL query string.
            language: ``"cypher"`` (default) or ``"sql"``.
            params: Named parameters referenced with ``$name`` in the query.
        """
        body: dict[str, Any] = {"language": language, "command": command}
        if params:
            body["params"] = params
        r = await self._c().post(
            f"{self._base}/api/v1/query/{db}",
            json=body,
        )
        _raise_for_status(r)
        return r.json().get("result", [])

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    async def begin(self, db: str) -> str:
        """Begin a transaction and return the ``arcadedb-session-id`` token."""
        r = await self._c().post(f"{self._base}/api/v1/begin/{db}")
        _raise_for_status(r)
        return r.headers["arcadedb-session-id"]

    async def commit(self, db: str, session_id: str) -> None:
        """Commit a transaction."""
        r = await self._c().post(
            f"{self._base}/api/v1/commit/{db}",
            headers={"arcadedb-session-id": session_id},
        )
        _raise_for_status(r)

    async def rollback(self, db: str, session_id: str) -> None:
        """Rollback a transaction (best-effort)."""
        r = await self._c().post(
            f"{self._base}/api/v1/rollback/{db}",
            headers={"arcadedb-session-id": session_id},
        )
        if r.status_code not in (200, 204):
            logger.warning("ArcadeDB rollback returned HTTP %s", r.status_code)
