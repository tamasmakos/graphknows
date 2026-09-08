"""Thin async httpx client for the ArcadeDB REST API.

Covers the subset of endpoints needed by graphknows:
- Database lifecycle (create / drop / exists)
- Read-only queries  (POST /api/v1/query/{db})
- Read-write commands (POST /api/v1/command/{db})
- Transactions      (POST /api/v1/begin|commit|rollback/{db})

Authentication uses HTTP Basic Auth (root + server password).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

import httpx

from graphknows.exceptions import StoreError

# ArcadeDB pins an open HTTP transaction to this header; every statement that
# carries it joins that transaction. A ContextVar rather than an attribute
# because ingestion writes several chunks concurrently on one client: each
# asyncio task gets its own copy, so one chunk's session never leaks into
# another's statements.
SESSION_HEADER = "arcadedb-session-id"
_session: ContextVar[str | None] = ContextVar("arcadedb_session", default=None)

# _send's own retry budget (FR-034), on top of AsyncHTTPTransport(retries=3).
# The transport's retries cover only a failed *connection attempt*; they do
# not cover a response that arrived (a 5xx — ArcadeDB wraps every server-side
# exception, lock contention included, as a 500 with no dedicated status) nor
# a TransportError raised after the connection was already open (a read/write
# timeout mid-request). Backoff is small and unjittered: this is a same-host
# service call on the ingest hot path, not a multi-tenant provider API.
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_BASE_S = 0.1
_RETRY_BACKOFF_CAP_S = 2.0


def _raise_for_status(r: httpx.Response) -> None:
    """Like ``Response.raise_for_status`` but include the response body.

    ArcadeDB returns the real error (``"detail"`` / ``"exception"``) in the
    body; the default httpx message only carries the status code, which hides
    idempotent signals like "already exists" from callers that match on them.

    Raises :class:`StoreError`, not the raw ``httpx`` exception — nothing above
    this client should ever have to catch ``httpx`` directly — chaining the
    original as ``__cause__`` so the status code and response stay reachable.
    """
    if r.is_success:
        return
    try:
        body = r.text
    except Exception:  # pragma: no cover - body already consumed/streamed
        body = ""
    cause = httpx.HTTPStatusError(
        f"{r.status_code} {r.reason_phrase} for {r.request.url}: {body}",
        request=r.request,
        response=r,
    )
    raise StoreError(str(cause)) from cause


logger = logging.getLogger(__name__)


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

    async def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Issue one request, retrying transient failures (FR-034).

        Retried, up to ``_RETRY_ATTEMPTS`` total tries: an HTTP 5xx response and
        a ``httpx.TransportError`` raised after the connection was already open
        (connection-*establishment* failures are the transport's own job, via
        ``AsyncHTTPTransport(retries=3)``). Any response under 500 — including
        one carrying an application error an idempotent caller matches on, such
        as "already exists" — is returned on the first try, unretried.

        Retrying a write this way trades a rare double-apply (the first attempt
        actually reached ArcadeDB but the response was what timed out) for not
        surfacing a transient 5xx/timeout as a hard failure; ``transaction()``
        is how a caller that cannot accept that gets atomicity instead.

        A final ``httpx.TransportError`` still becomes :class:`StoreError` —
        the counterpart of ``_raise_for_status`` for a request that never got a
        response at all — so no caller ever sees a raw ``httpx`` exception.
        """
        last_exc: httpx.TransportError | None = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                response = await self._c().request(method, url, **kwargs)
            except httpx.TransportError as exc:
                last_exc = exc
            else:
                if response.status_code < 500 or attempt == _RETRY_ATTEMPTS:
                    return response
            if attempt < _RETRY_ATTEMPTS:
                wait = min(_RETRY_BACKOFF_CAP_S, _RETRY_BACKOFF_BASE_S * 2 ** (attempt - 1))
                await asyncio.sleep(wait)
        assert last_exc is not None  # a 5xx response returns above on the last attempt
        raise StoreError(f"ArcadeDB request to {url} failed: {last_exc}") from last_exc

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------

    async def _server_command(self, command: str) -> dict:
        """Execute a server-level SQL command via /api/v1/server."""
        r = await self._send(
            "POST",
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
        r = await self._send("GET", f"{self._base}/api/v1/exists/{name}")
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
    ) -> list[dict[str, Any]]:
        """Execute a read-write command and return the result list.

        Args:
            db: Target database name.
            command: Cypher or SQL command string.
            language: ``"cypher"`` (default) or ``"sql"``.
            params: Named parameters referenced with ``$name`` in the command.
        """
        body: dict[str, Any] = {"language": language, "command": command}
        if params:
            body["params"] = params
        r = await self._send(
            "POST",
            f"{self._base}/api/v1/command/{db}",
            json=body,
            headers=self._session_headers(),
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
        r = await self._send(
            "POST",
            f"{self._base}/api/v1/query/{db}",
            json=body,
            headers=self._session_headers(),
        )
        _raise_for_status(r)
        return r.json().get("result", [])

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def _session_headers(self) -> dict[str, str]:
        """Header binding this statement to the open transaction, if any."""
        session = _session.get()
        return {SESSION_HEADER: session} if session else {}

    @asynccontextmanager
    async def transaction(self, db: str) -> AsyncIterator[None]:
        """Run the enclosed statements as one ArcadeDB transaction.

        Commits on a clean exit; on any exception the transaction is rolled
        back and the exception re-raised, so a half-written unit of work never
        becomes visible and never passes for a successful one.

        Args:
            db: Target database name.
        """
        r = await self._send("POST", f"{self._base}/api/v1/begin/{db}")
        _raise_for_status(r)
        session = r.headers[SESSION_HEADER]
        token = _session.set(session)
        try:
            yield
        except BaseException:
            # A rollback that fails must not replace the failure that caused it:
            # ArcadeDB answers 500 for a transaction it has already expired, and
            # that error surfacing instead of the real one is how an expired
            # transaction reads as a mysterious commit error (CI, 2026-09-07).
            try:
                await self._end_transaction(db, "rollback", session)
            except Exception:
                logger.warning("rollback of %s failed; re-raising the original error", db)
            raise
        else:
            await self._end_transaction(db, "commit", session)
        finally:
            _session.reset(token)

    async def _end_transaction(self, db: str, verb: str, session: str) -> None:
        """POST ``/api/v1/commit|rollback/{db}`` for an open session."""
        r = await self._send(
            "POST",
            f"{self._base}/api/v1/{verb}/{db}",
            headers={SESSION_HEADER: session},
        )
        _raise_for_status(r)
