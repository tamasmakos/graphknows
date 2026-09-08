"""ArcadeDBClient._send retries what AsyncHTTPTransport(retries=3) does not (FR-034).

The transport's own ``retries`` only cover a failed *connection attempt*. An
HTTP 5xx response, a mid-request timeout (the connection was already open —
ArcadeDB was mid-read/write when it dropped) and ArcadeDB lock contention
(reported as a 500, ArcadeDB having no dedicated status for it) all sailed
through unretried before this fix. Each test drives the real client against a
stub ``httpx.MockTransport`` handler in place of the network, so no ArcadeDB
server is needed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx
import pytest

from graphknows.exceptions import StoreError
from graphknows.storage.arcadedb.client import SESSION_HEADER, ArcadeDBClient

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def _client_with(handler: Handler) -> ArcadeDBClient:
    """A connected ArcadeDBClient wired to *handler* instead of the network."""
    client = ArcadeDBClient("http://arcadedb.test")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


async def test_retries_a_5xx_response_then_succeeds() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(500, json={"detail": "lock contention"})
        return httpx.Response(200, json={"result": True})

    assert await _client_with(handler).database_exists("db") is True
    assert calls == 3


async def test_gives_up_after_exhausting_the_5xx_retry_budget() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, json={"detail": "still locked"})

    with pytest.raises(StoreError, match="still locked"):
        await _client_with(handler)._server_command("create database x")
    assert calls == 3


async def test_retries_a_mid_request_timeout_then_succeeds() -> None:
    """A timeout after the connection was already open, not a connect failure."""
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ReadTimeout("timed out mid-response", request=request)
        return httpx.Response(200, json={"result": True})

    assert await _client_with(handler).database_exists("db") is True
    assert calls == 3


async def test_gives_up_after_exhausting_the_timeout_retry_budget() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out mid-response", request=request)

    with pytest.raises(StoreError, match="timed out mid-response"):
        await _client_with(handler)._server_command("create database x")
    assert calls == 3


async def test_a_4xx_response_is_not_retried() -> None:
    """An application error (a bad command) fails fast — retrying buys nothing."""
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"detail": "bad request"})

    with pytest.raises(StoreError, match="bad request"):
        await _client_with(handler)._server_command("nonsense")
    assert calls == 1


async def test_transaction_begin_retries_lock_contention_then_commits() -> None:
    """FR-034 names lock contention explicitly; exercised through transaction()."""
    begin_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal begin_calls
        if request.url.path == "/api/v1/begin/mem":
            begin_calls += 1
            if begin_calls == 1:
                return httpx.Response(500, json={"exception": "ConcurrentModificationException"})
            return httpx.Response(200, headers={SESSION_HEADER: "sess-1"})
        return httpx.Response(200, json={"result": []})  # commit

    async with _client_with(handler).transaction("mem"):
        pass
    assert begin_calls == 2
