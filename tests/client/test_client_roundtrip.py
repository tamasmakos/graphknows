"""Round-trip tests for GraphKnowsMCPClient over a real stdio subprocess.

Spawns ``fake_mcp_server.py`` — a stdlib-only stub, not the real
``graphknows-mcp`` — so ``integrations.client`` proves itself against real
subprocess framing (newline-delimited JSON on stdin/stdout) and real
JSON-RPC request/response correlation without pulling the ML/extraction
stack into this test's imports (FR-023, SC-015).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from graphknows.integrations.client import GraphKnowsMCPClient, MCPClientError

FAKE_SERVER = [sys.executable, str(Path(__file__).parent / "fake_mcp_server.py")]


async def test_initialize_list_tools_and_call_round_trip() -> None:
    async with GraphKnowsMCPClient(command=FAKE_SERVER) as client:
        tools = await client.list_tools()
        assert tools == [{"name": "echo", "description": "echoes arguments"}]

        result = await client.call_tool("echo", {"value": 42})
        assert result == {"echoed": {"value": 42}}


async def test_server_error_raises_mcp_client_error() -> None:
    async with GraphKnowsMCPClient(command=FAKE_SERVER) as client:
        with pytest.raises(MCPClientError):
            await client.call_tool("boom")


async def test_concurrent_requests_correlate_by_id_not_arrival_order() -> None:
    """The server answers the second call first; the client must still hand
    each caller its own result, proving correlation runs on JSON-RPC id
    (``request()``'s ``_pending`` map) rather than send/arrival order."""
    async with GraphKnowsMCPClient(command=FAKE_SERVER) as client:
        slow, fast = await asyncio.gather(
            client.call_tool("echo", {"which": "slow", "delay_s": 0.2}),
            client.call_tool("echo", {"which": "fast"}),
        )
    assert slow == {"echoed": {"which": "slow", "delay_s": 0.2}}
    assert fast == {"echoed": {"which": "fast"}}
