"""Unit tests for graphknows.integrations.client — stdlib-only, no live MCP server."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphknows.integrations.client import (
    GraphKnowsMCPClient,
    MCPClientError,
    RemoteMemory,
    normalize_tool_result,
)
from graphknows.exceptions import GraphKnowsError


def test_mcp_client_error_is_graphknows_error() -> None:
    assert issubclass(MCPClientError, GraphKnowsError)


class TestNormalizeToolResult:
    def test_structured_content_preferred(self) -> None:
        assert normalize_tool_result({"structuredContent": {"a": 1}}) == {"a": 1}

    def test_text_content_json_decoded(self) -> None:
        result = {"content": [{"type": "text", "text": '{"chunks": 3}'}]}
        assert normalize_tool_result(result) == {"chunks": 3}

    def test_text_content_non_json_wrapped(self) -> None:
        result = {"content": [{"type": "text", "text": "hello"}]}
        assert normalize_tool_result(result) == {"text": "hello"}

    def test_passthrough_non_dict(self) -> None:
        assert normalize_tool_result([1, 2]) == [1, 2]


class TestRemoteMemory:
    @pytest.mark.asyncio
    async def test_ingest_calls_memory_ingest(self) -> None:
        client = GraphKnowsMCPClient()
        client.call_tool = AsyncMock(return_value={"chunks": 2, "entities": 1})
        mem = RemoteMemory(client=client)
        out = await mem.ingest_memory("hi", session_id="s1", title="t", metadata={"k": "v"})
        assert out == {"chunks": 2, "entities": 1}
        client.call_tool.assert_awaited_once_with(
            "memory_ingest",
            {"text": "hi", "session_id": "s1", "title": "t", "metadata": {"k": "v"}},
        )

    @pytest.mark.asyncio
    async def test_recall_calls_memory_query(self) -> None:
        client = GraphKnowsMCPClient()
        client.call_tool = AsyncMock(return_value={"hits": []})
        mem = RemoteMemory(client=client)
        await mem.recall_memory("q", session_id="s1", top_k=7, scope="ltm")
        client.call_tool.assert_awaited_once_with(
            "memory_query", {"query": "q", "session_id": "s1", "top_k": 7, "scope": "ltm"}
        )

    @pytest.mark.asyncio
    async def test_recall_default_scope_matches_in_process_memory(self) -> None:
        # RemoteMemory must default scope="stm" to match graphknows.Memory.recall_memory.
        client = GraphKnowsMCPClient()
        client.call_tool = AsyncMock(return_value={"hits": []})
        mem = RemoteMemory(client=client)
        await mem.recall_memory("q", session_id="s1")
        assert client.call_tool.await_args.args[1]["scope"] == "stm"

    @pytest.mark.asyncio
    async def test_flush_and_purge(self) -> None:
        client = GraphKnowsMCPClient()
        client.call_tool = AsyncMock(return_value={})
        mem = RemoteMemory(client=client)
        await mem.flush_memory("s1")
        await mem.purge_memory(scope="stm", session_id="s1")
        assert client.call_tool.await_args_list[0].args[0] == "memory_flush"
        assert client.call_tool.await_args_list[1].args[0] == "memory_purge"

    @pytest.mark.asyncio
    async def test_non_dict_result_wrapped(self) -> None:
        client = GraphKnowsMCPClient()
        client.call_tool = AsyncMock(return_value="oops")
        mem = RemoteMemory(client=client)
        assert await mem.flush_memory("s1") == {"result": "oops"}
