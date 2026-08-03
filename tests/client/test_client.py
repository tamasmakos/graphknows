"""Unit tests for graphknows.integrations.client — stdlib-only, no live MCP server."""

from __future__ import annotations

from graphknows.exceptions import GraphKnowsError
from graphknows.integrations.client import MCPClientError, normalize_tool_result


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
