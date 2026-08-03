"""graphknows.integrations.client — official out-of-process client for the MCP server.

Stdlib-only (no ML/extraction dependencies), so it installs with the core
package. Drive ``graphknows-mcp`` from any framework::

    from graphknows.integrations.client import GraphKnowsMCPClient

    async with GraphKnowsMCPClient(command="graphknows-mcp") as client:
        await client.call_tool("memory_ingest", {"text": "some text", "session_id": "s1"})
        hits = await client.call_tool("memory_query", {"query": "my query", "session_id": "s1"})

:class:`GraphKnowsMCPClient` is the JSON-RPC transport and the whole surface. A
``RemoteMemory`` wrapper mirroring :class:`graphknows.Memory`'s verbs used to sit
on top of it; it had no caller outside its own test, so the second mental model
was cost without a user.
"""

from __future__ import annotations

from graphknows.integrations.client.mcp import (
    GraphKnowsMCPClient,
    MCPClientError,
    normalize_tool_result,
)

__all__ = [
    "GraphKnowsMCPClient",
    "MCPClientError",
    "normalize_tool_result",
]
