"""graphknows.client — official out-of-process client for the MCP server.

Stdlib-only (no ML/extraction dependencies), so it installs with the core
package. Drive ``graphknows-mcp`` from any framework::

    from graphknows.client import RemoteMemory

    async with RemoteMemory(command="graphknows-mcp") as mem:
        await mem.ingest_memory("some text", session_id="s1")
        hits = await mem.recall_memory("my query", session_id="s1")

``RemoteMemory`` mirrors the in-process :class:`graphknows.Memory` verbs;
:class:`GraphKnowsMCPClient` is the lower-level JSON-RPC transport.
"""

from __future__ import annotations

from graphknows.client.mcp import GraphKnowsMCPClient, MCPClientError, normalize_tool_result
from graphknows.client.memory import RemoteMemory

__all__ = [
    "GraphKnowsMCPClient",
    "MCPClientError",
    "RemoteMemory",
    "normalize_tool_result",
]
