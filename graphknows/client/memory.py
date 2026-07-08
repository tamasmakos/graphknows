"""RemoteMemory — the same verbs as :class:`graphknows.Memory`, over MCP.

In-process callers use ``graphknows.Memory``; out-of-process callers (a
different runtime, a sandboxed agent, a service that shouldn't load the ML
stack) use ``RemoteMemory`` to drive a ``graphknows-mcp`` subprocess with the
same method names — one mental model, two transports.
"""

from __future__ import annotations

from typing import Any

from graphknows.client.mcp import GraphKnowsMCPClient


class RemoteMemory:
    """Memory operations over the MCP stdio transport.

    Args:
        client: An existing :class:`GraphKnowsMCPClient`. When omitted one is
            created from ``command`` and owned by this instance (started/closed
            with the async context manager).
        command: The MCP server command (used only when ``client`` is omitted).
    """

    def __init__(
        self,
        client: GraphKnowsMCPClient | None = None,
        *,
        command: str = "graphknows-mcp",
    ) -> None:
        self._owns_client = client is None
        self._client = client or GraphKnowsMCPClient(command=command)

    @property
    def client(self) -> GraphKnowsMCPClient:
        """The underlying MCP client."""
        return self._client

    async def __aenter__(self) -> RemoteMemory:
        if self._owns_client:
            await self._client.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._owns_client:
            await self._client.close()

    async def ingest_memory(
        self,
        text: str,
        *,
        session_id: str,
        title: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ingest text into short-term memory (MCP ``memory_ingest``)."""
        args: dict[str, Any] = {"text": text, "session_id": session_id}
        if title:
            args["title"] = title
        if metadata:
            args["metadata"] = metadata
        return await self._call("memory_ingest", args)

    async def recall_memory(
        self,
        query: str,
        *,
        session_id: str = "",
        top_k: int = 5,
        scope: str = "stm",
    ) -> dict[str, Any]:
        """Recall ranked hits (MCP ``memory_query``).

        The default ``scope`` matches the in-process :class:`graphknows.Memory`
        so the two transports return the same tier for the same call.
        """
        return await self._call(
            "memory_query",
            {"query": query, "session_id": session_id, "top_k": top_k, "scope": scope},
        )

    async def flush_memory(self, session_id: str) -> dict[str, Any]:
        """Promote a session from STM to LTM (MCP ``memory_flush``)."""
        return await self._call("memory_flush", {"session_id": session_id})

    async def purge_memory(self, scope: str = "both", session_id: str = "") -> dict[str, Any]:
        """Purge STM and/or LTM (MCP ``memory_purge``)."""
        return await self._call("memory_purge", {"scope": scope, "session_id": session_id})

    async def _call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._client.call_tool(tool, arguments)
        return result if isinstance(result, dict) else {"result": result}
