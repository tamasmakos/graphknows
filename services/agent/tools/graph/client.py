"""GraphRAGClient — single HTTP client for the graphrag MCP/REST service.

Replaces the old individual thin wrappers (retrieve.py, get_path.py,
get_chunks.py, get_community.py) with a unified, typed interface.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger("agent.tools.graph.client")

_DEFAULT_TIMEOUT = 30.0


class GraphRAGClient:
    """Async HTTP client for the graphrag service.

    All methods are coroutine-safe and share a single httpx.AsyncClient
    when used as an async context manager, or create short-lived clients
    otherwise.

    Args:
        base_url: Base URL of the graphrag service, e.g. ``http://graphrag:8010``.
        timeout: Request timeout in seconds.
    """

    def __init__(self, base_url: str, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "GraphRAGClient":
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        # Ephemeral client — caller should prefer context manager usage
        return httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)

    async def _post(self, path: str, payload: dict) -> dict:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            resp = await client.post(path, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def retrieve(
        self,
        query: str,
        session_id: str,
        *,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Hybrid retrieval: dense vector + graph 2-hop + RRF fusion.

        Args:
            query: Natural-language query string.
            session_id: Active session for scoped retrieval.
            top_k: Number of results to return.

        Returns:
            Dict with keys ``chunks``, ``graph_paths``, ``entities``.
        """
        return await self._post("/retrieve", {
            "query": query,
            "session_id": session_id,
            "top_k": top_k,
        })

    async def get_path(
        self,
        entity_a: str,
        entity_b: str,
        *,
        max_hops: int = 3,
    ) -> dict[str, Any]:
        """Shortest graph path between two entities in Neo4j.

        Args:
            entity_a: Source entity name or ID.
            entity_b: Target entity name or ID.
            max_hops: Maximum relationship hops.

        Returns:
            Dict with ``path`` list and ``length``.
        """
        return await self._post("/path", {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "max_hops": max_hops,
        })

    async def get_chunks(
        self,
        entity: str,
        session_id: str,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve text chunks associated with an entity.

        Args:
            entity: Entity name or ID.
            session_id: Session namespace for pgvector lookup.
            top_k: Maximum number of chunks to return.

        Returns:
            Dict with ``chunks`` list.
        """
        return await self._post("/chunks", {
            "entity": entity,
            "session_id": session_id,
            "top_k": top_k,
        })

    async def get_community(
        self,
        entity: str,
        *,
        depth: int = 2,
    ) -> dict[str, Any]:
        """Retrieve the graph community (neighbourhood) around an entity.

        Args:
            entity: Entity name or ID.
            depth: Neighbourhood depth (hops).

        Returns:
            Dict with ``nodes`` and ``edges``.
        """
        return await self._post("/community", {
            "entity": entity,
            "depth": depth,
        })

    async def ingest_chunks(
        self,
        chunks: list[dict],
        session_id: str,
    ) -> dict[str, Any]:
        """Push text chunks into pgvector for a given session.

        Args:
            chunks: List of chunk dicts with at least ``text`` and ``id`` keys.
            session_id: Session namespace.

        Returns:
            Dict with ``ingested`` count.
        """
        return await self._post("/ingest", {
            "chunks": chunks,
            "session_id": session_id,
        })

    async def health(self) -> dict[str, Any]:
        """Check graphrag service health."""
        async with httpx.AsyncClient(base_url=self._base_url, timeout=5.0) as client:
            resp = await client.get("/health")
            resp.raise_for_status()
            return resp.json()
