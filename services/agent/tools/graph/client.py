"""GraphKnowsClient — HTTP client for the graphknows unified memory service.

Single interface for retrieve (-> XML string), community lookup, ingest, and health check.

Two implementations share ``GraphKnowsProtocol``:

* ``GraphKnowsClient`` — HTTP (default; graphknows runs as a separate service).
* ``InProcessGraphKnows`` — in-process (wraps a live ``GraphKnowsContainer``; no HTTP).

Select via ``GRAPHKNOWS_MODE`` env var: ``http`` (default) or ``inprocess``.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import httpx

log = logging.getLogger("agent.tools.graph.client")

_DEFAULT_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Shared protocol — both implementations satisfy this interface
# ---------------------------------------------------------------------------


@runtime_checkable
class GraphKnowsProtocol(Protocol):
    """Structural protocol for graphknows clients.

    Typed duck-typing: any object with these async methods can be injected
    wherever a graphknows client is expected, regardless of transport.
    """

    async def retrieve(
        self,
        query: str,
        session_id: str,
        *,
        k_entity: int = 10,
        k_semantic: int = 20,
        k_final: int = 10,
    ) -> str: ...

    async def ingest(
        self,
        documents: list[dict[str, Any]],
        config_override: dict | None = None,
    ) -> dict[str, Any]: ...

    async def health(self) -> dict[str, Any]: ...


class GraphKnowsClient:
    """Async HTTP client for the graphknows service.

    Args:
        base_url: Base URL of the graphknows service, e.g. http://graphknows:8025.
        timeout: Request timeout in seconds.
    """

    def __init__(self, base_url: str, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def _post(self, path: str, payload: dict) -> Any:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            resp = await client.post(path, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def retrieve(
        self,
        query: str,
        session_id: str,
        *,
        k_entity: int = 10,
        k_semantic: int = 20,
        k_final: int = 10,
    ) -> str:
        result = await self._post(
            "/retrieve",
            {
                "query": query,
                "session_id": session_id,
                "k_entity": k_entity,
                "k_semantic": k_semantic,
                "k_final": k_final,
            },
        )
        return result if isinstance(result, str) else str(result)

    async def community(self, entity_name: str) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            resp = await client.get(f"/community/{entity_name}")
            resp.raise_for_status()
            return resp.json()

    async def ingest(
        self,
        documents: list[dict[str, Any]],
        config_override: dict | None = None,
    ) -> dict[str, Any]:
        """Enqueue an ingestion run. Returns {run_id, status, queued_at}."""
        payload: dict[str, Any] = {"documents": documents}
        if config_override:
            payload["config_override"] = config_override
        return await self._post("/ingest", payload)

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=5.0) as client:
            resp = await client.get("/health")
            resp.raise_for_status()
            return resp.json()


# ---------------------------------------------------------------------------
# Backward-compatibility alias (old code may reference HybridRAGClient)
# ---------------------------------------------------------------------------
HybridRAGClient = GraphKnowsClient


# ---------------------------------------------------------------------------
# In-process implementation — wraps GraphKnowsContainer directly (no HTTP)
# ---------------------------------------------------------------------------


class InProcessGraphKnows:
    """In-process graphknows client backed by a live ``GraphKnowsContainer``.

    Use when the agent runs in the same process as graphknows to eliminate
    HTTP overhead.  Satisfies ``GraphKnowsProtocol``.

    Args:
        container: An entered ``GraphKnowsContainer`` instance.
    """

    def __init__(self, container: Any) -> None:
        self._container = container

    async def retrieve(
        self,
        query: str,
        session_id: str,
        *,
        k_entity: int = 10,
        k_semantic: int = 20,
        k_final: int = 10,
    ) -> str:
        ctx = await self._container.retriever.retrieve(
            query, session_id, k_entity=k_entity, k_semantic=k_semantic, k_final=k_final
        )
        return ctx.to_xml()

    async def ingest(
        self,
        documents: list[dict[str, Any]],
        config_override: dict | None = None,
    ) -> dict[str, Any]:
        from graphknows.ingestion.pipeline import run_ingestion

        stats = await run_ingestion(
            documents=documents,
            config_override=config_override,
        )
        return {"status": "completed", "stats": stats}

    async def health(self) -> dict[str, Any]:
        return {"status": "ok", "mode": "inprocess"}
