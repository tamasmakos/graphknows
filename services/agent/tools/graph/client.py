"""GraphKnowsClient — HTTP client for the graphknows unified memory service.

Single interface for retrieve (-> XML string), community lookup, ingest, and health check.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger("agent.tools.graph.client")

_DEFAULT_TIMEOUT = 30.0


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
