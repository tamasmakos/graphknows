"""HybridRAGClient — HTTP client for the hybridrag retrieval service.

Single interface for retrieve (-> XML string), community lookup, and health check.
Ingest is handled by graphgen, not hybridrag.
Replaces the old GraphRAGClient.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger("agent.tools.graph.client")

_DEFAULT_TIMEOUT = 30.0


class HybridRAGClient:
    """Async HTTP client for the hybridrag service.

    Args:
        base_url: Base URL of the hybridrag service, e.g. http://hybridrag:8020.
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

    async def community(self, entity_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            resp = await client.get(f"/community/{entity_id}")
            resp.raise_for_status()
            return resp.json()

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=5.0) as client:
            resp = await client.get("/health")
            resp.raise_for_status()
            return resp.json()
