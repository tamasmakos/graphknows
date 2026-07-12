"""Retrieval stage: query memory and parse hits into typed passages."""

from __future__ import annotations

from typing import Any

from evaluation.common.config import BaseEvalConfig
from evaluation.common.datamodels import RetrievedPassage
from graphknows.client import GraphKnowsMCPClient


def _parse_hits(result: Any) -> list[RetrievedPassage]:
    """Extract ``RetrievedPassage`` objects from a ``memory_query`` response."""
    hits = result.get("hits", []) if isinstance(result, dict) else []
    passages: list[RetrievedPassage] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        passages.append(
            RetrievedPassage(
                text=str(hit.get("text", "")),
                score=hit.get("score"),
                sources=str(hit.get("sources", "")),
            )
        )
    return passages


class Retriever:
    """Queries one memory session via the ``memory_query`` MCP tool."""

    def __init__(self, client: GraphKnowsMCPClient, config: BaseEvalConfig) -> None:
        self._client = client
        self._config = config

    async def query(
        self, session_id: str, question: str, *, k: int, namespace: str = ""
    ) -> list[RetrievedPassage]:
        """Return the top-``k`` passages for ``question`` in ``session_id``."""
        result = await self._client.call_tool(
            "memory_query",
            {
                "query": question,
                "session_id": session_id,
                "top_k": k,
                "scope": self._config.recall_scope,
                "namespace": namespace,
            },
        )
        return _parse_hits(result)
