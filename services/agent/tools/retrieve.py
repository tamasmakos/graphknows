"""Retrieve tool — calls graphrag POST /retrieve."""
from __future__ import annotations
import httpx
from agent.tools.base import BaseTool


class RetrieveTool(BaseTool):
    """Hybrid retrieval over Neo4j + pgvector via the graphrag service.

    Attributes:
        name: Tool name "retrieve".
        description: Tool description for LLM.
    """
    name = "retrieve"
    description = "Search the knowledge graph for chunks and entities relevant to a query."

    def __init__(self, graphrag_url: str, client: httpx.AsyncClient) -> None:
        self._base = graphrag_url
        self._client = client

    async def run(self, params: dict) -> dict:
        resp = await self._client.post(f"{self._base}/retrieve", json=params)
        resp.raise_for_status()
        return resp.json()
