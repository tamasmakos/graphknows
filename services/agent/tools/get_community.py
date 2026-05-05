"""GetCommunity tool — calls graphrag GET /community/{entity_id}."""
from __future__ import annotations
import httpx
from agent.tools.base import BaseTool


class GetCommunityTool(BaseTool):
    """Get the Leiden community (topic cluster) an entity belongs to.

    Attributes:
        name: Tool name "get_community".
        description: Tool description for LLM.
    """
    name = "get_community"
    description = "Get the Leiden community (topic cluster) an entity belongs to."

    def __init__(self, graphrag_url: str, client: httpx.AsyncClient) -> None:
        self._base = graphrag_url
        self._client = client

    async def run(self, params: dict) -> dict:
        entity_id = params.get("entity_id", "")
        resp = await self._client.get(f"{self._base}/community/{entity_id}")
        resp.raise_for_status()
        return resp.json()
