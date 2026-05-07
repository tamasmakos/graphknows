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
    description = (
        'Get the Leiden community (topic cluster) an entity belongs to. '
        'Params: {"entity_id": "<entity name from retrieve results, e.g. ENERGY_PRICES>"}'
    )

    def __init__(self, graphrag_url: str, client: httpx.AsyncClient) -> None:
        self._base = graphrag_url
        self._client = client

    async def run(self, params: dict) -> dict:
        # Accept both entity_id and entity (LLM sometimes uses wrong field name).
        # Normalise to UPPER_SNAKE_CASE as stored in Neo4j / pgvector.
        raw = params.get("entity_id") or params.get("entity", "")
        entity_id = raw.upper().replace(" ", "_").replace("-", "_")
        if not entity_id:
            return {"community_id": None, "label": None, "members": [], "error": "entity_id is required"}
        resp = await self._client.get(f"{self._base}/community/{entity_id}")
        resp.raise_for_status()
        return resp.json()
