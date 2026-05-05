"""GetPath tool — calls graphrag POST /path."""
from __future__ import annotations
import httpx
from agent.tools.base import BaseTool


class GetPathTool(BaseTool):
    """Find shortest path between two entities in the knowledge graph.

    Attributes:
        name: Tool name "get_path".
        description: Tool description for LLM.
    """
    name = "get_path"
    description = (
        'Find shortest path between two entities in the knowledge graph. '
        'Params: {"entity_a": "<string>", "entity_b": "<string>", "max_hops": <int, default 3>}'
    )

    def __init__(self, graphrag_url: str, client: httpx.AsyncClient) -> None:
        self._base = graphrag_url
        self._client = client

    async def run(self, params: dict) -> dict:
        # Normalise field names — LLM may send entity1/entity2, entity_a/entity_b,
        # entity1_id/entity2_id, or any creative variant.
        entity_a = (
            params.get("entity_a")
            or params.get("entity1")
            or params.get("entity1_id")
            or params.get("entity_1")
            or ""
        )
        entity_b = (
            params.get("entity_b")
            or params.get("entity2")
            or params.get("entity2_id")
            or params.get("entity_2")
            or ""
        )
        max_hops = params.get("max_hops", 3)
        resp = await self._client.post(
            f"{self._base}/path",
            json={"entity_a": entity_a, "entity_b": entity_b, "max_hops": max_hops},
        )
        resp.raise_for_status()
        return resp.json()
