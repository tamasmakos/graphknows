"""GetChunks tool — calls graphrag POST /chunks."""
from __future__ import annotations
import httpx
from agent.tools.base import BaseTool


class GetChunksTool(BaseTool):
    """Get source document chunks that mention a specific entity.

    Attributes:
        name: Tool name "get_chunks".
        description: Tool description for LLM.
    """
    name = "get_chunks"
    description = (
        'Get source document chunks that mention a specific entity. '
        'Params: {"entity_id": "<entity name from retrieve results, e.g. ENERGY_PRICES>", "top_k": <int, default 5>}'
    )

    def __init__(self, graphrag_url: str, client: httpx.AsyncClient) -> None:
        self._base = graphrag_url
        self._client = client

    async def run(self, params: dict) -> dict:
        # Accept both entity_id and entity (LLM sometimes uses wrong field name).
        # Normalise to UPPER_SNAKE_CASE as stored in Neo4j / pgvector.
        raw = params.get("entity_id") or params.get("entity", "")
        entity_id = raw.upper().replace(" ", "_").replace("-", "_")
        top_k = params.get("top_k", 5)
        if not entity_id:
            return {"chunks": [], "error": "entity_id is required"}
        resp = await self._client.post(
            f"{self._base}/chunks",
            json={"entity_id": entity_id, "top_k": top_k},
        )
        resp.raise_for_status()
        return resp.json()
