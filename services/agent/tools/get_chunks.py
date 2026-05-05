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
    description = "Get source document chunks that mention a specific entity."

    def __init__(self, graphrag_url: str, client: httpx.AsyncClient) -> None:
        self._base = graphrag_url
        self._client = client

    async def run(self, params: dict) -> dict:
        resp = await self._client.post(f"{self._base}/chunks", json=params)
        resp.raise_for_status()
        return resp.json()
