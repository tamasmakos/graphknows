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
    description = "Find shortest path between two entities in the knowledge graph."

    def __init__(self, graphrag_url: str, client: httpx.AsyncClient) -> None:
        self._base = graphrag_url
        self._client = client

    async def run(self, params: dict) -> dict:
        resp = await self._client.post(f"{self._base}/path", json=params)
        resp.raise_for_status()
        return resp.json()
