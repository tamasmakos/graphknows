"""Build tool — calls graphgen POST /build."""
from __future__ import annotations
import httpx
from agent.tools.base import BaseTool


class BuildTool(BaseTool):
    """Trigger a knowledge graph build pipeline for given documents.

    Attributes:
        name: Tool name "build".
        description: Tool description for LLM.
    """
    name = "build"
    description = "Trigger a knowledge graph build pipeline for given documents."

    def __init__(self, graphgen_url: str, client: httpx.AsyncClient) -> None:
        self._base = graphgen_url
        self._client = client

    async def run(self, params: dict) -> dict:
        resp = await self._client.post(f"{self._base}/build", json=params)
        resp.raise_for_status()
        return resp.json()
