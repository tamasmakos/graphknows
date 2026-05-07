"""Search tool — stub for search service integration."""
from __future__ import annotations
from agent.tools.base import BaseTool


class SearchTool(BaseTool):
    """Search for documents from EU parliamentary sources.

    Attributes:
        name: Tool name "search".
        description: Tool description for LLM.
    """
    name = "search"
    description = "Search for documents from EU parliamentary sources."

    async def run(self, params: dict) -> dict:
        raise NotImplementedError("TODO: search service integration")
