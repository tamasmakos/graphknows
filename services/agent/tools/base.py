"""BaseTool ABC for all agent tools."""
from __future__ import annotations
from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Abstract base class for all agent tools.

    Attributes:
        name: Unique tool name used by the orchestrator for tool selection.
        description: Human-readable description passed to the LLM for tool selection.
    """
    name: str
    description: str

    @abstractmethod
    async def run(self, params: dict) -> dict:
        """Execute the tool with the given parameters.

        Args:
            params: Tool-specific parameters dict from the LLM.

        Returns:
            Result dict to be observed by the orchestrator.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
