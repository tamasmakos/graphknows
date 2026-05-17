"""BaseTool ABC and ToolRegistry for all agent tools."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for all agent tools.

    Attributes:
        name: Unique tool name used by the orchestrator for tool selection.
        description: Human-readable description passed to the LLM for tool selection.
        category: Domain grouping (e.g. "search", "graph", "extraction").
    """
    name: str
    description: str
    category: str = "general"

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

    def as_langchain_tool(self) -> Any:
        """Return a LangChain-compatible StructuredTool wrapping this tool.

        Returns:
            langchain_core.tools.StructuredTool instance.
        """
        from langchain_core.tools import StructuredTool

        async def _run(**kwargs: Any) -> dict:
            return await self.run(kwargs)

        return StructuredTool.from_function(
            coroutine=_run,
            name=self.name,
            description=self.description,
        )


class ToolRegistry:
    """Central registry for agent tools.

    Usage::

        registry = ToolRegistry()

        @registry.register
        class MyTool(BaseTool):
            name = "my_tool"
            ...

        tools = registry.build(categories=["search"])
    """

    def __init__(self) -> None:
        self._tools: dict[str, type[BaseTool]] = {}

    def register(self, cls: type[BaseTool]) -> type[BaseTool]:
        """Register a tool class. Use as a class decorator."""
        self._tools[cls.name] = cls
        return cls

    def build(
        self,
        *,
        categories: list[str] | None = None,
        names: list[str] | None = None,
        **init_kwargs: Any,
    ) -> list[BaseTool]:
        """Instantiate tools matching the given filters.

        Args:
            categories: If provided, only tools with matching ``category``.
            names: If provided, only tools with matching ``name``.
            **init_kwargs: Passed to each tool's ``__init__``.

        Returns:
            List of instantiated tool objects.
        """
        result: list[BaseTool] = []
        for tool_cls in self._tools.values():
            if categories and tool_cls.category not in categories:
                continue
            if names and tool_cls.name not in names:
                continue
            result.append(tool_cls(**init_kwargs))
        return result

    def list_tools(self, category: str | None = None) -> list[str]:
        """Return tool names, optionally filtered by category."""
        return [
            name for name, cls in self._tools.items()
            if category is None or cls.category == category
        ]


# Global singleton registry
registry = ToolRegistry()
