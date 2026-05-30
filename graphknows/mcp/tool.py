"""Framework-agnostic tool primitive for graphknows.

A :class:`GraphKnowsTool` is a plain async callable bundled with the metadata
that every agent framework needs: a name, a description, and a JSON Schema for
its parameters.  No framework is imported here — the caller wraps tools for
their own stack in one line::

    # LangChain
    StructuredTool.from_function(t.fn, name=t.name, description=t.description)

    # OpenAI Agents SDK
    FunctionTool(t.fn, name=t.name, description=t.description)

    # PydanticAI
    Tool(t.fn, name=t.name, description=t.description)

    # MCP (out-of-process) — use container.as_mcp_server() instead.

Example:
    from graphknows.mcp.tool import GraphKnowsTool
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphKnowsTool:
    """A named, documented, async-callable tool with a JSON Schema for its inputs.

    Attributes:
        name: Unique tool identifier (snake_case, matches MCP tool name).
        description: Human-readable description consumed by LLM tool-pickers.
        parameters: JSON Schema object describing the function's parameters.
            Compatible with OpenAI function-calling, MCP, and most agent SDKs.
        fn: The underlying async callable.  Receives keyword arguments matching
            the ``parameters`` schema.
        tags: Optional routing / categorisation tags for frameworks that support
            them (e.g. LangGraph).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Awaitable[Any]]
    tags: list[str] = field(default_factory=list)

    async def __call__(self, **kwargs: Any) -> Any:
        """Invoke the tool with keyword arguments."""
        return await self.fn(**kwargs)

    def as_openai_schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function schema dict."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
