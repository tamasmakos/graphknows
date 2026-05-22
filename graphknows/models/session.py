"""Session and streaming event models for the agent ReAct loop.

Example:
    from graphknows.models.session import ReasoningStep, GraphDelta, StreamEvent
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from graphknows.models.base import GraphKnowsBaseModel, ImmutableModel


class ReasoningStep(ImmutableModel):
    """A single step in the agent ReAct loop.

    Attributes:
        step_id: Monotonic step counter within the session query.
        tool: Tool name called at this step.
        input: Tool input parameters.
        output: Tool output summary or full result.
        elapsed_ms: Elapsed milliseconds for this step.
    """

    step_id: int
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: float | None = None


class GraphDelta(ImmutableModel):
    """Incremental graph update pushed to the frontend after each tool step.

    Attributes:
        added_nodes: New entity/topic node dicts in D3.js format.
        added_edges: New edge dicts.
        removed_node_ids: Entity ids to remove (session pruning).
    """

    added_nodes: list[dict[str, Any]] = Field(default_factory=list)
    added_edges: list[dict[str, Any]] = Field(default_factory=list)
    removed_node_ids: list[str] = Field(default_factory=list)


class GraphSnapshot(ImmutableModel):
    """Full graph snapshot for a session graph endpoint.

    Attributes:
        nodes: All entity/topic nodes in D3.js format.
        edges: All edges in D3.js format.
        session_id: Session this snapshot belongs to.
    """

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str


class StreamEvent(GraphKnowsBaseModel):
    """Server-sent event frame from the agent orchestrator.

    Attributes:
        event: Event type discriminator.
        data: Event payload dict.
    """

    event: Literal["step_start", "step_result", "answer", "error"]
    data: dict[str, Any] = Field(default_factory=dict)
