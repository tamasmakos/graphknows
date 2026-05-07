"""Session and streaming event models for the agent service."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ReasoningStep(BaseModel):
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
    elapsed_ms: Optional[float] = None


class GraphDelta(BaseModel):
    """Incremental graph update pushed to the frontend after each tool step.

    Attributes:
        added_nodes: New entity/topic node dicts in D3.js format.
        added_edges: New edge dicts.
        removed_node_ids: Entity ids to remove (session pruning).
    """

    added_nodes: list[dict[str, Any]] = Field(default_factory=list)
    added_edges: list[dict[str, Any]] = Field(default_factory=list)
    removed_node_ids: list[str] = Field(default_factory=list)


class GraphSnapshot(BaseModel):
    """Full graph snapshot for GET /sessions/{id}/graph endpoint.

    Attributes:
        nodes: All entity/topic nodes in D3.js format.
        edges: All edges in D3.js format.
        session_id: Session this snapshot belongs to.
    """

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str


class StreamEvent(BaseModel):
    """Server-sent event frame from the IterativeOrchestrator.

    Attributes:
        event: Event type discriminator.
        data: Event payload.
    """

    event: Literal["step_start", "step_result", "answer", "error"]
    data: dict[str, Any] = Field(default_factory=dict)
