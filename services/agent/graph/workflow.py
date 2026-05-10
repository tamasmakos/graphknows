"""Prototype workflow spec for the evidence-led investigative agent redesign."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from services.agent.graph.state import MacroPhase


NodeKind = Literal[
    "planner",
    "retrieval",
    "graph",
    "verification",
    "synthesis",
    "writeback",
]


@dataclass(frozen=True)
class NodeSpec:
    """Specification for a planned workflow node.

    Attributes:
        name: Stable node identifier.
        phase: Macro-phase that owns the node.
        kind: Node family used for implementation and observability.
        purpose: Human-readable description of node responsibility.
        reads: State fields consumed by the node.
        writes: State fields produced or updated by the node.
        routes_to: Downstream nodes reachable from this node.
    """

    name: str
    phase: MacroPhase
    kind: NodeKind
    purpose: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    routes_to: tuple[str, ...] = ()


REQUIRED_STREAM_EVENTS: tuple[str, ...] = (
    "phase_change",
    "thread_started",
    "evidence_added",
    "graph_building",
    "graph_ready",
    "claim_verified",
    "contradiction_detected",
    "report_ready",
)


def build_workflow_spec() -> tuple[NodeSpec, ...]:
    """Return the node-level prototype for the redesigned workflow."""

    return (
        NodeSpec(
            name="scope_frame",
            phase="scope",
            kind="planner",
            purpose="Frame the hypothesis, unit of analysis, and scope boundaries.",
            reads=("messages", "user_query"),
            writes=("hypothesis", "unit_of_analysis", "current_macro_phase"),
            routes_to=("scope_map",),
        ),
        NodeSpec(
            name="scope_map",
            phase="scope",
            kind="planner",
            purpose="Map actors, institutions, and prior corpus context.",
            reads=("hypothesis", "unit_of_analysis", "hypothesis_graph"),
            writes=("key_actors", "hypothesis_graph"),
            routes_to=("scope_plan",),
        ),
        NodeSpec(
            name="scope_plan",
            phase="scope",
            kind="planner",
            purpose="Produce sub-questions, target sources, and evidence threads.",
            reads=("hypothesis", "key_actors"),
            writes=("evidence_plan", "evidence_threads", "current_macro_phase"),
            routes_to=("investigate_dispatch_threads",),
        ),
        NodeSpec(
            name="investigate_dispatch_threads",
            phase="investigate",
            kind="retrieval",
            purpose="Start parallel evidence collection lanes by sub-question.",
            reads=("evidence_plan", "evidence_threads"),
            writes=("evidence_threads",),
            routes_to=("investigate_collect_thread_evidence",),
        ),
        NodeSpec(
            name="investigate_collect_thread_evidence",
            phase="investigate",
            kind="retrieval",
            purpose="Collect evidence and append normalized ledger entries.",
            reads=("evidence_threads", "evidence_plan"),
            writes=("evidence_threads", "evidence_ledger"),
            routes_to=(
                "investigate_collect_thread_evidence",
                "investigate_trigger_graph_build",
            ),
        ),
        NodeSpec(
            name="investigate_trigger_graph_build",
            phase="investigate",
            kind="graph",
            purpose="Build the session hypothesis graph after evidence coverage is sufficient.",
            reads=("evidence_ledger", "hypothesis_graph"),
            writes=("hypothesis_graph",),
            routes_to=("investigate_extract_claims",),
        ),
        NodeSpec(
            name="investigate_extract_claims",
            phase="investigate",
            kind="verification",
            purpose="Extract atomic claims from the gathered evidence and draft findings.",
            reads=("evidence_ledger", "hypothesis_graph"),
            writes=("claims",),
            routes_to=("investigate_verify_claims",),
        ),
        NodeSpec(
            name="investigate_verify_claims",
            phase="investigate",
            kind="verification",
            purpose="Verify each claim with hybrid graph paths and supporting passages.",
            reads=("claims", "evidence_ledger", "hypothesis_graph"),
            writes=("claims",),
            routes_to=("investigate_surface_contradictions",),
        ),
        NodeSpec(
            name="investigate_surface_contradictions",
            phase="investigate",
            kind="verification",
            purpose="Promote unresolved conflicts into explicit contradiction records.",
            reads=("claims", "evidence_ledger"),
            writes=("contradictions", "current_macro_phase"),
            routes_to=("communicate_report", "investigate_collect_thread_evidence"),
        ),
        NodeSpec(
            name="communicate_report",
            phase="communicate",
            kind="synthesis",
            purpose="Write the report from verified claims and explicit contradictions.",
            reads=("claims", "contradictions", "evidence_ledger", "hypothesis_graph"),
            writes=("findings", "report"),
            routes_to=("communicate_writeback",),
        ),
        NodeSpec(
            name="communicate_writeback",
            phase="communicate",
            kind="writeback",
            purpose="Persist validated outputs while keeping the hypothesis graph bounded.",
            reads=("report", "claims", "hypothesis_graph"),
            writes=("hypothesis_graph",),
        ),
    )


def node_names_by_phase() -> dict[MacroPhase, tuple[str, ...]]:
    """Return workflow nodes grouped by macro-phase."""

    grouped: dict[MacroPhase, list[str]] = {
        "scope": [],
        "investigate": [],
        "communicate": [],
    }
    for node in build_workflow_spec():
        grouped[node.phase].append(node.name)
    return {phase: tuple(names) for phase, names in grouped.items()}
