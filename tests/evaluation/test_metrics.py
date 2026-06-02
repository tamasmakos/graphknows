"""Unit tests for evaluation metrics."""

from __future__ import annotations

from evaluation.core.metrics import (
    ExpectedEvidence,
    ToolObservation,
    aggregate_observations,
    observation_to_record,
)


def test_aggregate_observations_computes_tool_rows() -> None:
    observations = [
        ToolObservation("case-1", "memory_recall", True, 10.0, {"hits": []}),
        ToolObservation("case-2", "memory_recall", False, 20.0, error="boom"),
    ]

    summary = aggregate_observations(observations)

    assert summary["total_calls"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["error_rate"] == 0.5
    assert summary["tools"][0]["tool_name"] == "memory_recall"
    assert summary["tools"][0]["calls"] == 2


def test_observation_scores_expected_evidence() -> None:
    observation = ToolObservation(
        "case-1",
        "memory_recall",
        True,
        1.0,
        {
            "hits": [
                {
                    "text": "Project Helios stores durable entities in Neo4j.",
                    "entities": ["Project Helios", "Neo4j"],
                }
            ]
        },
        expected=ExpectedEvidence(terms=("helios",), entities=("neo4j",)),
    )

    record = observation_to_record(observation, top_k=5)

    assert record["shape_ok"] is True
    assert record["quality"]["expected_text_found"] is True
    assert record["quality"]["expected_entity_found"] is True
    assert record["quality"]["answer_support_hit"] is True
