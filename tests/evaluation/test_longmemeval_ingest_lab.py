"""Tests for the LongMemEval research ingest lab helpers."""

from __future__ import annotations

from evaluation.benchmarks.longmemeval import (
    CapsuleOptions,
    build_trajectory_capsule,
    capsule_documents,
    select_states,
)


def _trajectory() -> dict:
    return {
        "id": "t-1",
        "domain": "shopping",
        "environment": "browser",
        "goal": "Remember that Express shipping was saved.",
        "outcome": "Checkout later selected Standard by mistake.",
        "states": [
            {"step": 0, "observation": "Opened account settings."},
            {"step": 1, "observation": "Saved shipping preference Express."},
            {"step": 2, "observation": "Viewed unrelated recommendations."},
            {"step": 3, "observation": "Checkout showed Standard shipping."},
        ],
    }


def test_select_states_keeps_first_last_and_query_overlap() -> None:
    states = select_states(_trajectory(), "Which shipping preference was saved?", max_states=3)
    steps = [state["step"] for state in states]
    assert 0 in steps
    assert 3 in steps
    assert 1 in steps


def test_capsule_is_deterministic_and_budget_bound() -> None:
    options = CapsuleOptions(max_states=3, max_chars_per_state=80, max_chars_per_capsule=220)
    first = build_trajectory_capsule(_trajectory(), "shipping preference", options)
    second = build_trajectory_capsule(_trajectory(), "shipping preference", options)
    assert first == second
    assert len(first) <= 220
    assert "Trajectory id: t-1" in first
    assert "Express" in first


def test_capsule_documents_preserve_ids() -> None:
    docs = capsule_documents([_trajectory()], "shipping preference", CapsuleOptions())
    assert docs[0]["id"] == "t-1"
    assert docs[0]["source"] == "longmemeval-v2"
