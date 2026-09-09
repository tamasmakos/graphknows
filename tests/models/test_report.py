"""Result objects: counters that reach the caller, and no silent emptiness."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphknows.models import (
    Counters,
    Evidence,
    Fact,
    FactWithEvidence,
    IngestReport,
    RecallBudget,
    RecallResult,
)

R9_COUNTERS = {
    "sources_deduplicated",
    "files_zero_segments",
    "records_skipped_unknown_type",
    "blocks_skipped_unknown_type",
    "records_skipped_malformed",
    "records_skipped_duplicate",
    "time_anchor_inferred",
    "candidates_truncated",
    "merges_committed",
    "candidate_edges_written",
    "merges_vetoed",
    "ingest_queue_wait_ms",
    "symbols_resolved",
    "symbols_unresolved",
    "facts_returned",
    "facts_truncated_by_budget",
    "facts_excluded_tombstoned",
    "pool_below_floor",
}


def _fact_with_evidence() -> FactWithEvidence:
    return FactWithEvidence(
        fact=Fact(id="f1", subject="e1", predicate="code:calls", object="e2"),
        evidence=[Evidence(source_uri="file://a.py", byte_range=(0, 12), text="a()")],
    )


def test_every_r9_counter_is_named_and_starts_at_zero() -> None:
    counters = Counters()
    assert set(Counters.model_fields) >= R9_COUNTERS
    assert all(getattr(counters, name) == 0 for name in R9_COUNTERS)


def test_ingest_report_carries_counters_by_default() -> None:
    report = IngestReport(source_id="src-1", segments_written=3, facts_written=2)
    assert report.entities_touched == 0
    assert report.counters.records_skipped_malformed == 0


def test_recall_result_states_its_budget_and_truncation() -> None:
    result = RecallResult(
        facts=[_fact_with_evidence()],
        budget=RecallBudget(max_facts=1),
        truncated_by="rrf score",
    )
    assert result.no_evidence is False
    assert result.budget.max_facts == 1
    assert result.truncated_by == "rrf score"


def test_empty_recall_must_say_nothing_is_known() -> None:
    with pytest.raises(ValidationError, match="no_evidence"):
        RecallResult()
    assert RecallResult(no_evidence=True).facts == []


def test_fact_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FactWithEvidence(
            fact=Fact(id="f1", subject="e1", predicate="code:calls", object="e2"),
            evidence=[],
        )


def test_budget_bounds_are_positive() -> None:
    with pytest.raises(ValidationError):
        RecallBudget(max_facts=0)
