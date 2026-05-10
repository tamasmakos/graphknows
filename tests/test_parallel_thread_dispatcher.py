"""Tests for the parallel evidence thread dispatcher nodes (Slice 5).

Coverage:
- investigate_dispatch_threads: thread creation from evidence_plan, deduplication,
  status set to running, stream events emitted with correct fields.
- investigate_collect_thread_evidence: evidence tagging with thread_id,
  ledger append, idempotent de-dup, stream events, thread status transition to complete.
- Thread status transitions: pending → running (dispatch), running → complete (collect).
"""
from __future__ import annotations

from services.agent.graph.state import (
    EvidenceRecord,
    EvidenceThread,
    build_initial_investigative_state,
)
from services.agent.graph.thread_dispatcher import (
    investigate_collect_thread_evidence,
    investigate_dispatch_threads,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plan_entry(
    sub_question_id: str,
    title: str = "",
    source_classes: list[str] | None = None,
    thread_id: str | None = None,
) -> dict:
    return {
        "sub_question_id": sub_question_id,
        "title": title or sub_question_id,
        "source_classes": source_classes or [],
        "thread_id": thread_id,
    }


def _minimal_evidence_record(
    evidence_id: str,
    thread_id: str,
    sub_question_id: str = "sq-1",
    source_type: str = "ep_debates",
    summary: str = "Test evidence summary.",
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        thread_id=thread_id,
        sub_question_id=sub_question_id,
        summary=summary,
        source_authority=5,
        independence=3,
        directness=5,
        recency=4,
        claim_ids=[],
        tags=[source_type],
        provenance={
            "retrieval": {
                "retrieval_id": f"ret-{evidence_id}",
                "query": "test query",
                "tool_name": "search_eu",
                "retrieved_at": "2026-05-10T18:00:00+00:00",
                "rank": 0,
                "score": 0.9,
                "source_type": source_type,
            },
            "document": {
                "document_id": f"doc-{evidence_id}",
                "title": "Test document",
                "url": "https://example.test/doc",
                "publisher": "Test Publisher",
                "published_at": "2024-01-01",
            },
            "passage": {
                "chunk_id": f"chunk-{evidence_id}",
                "text": "Relevant passage text.",
                "start_char": 0,
                "end_char": 22,
            },
            "relation": None,
            "graph_path": None,
        },
    )


# ---------------------------------------------------------------------------
# investigate_dispatch_threads
# ---------------------------------------------------------------------------


def test_dispatch_threads_creates_one_thread_per_plan_entry():
    state = build_initial_investigative_state("sess-1", "test query")
    state["evidence_plan"] = [
        _plan_entry("sq-1", "Policy text lane", ["eurlex", "ep_debates"]),
        _plan_entry("sq-2", "Actor mapping lane", ["ep_debates"]),
    ]

    result = investigate_dispatch_threads(state, emit_events=False)

    threads = result["evidence_threads"]
    assert len(threads) == 2
    sq_ids = {t["sub_question_id"] for t in threads}
    assert sq_ids == {"sq-1", "sq-2"}


def test_dispatch_threads_sets_status_running():
    state = build_initial_investigative_state("sess-2", "test query")
    state["evidence_plan"] = [_plan_entry("sq-1", "Lane A")]

    result = investigate_dispatch_threads(state, emit_events=False)

    for thread in result["evidence_threads"]:
        assert thread["status"] == "running"


def test_dispatch_threads_emits_thread_started_events():
    state = build_initial_investigative_state("sess-3", "test query")
    state["evidence_plan"] = [
        _plan_entry("sq-1", "Policy text", ["eurlex"]),
        _plan_entry("sq-2", "Chronology", ["ep_debates"]),
    ]

    result = investigate_dispatch_threads(state, emit_events=True)

    events = result["stream_events"]
    assert len(events) == 2
    types = {e["type"] for e in events}
    assert types == {"thread_started"}

    event_sq_ids = {e["sub_question_id"] for e in events}
    assert event_sq_ids == {"sq-1", "sq-2"}

    for event in events:
        assert "thread_id" in event
        assert "session_id" in event
        assert "occurred_at" in event
        assert "source_classes" in event


def test_dispatch_threads_uses_provided_thread_id():
    state = build_initial_investigative_state("sess-4", "test query")
    state["evidence_plan"] = [
        _plan_entry("sq-1", "Lane A", thread_id="thread-fixed-id")
    ]

    result = investigate_dispatch_threads(state, emit_events=False)

    thread = result["evidence_threads"][0]
    assert thread["thread_id"] == "thread-fixed-id"


def test_dispatch_threads_deduplicates_existing_sub_questions():
    state = build_initial_investigative_state("sess-5", "test query")
    existing_thread = EvidenceThread(
        thread_id="thread-existing",
        title="Existing lane",
        sub_question_id="sq-1",
        source_classes=["eurlex"],
        status="running",
        evidence_ids=[],
        notes=[],
    )
    state["evidence_threads"] = [existing_thread]
    state["evidence_plan"] = [
        _plan_entry("sq-1", "Duplicate lane"),  # already dispatched
        _plan_entry("sq-2", "New lane"),
    ]

    result = investigate_dispatch_threads(state, emit_events=False)

    threads = result["evidence_threads"]
    assert len(threads) == 2  # existing sq-1 + new sq-2
    sq_ids = {t["sub_question_id"] for t in threads}
    assert sq_ids == {"sq-1", "sq-2"}
    # The existing thread must be preserved unchanged
    existing = next(t for t in threads if t["sub_question_id"] == "sq-1")
    assert existing["thread_id"] == "thread-existing"


def test_dispatch_threads_empty_plan_returns_no_threads():
    state = build_initial_investigative_state("sess-6", "test query")
    state["evidence_plan"] = []

    result = investigate_dispatch_threads(state, emit_events=True)

    assert result["evidence_threads"] == []
    assert result["stream_events"] == []


# ---------------------------------------------------------------------------
# investigate_collect_thread_evidence
# ---------------------------------------------------------------------------


def test_collect_thread_evidence_appends_to_ledger():
    state = build_initial_investigative_state("sess-10", "test query")
    state["evidence_plan"] = [_plan_entry("sq-1", "Policy lane")]
    dispatch_result = investigate_dispatch_threads(state, emit_events=False)
    state["evidence_threads"] = dispatch_result["evidence_threads"]

    thread_id = state["evidence_threads"][0]["thread_id"]
    record = _minimal_evidence_record("ev-001", thread_id, "sq-1")

    result = investigate_collect_thread_evidence(
        state, incoming_records=[record], emit_events=False
    )

    assert len(result["evidence_ledger"]) == 1
    assert result["evidence_ledger"][0]["evidence_id"] == "ev-001"


def test_collect_thread_evidence_tags_records_with_thread_id():
    state = build_initial_investigative_state("sess-11", "test query")
    state["evidence_plan"] = [_plan_entry("sq-1", "Lane A")]
    dispatch_result = investigate_dispatch_threads(state, emit_events=False)
    state["evidence_threads"] = dispatch_result["evidence_threads"]

    thread_id = state["evidence_threads"][0]["thread_id"]
    record = _minimal_evidence_record("ev-002", thread_id, "sq-1")

    result = investigate_collect_thread_evidence(
        state, incoming_records=[record], emit_events=False
    )

    ledger_record = result["evidence_ledger"][0]
    assert ledger_record["thread_id"] == thread_id


def test_collect_thread_evidence_updates_thread_evidence_ids():
    state = build_initial_investigative_state("sess-12", "test query")
    state["evidence_plan"] = [_plan_entry("sq-1", "Lane A")]
    dispatch_result = investigate_dispatch_threads(state, emit_events=False)
    state["evidence_threads"] = dispatch_result["evidence_threads"]

    thread_id = state["evidence_threads"][0]["thread_id"]
    r1 = _minimal_evidence_record("ev-a", thread_id, "sq-1")
    r2 = _minimal_evidence_record("ev-b", thread_id, "sq-1")

    result = investigate_collect_thread_evidence(
        state, incoming_records=[r1, r2], emit_events=False
    )

    updated_thread = next(
        t for t in result["evidence_threads"] if t["thread_id"] == thread_id
    )
    assert "ev-a" in updated_thread["evidence_ids"]
    assert "ev-b" in updated_thread["evidence_ids"]


def test_collect_thread_evidence_emits_evidence_added_events():
    state = build_initial_investigative_state("sess-13", "test query")
    state["evidence_plan"] = [_plan_entry("sq-1", "Lane A")]
    dispatch_result = investigate_dispatch_threads(state, emit_events=False)
    state["evidence_threads"] = dispatch_result["evidence_threads"]

    thread_id = state["evidence_threads"][0]["thread_id"]
    record = _minimal_evidence_record("ev-evt", thread_id, "sq-1", source_type="eurlex")

    result = investigate_collect_thread_evidence(
        state, incoming_records=[record], emit_events=True
    )

    events = result["stream_events"]
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == "evidence_added"
    assert ev["evidence_id"] == "ev-evt"
    assert ev["thread_id"] == thread_id
    assert ev["sub_question_id"] == "sq-1"
    assert ev["source_type"] == "eurlex"
    assert "occurred_at" in ev
    assert "session_id" in ev


def test_collect_thread_evidence_is_idempotent_for_duplicate_ids():
    state = build_initial_investigative_state("sess-14", "test query")
    state["evidence_plan"] = [_plan_entry("sq-1", "Lane A")]
    dispatch_result = investigate_dispatch_threads(state, emit_events=False)
    state["evidence_threads"] = dispatch_result["evidence_threads"]

    thread_id = state["evidence_threads"][0]["thread_id"]
    record = _minimal_evidence_record("ev-dup", thread_id, "sq-1")

    # First ingest
    result1 = investigate_collect_thread_evidence(
        state, incoming_records=[record], emit_events=False
    )
    # Simulate next call with same record
    state["evidence_ledger"] = result1["evidence_ledger"]
    state["evidence_threads"] = result1["evidence_threads"]
    result2 = investigate_collect_thread_evidence(
        state, incoming_records=[record], emit_events=False
    )

    # Only one record in ledger total
    assert len(result2["evidence_ledger"]) == 1


# ---------------------------------------------------------------------------
# Thread status transitions
# ---------------------------------------------------------------------------


def test_thread_status_transitions_running_to_complete_on_empty_batch():
    """Passing empty incoming_records signals end-of-retrieval → complete."""
    state = build_initial_investigative_state("sess-20", "test query")
    state["evidence_plan"] = [_plan_entry("sq-1", "Lane A")]
    dispatch_result = investigate_dispatch_threads(state, emit_events=False)
    state["evidence_threads"] = dispatch_result["evidence_threads"]

    # Verify starting status
    assert state["evidence_threads"][0]["status"] == "running"

    result = investigate_collect_thread_evidence(
        state, incoming_records=[], emit_events=False
    )

    for thread in result["evidence_threads"]:
        assert thread["status"] == "complete"


def test_thread_status_pending_not_affected_by_empty_batch():
    """Only running threads are completed by an empty batch; pending stays pending."""
    state = build_initial_investigative_state("sess-21", "test query")
    pending_thread = EvidenceThread(
        thread_id="thread-pending",
        title="Pending lane",
        sub_question_id="sq-2",
        source_classes=[],
        status="pending",
        evidence_ids=[],
        notes=[],
    )
    running_thread = EvidenceThread(
        thread_id="thread-running",
        title="Running lane",
        sub_question_id="sq-1",
        source_classes=[],
        status="running",
        evidence_ids=[],
        notes=[],
    )
    state["evidence_threads"] = [pending_thread, running_thread]

    result = investigate_collect_thread_evidence(
        state, incoming_records=[], emit_events=False
    )

    by_id = {t["thread_id"]: t for t in result["evidence_threads"]}
    assert by_id["thread-running"]["status"] == "complete"
    assert by_id["thread-pending"]["status"] == "pending"


def test_multiple_threads_collect_evidence_independently():
    """Records tagged with different thread_ids are routed to the correct threads."""
    state = build_initial_investigative_state("sess-22", "test query")
    state["evidence_plan"] = [
        _plan_entry("sq-1", "Policy text", thread_id="thread-policy"),
        _plan_entry("sq-2", "Actor mapping", thread_id="thread-actors"),
    ]
    dispatch_result = investigate_dispatch_threads(state, emit_events=False)
    state["evidence_threads"] = dispatch_result["evidence_threads"]

    r_policy = _minimal_evidence_record("ev-policy", "thread-policy", "sq-1")
    r_actors = _minimal_evidence_record("ev-actors", "thread-actors", "sq-2")

    result = investigate_collect_thread_evidence(
        state, incoming_records=[r_policy, r_actors], emit_events=False
    )

    by_id = {t["thread_id"]: t for t in result["evidence_threads"]}
    assert "ev-policy" in by_id["thread-policy"]["evidence_ids"]
    assert "ev-actors" in by_id["thread-actors"]["evidence_ids"]
    assert "ev-actors" not in by_id["thread-policy"]["evidence_ids"]
    assert len(result["evidence_ledger"]) == 2
