from __future__ import annotations

from services.agent.graph.state import EvidenceThread, build_initial_investigative_state
from services.agent.graph.thread_dispatcher import (
    all_threads_done,
    collect_thread_evidence,
    dispatch_threads,
)


def _make_thread(n: int) -> EvidenceThread:
    return {
        "thread_id": f"thread-{n}",
        "title": f"Thread {n}",
        "sub_question_id": f"sq-{n}",
        "source_classes": ["eprs_briefings"],
        "status": "pending",
        "evidence_ids": [],
        "notes": [],
    }


def test_dispatch_threads_marks_all_pending_threads_running():
    state = build_initial_investigative_state("sess-1", "query")
    state["evidence_threads"] = [_make_thread(1), _make_thread(2)]

    result = dispatch_threads(state)

    assert all(t["status"] == "running" for t in result["evidence_threads"])
    assert len(result["thread_events"]) == 2
    assert all(e["kind"] == "thread_started" for e in result["thread_events"])
    assert all(e["evidence_id"] is None for e in result["thread_events"])


def test_collect_thread_evidence_appends_to_ledger_with_provenance():
    state = build_initial_investigative_state("sess-2", "query")
    state["evidence_threads"] = [_make_thread(1)]
    dispatch = dispatch_threads(state)
    state["evidence_threads"] = dispatch["evidence_threads"]
    state["thread_events"] = dispatch["thread_events"]

    result = collect_thread_evidence(state, "thread-1")

    assert len(result["evidence_ledger"]) == 1
    record = result["evidence_ledger"][0]
    provenance = record["provenance"]
    assert provenance["retrieval"]["retrieval_id"]
    assert provenance["document"]["document_id"]
    assert provenance["passage"] is None
    assert provenance["relation"] is None
    assert provenance["graph_path"] is None


def test_collect_thread_evidence_emits_evidence_added_events():
    state = build_initial_investigative_state("sess-3", "query")
    state["evidence_threads"] = [_make_thread(1)]
    dispatch = dispatch_threads(state)
    state["evidence_threads"] = dispatch["evidence_threads"]
    state["thread_events"] = dispatch["thread_events"]

    result = collect_thread_evidence(state, "thread-1")

    assert len(result["thread_events"]) == 2
    added_event = result["thread_events"][-1]
    ledger_record = result["evidence_ledger"][0]
    assert added_event["kind"] == "evidence_added"
    assert added_event["thread_id"] == "thread-1"
    assert added_event["evidence_id"] == ledger_record["evidence_id"]


def test_all_threads_done_requires_every_thread_terminal():
    done_state = build_initial_investigative_state("sess-4", "query")
    done_state["evidence_threads"] = [_make_thread(1), _make_thread(2)]
    done_state["evidence_threads"][0]["status"] = "complete"
    done_state["evidence_threads"][1]["status"] = "blocked"

    not_done_state = build_initial_investigative_state("sess-5", "query")
    not_done_state["evidence_threads"] = [_make_thread(1), _make_thread(2)]
    not_done_state["evidence_threads"][0]["status"] = "complete"
    not_done_state["evidence_threads"][1]["status"] = "running"

    assert all_threads_done(done_state) is True
    assert all_threads_done(not_done_state) is False


def test_evidence_ids_in_thread_match_ledger_entries():
    state = build_initial_investigative_state("sess-6", "query")
    state["evidence_threads"] = [_make_thread(1), _make_thread(2)]
    dispatch = dispatch_threads(state)
    state["evidence_threads"] = dispatch["evidence_threads"]
    state["thread_events"] = dispatch["thread_events"]

    first = collect_thread_evidence(state, "thread-1")
    state["evidence_threads"] = first["evidence_threads"]
    state["evidence_ledger"] = first["evidence_ledger"]
    state["thread_events"] = first["thread_events"]

    second = collect_thread_evidence(state, "thread-2")

    for thread in second["evidence_threads"]:
        ledger_ids = {
            record["evidence_id"]
            for record in second["evidence_ledger"]
            if record["thread_id"] == thread["thread_id"]
        }
        assert set(thread["evidence_ids"]) == ledger_ids


def test_dispatch_then_collect_pipeline():
    state = build_initial_investigative_state("sess-7", "query")
    state["evidence_threads"] = [_make_thread(1), _make_thread(2)]

    dispatch = dispatch_threads(state)
    state["evidence_threads"] = dispatch["evidence_threads"]
    state["thread_events"] = dispatch["thread_events"]

    one = collect_thread_evidence(state, "thread-1")
    state["evidence_threads"] = one["evidence_threads"]
    state["evidence_ledger"] = one["evidence_ledger"]
    state["thread_events"] = one["thread_events"]

    two = collect_thread_evidence(state, "thread-2")

    assert len(dispatch["thread_events"]) == 2
    assert len(two["thread_events"]) == 4
    assert len(two["evidence_ledger"]) == 2
    assert all_threads_done({**state, **two}) is True
