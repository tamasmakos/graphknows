"""Parallel evidence thread dispatcher node functions."""
from __future__ import annotations

import uuid
from typing import Any

from services.agent.graph.state import (
    DocumentRef,
    EvidenceRecord,
    EvidenceThread,
    InvestigativeState,
    ProvenanceEnvelope,
    RetrievalEventRef,
    ThreadProgressEvent,
    utc_now_iso,
)


def dispatch_threads(state: InvestigativeState) -> dict[str, Any]:
    """Node: investigate_dispatch_threads.

    Mark all pending evidence threads as ``running`` and emit one
    ``thread_started`` event per started thread.

    Args:
        state: Current investigative state.

    Returns:
        Partial state update with ``evidence_threads`` and ``thread_events``.
    """

    updated_threads: list[EvidenceThread] = []
    new_events: list[ThreadProgressEvent] = []

    for thread in state.get("evidence_threads", []):
        updated_thread = EvidenceThread(**thread)
        if updated_thread["status"] == "pending":
            updated_thread["status"] = "running"
            new_events.append(
                ThreadProgressEvent(
                    event_id=uuid.uuid4().hex,
                    kind="thread_started",
                    thread_id=updated_thread["thread_id"],
                    evidence_id=None,
                    emitted_at=utc_now_iso(),
                )
            )
        updated_threads.append(updated_thread)

    return {
        "evidence_threads": updated_threads,
        "thread_events": [*state.get("thread_events", []), *new_events],
    }


def collect_thread_evidence(state: InvestigativeState, thread_id: str) -> dict[str, Any]:
    """Node: investigate_collect_thread_evidence.

    Collect one stub evidence record for the thread, append it to the ledger,
    mark the thread as ``complete``, and emit one ``evidence_added`` event.

    Args:
        state: Current investigative state.
        thread_id: Target evidence thread id.

    Returns:
        Partial state update with ``evidence_threads``, ``evidence_ledger``,
        and ``thread_events``.

    Raises:
        ValueError: If the requested thread id is not present in state.
    """

    threads = [EvidenceThread(**thread) for thread in state.get("evidence_threads", [])]
    target_index = next((idx for idx, t in enumerate(threads) if t["thread_id"] == thread_id), -1)
    if target_index < 0:
        raise ValueError(f"Unknown thread_id: {thread_id}")

    thread = threads[target_index]
    now = utc_now_iso()
    evidence_id = uuid.uuid4().hex

    retrieval: RetrievalEventRef = {
        "retrieval_id": uuid.uuid4().hex,
        "query": f"evidence for {thread['title']}",
        "tool_name": "thread_dispatcher_stub",
        "retrieved_at": now,
        "rank": 1,
        "score": None,
        "source_type": thread["source_classes"][0] if thread["source_classes"] else "unknown",
    }
    document: DocumentRef = {
        "document_id": uuid.uuid4().hex,
        "title": f"Source for {thread['title']}",
        "url": f"https://example.test/doc/{thread_id}",
        "publisher": "Stub Publisher",
        "published_at": now,
    }
    provenance: ProvenanceEnvelope = {
        "retrieval": retrieval,
        "document": document,
        "passage": None,
        "relation": None,
        "graph_path": None,
    }

    record: EvidenceRecord = {
        "evidence_id": evidence_id,
        "thread_id": thread_id,
        "sub_question_id": thread["sub_question_id"],
        "summary": f"Stub evidence for {thread['title']}",
        "source_authority": 3,
        "independence": 3,
        "directness": 3,
        "recency": 3,
        "claim_ids": [],
        "tags": ["stub"],
        "provenance": provenance,
    }

    thread["status"] = "complete"
    thread["evidence_ids"] = [*thread["evidence_ids"], evidence_id]
    threads[target_index] = thread

    event = ThreadProgressEvent(
        event_id=uuid.uuid4().hex,
        kind="evidence_added",
        thread_id=thread_id,
        evidence_id=evidence_id,
        emitted_at=now,
    )

    return {
        "evidence_threads": threads,
        "evidence_ledger": [*state.get("evidence_ledger", []), record],
        "thread_events": [*state.get("thread_events", []), event],
    }


def all_threads_done(state: InvestigativeState) -> bool:
    """Return True when every thread has terminal status.

    Terminal thread statuses are ``complete`` and ``blocked``.

    Args:
        state: Current investigative state.

    Returns:
        True if all threads are terminal; otherwise False.
    """

    return all(
        thread["status"] in {"complete", "blocked"}
        for thread in state.get("evidence_threads", [])
    )
