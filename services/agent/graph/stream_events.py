"""Typed stream event helpers for the investigative workflow WebSocket stream.

Every event emitted over the WS channel has a ``type`` field drawn from
``REQUIRED_STREAM_EVENTS`` (defined in workflow.py).  Callers build events
with the factory functions here and forward them to the WS sender of their
choice (FastAPI ``EventSourceResponse``, ``asyncio.Queue``, etc.).

Design contract
---------------
- Events are plain dicts so they serialise to JSON without extra setup.
- Every event carries ``session_id`` and ``occurred_at`` (ISO-8601 UTC).
- Thread-scoped events additionally carry ``thread_id``.
- Evidence events additionally carry ``evidence_id`` and ``sub_question_id``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Generic builder
# ---------------------------------------------------------------------------


def build_event(
    event_type: str,
    session_id: str,
    payload: dict[str, Any] | None = None,
    *,
    occurred_at: str | None = None,
) -> dict[str, Any]:
    """Return a base stream event dict.

    Args:
        event_type: One of the REQUIRED_STREAM_EVENTS strings.
        session_id: Investigation session identifier.
        payload: Additional keys merged into the event.
        occurred_at: ISO-8601 timestamp override (current UTC if None).

    Returns:
        A dict safe to pass to ``json.dumps``.
    """
    event: dict[str, Any] = {
        "type": event_type,
        "session_id": session_id,
        "occurred_at": occurred_at or _utc_now_iso(),
    }
    if payload:
        event.update(payload)
    return event


# ---------------------------------------------------------------------------
# Specific event factories
# ---------------------------------------------------------------------------


def thread_started_event(
    session_id: str,
    thread_id: str,
    title: str,
    source_classes: list[str],
    sub_question_id: str,
    *,
    occurred_at: str | None = None,
) -> dict[str, Any]:
    """Emit when a thread transitions from pending → running.

    Args:
        session_id: Investigation session identifier.
        thread_id: Unique thread identifier.
        title: Human-readable thread title.
        source_classes: Source-class lanes assigned to the thread.
        sub_question_id: Sub-question the thread is investigating.
        occurred_at: ISO-8601 timestamp override.

    Returns:
        A ``thread_started`` stream event dict.
    """
    return build_event(
        "thread_started",
        session_id,
        {
            "thread_id": thread_id,
            "title": title,
            "source_classes": source_classes,
            "sub_question_id": sub_question_id,
        },
        occurred_at=occurred_at,
    )


def evidence_added_event(
    session_id: str,
    thread_id: str,
    evidence_id: str,
    sub_question_id: str,
    summary: str,
    source_type: str,
    *,
    occurred_at: str | None = None,
) -> dict[str, Any]:
    """Emit when a new EvidenceRecord is appended to the ledger.

    Args:
        session_id: Investigation session identifier.
        thread_id: Thread that produced the evidence.
        evidence_id: Unique evidence record identifier.
        sub_question_id: Sub-question being investigated.
        summary: Short human-readable summary of the evidence.
        source_type: Source-class label (e.g. ``"ep_debates"``).
        occurred_at: ISO-8601 timestamp override.

    Returns:
        An ``evidence_added`` stream event dict.
    """
    return build_event(
        "evidence_added",
        session_id,
        {
            "thread_id": thread_id,
            "evidence_id": evidence_id,
            "sub_question_id": sub_question_id,
            "summary": summary,
            "source_type": source_type,
        },
        occurred_at=occurred_at,
    )


def phase_change_event(
    session_id: str,
    from_phase: str,
    to_phase: str,
    *,
    occurred_at: str | None = None,
) -> dict[str, Any]:
    """Emit on every macro-phase transition.

    Args:
        session_id: Investigation session identifier.
        from_phase: Previous macro-phase label.
        to_phase: Incoming macro-phase label.
        occurred_at: ISO-8601 timestamp override.

    Returns:
        A ``phase_change`` stream event dict.
    """
    return build_event(
        "phase_change",
        session_id,
        {"from_phase": from_phase, "to_phase": to_phase},
        occurred_at=occurred_at,
    )
