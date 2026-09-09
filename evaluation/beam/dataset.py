"""BEAM dataset loading and document normalization.

One conversation → one ingest unit (``group_id == conversation_id``): its ``chat``
sessions plus a user-profile document are ingested, and every probing question
across the ten categories becomes an :class:`EvalCase`. The gold-answer field
name varies by category (see ``_GOLD_FIELD``); the two compliance categories
carry no gold string and are graded against their rubric (see the adapter).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dateparser import parse as _parse_date

from evaluation.beam.config import BeamConfig
from evaluation.common.datamodels import EvalCase, EvalDocument, IngestUnit
from evaluation.common.dataset import clip, message_turns, reference_date

# Category → the field holding its gold answer.
_GOLD_FIELD = {
    "event_ordering": "answer",
    "information_extraction": "answer",
    "knowledge_update": "answer",
    "multi_session_reasoning": "answer",
    "temporal_reasoning": "answer",
    "contradiction_resolution": "ideal_answer",
    "abstention": "ideal_response",
    "summarization": "ideal_summary",
}
# Categories with no single gold string — graded against a compliance rubric.
_RUBRIC_CATEGORIES = frozenset({"instruction_following", "preference_following"})


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist (BEAM dataset missing).")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{path} must be a JSON array of conversations")
    return [row for row in payload if isinstance(row, dict)]


def _parse_iso(raw: str) -> str:
    """ISO datetime from a BEAM time anchor (``March 15, 2024`` / ``March-15-2024``)."""
    text = (raw or "").strip()
    if not text:
        return ""
    dt = _parse_date(text) or _parse_date(text.replace("-", " "))
    return dt.isoformat() if dt else ""


def _message_line(msg: Any) -> str:
    if not isinstance(msg, dict):
        return ""
    content = str(msg.get("content") or "").strip()
    if not content:
        return ""
    role = str(msg.get("role") or "message")
    return f"{role}: {content}"


def _split_session(session: list[dict[str, Any]], max_chars: int) -> list[str]:
    """Split a session into bounded text blocks on message boundaries.

    BEAM sessions run 150K-185K chars — far too large to ingest as one document
    (a single oversized ingest crashes the extraction model). Splitting on message
    boundaries keeps every message intact and every block within ``max_chars``,
    preserving all content (unlike clipping). A lone message longer than
    ``max_chars`` is hard-wrapped.
    """
    blocks: list[str] = []
    current: list[str] = []
    size = 0
    for msg in session:
        line = _message_line(msg)
        if not line:
            continue
        pieces = (
            [line[i : i + max_chars] for i in range(0, len(line), max_chars)]
            if len(line) > max_chars
            else [line]
        )
        for piece in pieces:
            if current and size + len(piece) > max_chars:
                blocks.append("\n".join(current))
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 1
    if current:
        blocks.append("\n".join(current))
    return blocks


def _session_anchor(session: list[dict[str, Any]]) -> str:
    for msg in session:
        if isinstance(msg, dict) and msg.get("time_anchor"):
            return _parse_iso(str(msg["time_anchor"]))
    return ""


def build_documents(
    record: dict[str, Any], *, max_documents: int, max_chars: int, turn_fed: bool = False
) -> list[EvalDocument]:
    """User-profile document + chat sessions split into bounded ingest documents.

    ``turn_fed`` keeps each message separate — one turn-fed document per session,
    chunked by the pipeline's own sliding window instead of by ``max_chars``.
    """
    documents: list[EvalDocument] = []

    profile = record.get("user_profile") or {}
    if isinstance(profile, dict):
        parts = [f"{k}: {v}" for k, v in profile.items() if v]
        if parts:
            documents.append(
                EvalDocument(title="user-profile", text=clip("\n".join(parts), max_chars))
            )

    chat = record.get("chat") or []
    for index, session in enumerate(chat):
        if not isinstance(session, list):
            continue
        anchor = _session_anchor(session)
        if turn_fed:
            if len(documents) >= max_documents:
                return documents
            if turns := message_turns(session, timestamp=anchor):
                documents.append(
                    EvalDocument(title=f"session-{index}", text="", anchor_date=anchor, turns=turns)
                )
            continue
        blocks = _split_session(session, max_chars)
        for part, block in enumerate(blocks):
            if len(documents) >= max_documents:
                return documents
            title = f"session-{index}" if len(blocks) == 1 else f"session-{index}-part{part}"
            documents.append(EvalDocument(title=title, text=block, anchor_date=anchor))
    return documents


def _message_id_map(record: dict[str, Any]) -> dict[int, str]:
    """Global message-id -> content, across every chat session (ids are global)."""
    out: dict[int, str] = {}
    for session in record.get("chat") or []:
        if not isinstance(session, list):
            continue
        for msg in session:
            if isinstance(msg, dict) and isinstance(msg.get("id"), int):
                out[msg["id"]] = " ".join(str(msg.get("content") or "").split())
    return out


def _flatten_source_ids(value: Any) -> list[int]:
    """``source_chat_ids`` is a flat list, or a dict of labelled lists (contradiction/
    temporal/update questions carry ``{first_statement: [...], second_statement: [...]}``)."""
    if isinstance(value, list):
        return [i for i in value if isinstance(i, int)]
    if isinstance(value, dict):
        out: list[int] = []
        for v in value.values():
            if isinstance(v, list):
                out.extend(i for i in v if isinstance(i, int))
            elif isinstance(v, int):
                out.append(v)
        return out
    return []


def _evidence_windows(source_ids: Any, id_map: dict[int, str]) -> list[str]:
    """Gold evidence for a BEAM question, as substring-matchable windows.

    BEAM ships ``source_chat_ids`` — the message ids that answer each question —
    but its messages are long (median ~1150 chars) and the ingest chunker splits
    them, so a whole message rarely sits in one CHUNK verbatim (unlike LoCoMo's
    short turns). Each source message is therefore emitted as overlapping
    50-char windows: ``evidence_recall`` (exact substring per item) then scores
    the fraction of a message's windows that were retrieved, which degrades
    gracefully when a message is split across several retrieved/unretrieved
    chunks. A short message yields a single window (itself).
    """
    windows: list[str] = []
    for mid in dict.fromkeys(_flatten_source_ids(source_ids)):
        content = id_map.get(mid, "")
        if len(content) < 20:
            continue
        if len(content) <= 80:
            windows.append(content)
            continue
        windows.extend(content[i : i + 50] for i in range(0, len(content) - 49, 40))
    return windows


def _build_cases(record: dict[str, Any], conv_id: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    id_map = _message_id_map(record)
    probing = record.get("probing_questions") or {}
    for category, questions in probing.items():
        if not isinstance(questions, list):
            continue
        for i, q in enumerate(questions, start=1):
            if not isinstance(q, dict):
                continue
            question = str(q.get("question") or "")
            if not question:
                continue
            is_rubric = category in _RUBRIC_CATEGORIES
            gold = "" if is_rubric else str(q.get(_GOLD_FIELD.get(category, "answer")) or "")
            extras: dict[str, Any] = {
                "rubric": q.get("rubric"),
                "source_chat_ids": q.get("source_chat_ids"),
                "difficulty": q.get("difficulty"),
                # Resolved gold evidence -> the shared evidence_recall metric,
                # so BEAM retrieval quality is measurable (was proxy-only).
                "evidence": _evidence_windows(q.get("source_chat_ids"), id_map),
            }
            if is_rubric:
                extras.update(
                    being_tested=q.get("instruction_being_tested")
                    or q.get("preference_being_tested"),
                    expected_compliance=q.get("expected_compliance"),
                    compliance_indicators=q.get("compliance_indicators"),
                    non_compliance_signs=q.get("non_compliance_signs"),
                )
            cases.append(
                EvalCase(
                    case_id=f"{conv_id}-{category}-{i}",
                    group_id=conv_id,
                    question=question,
                    gold=gold,
                    category=category,
                    extras=extras,
                )
            )
    return cases


def load_units(config: BeamConfig, *, limit: int, offset: int) -> list[IngestUnit]:
    """Load a window of BEAM conversations; one ingest unit per conversation.

    ``--limit``/``--offset`` slice CONVERSATIONS (each carries ~20 questions),
    unlike LoCoMo/LongMemEval where they slice questions.
    """
    records = _load_records(config.data_root / config.questions_file)
    selected = records[offset : offset + limit]

    units: list[IngestUnit] = []
    for record in selected:
        conv_id = str(record.get("conversation_id") or "")
        if not conv_id:
            raise ValueError("BEAM conversation has no conversation_id")
        documents = build_documents(
            record,
            max_documents=config.max_documents,
            max_chars=config.max_chars_per_document,
            turn_fed=config.turn_fed,
        )
        cases = _build_cases(record, conv_id)
        if not cases:
            continue
        units.append(
            IngestUnit(
                group_id=conv_id,
                documents=documents,
                cases=cases,
                reference_date=reference_date(documents),
            )
        )
    return units
