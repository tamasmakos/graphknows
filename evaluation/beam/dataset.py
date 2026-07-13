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
        raise ValueError(f"{path} must be a JSON array of conversations")
    return [row for row in payload if isinstance(row, dict)]


def _parse_iso(raw: str) -> str:
    """ISO datetime from a BEAM time anchor (``March 15, 2024`` / ``March-15-2024``)."""
    text = (raw or "").strip()
    if not text:
        return ""
    dt = _parse_date(text) or _parse_date(text.replace("-", " "))
    return dt.isoformat() if dt else ""


def _clip(text: str, max_chars: int) -> str:
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def _session_text(session: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for msg in session:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        role = str(msg.get("role") or "message")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _session_anchor(session: list[dict[str, Any]]) -> str:
    for msg in session:
        if isinstance(msg, dict) and msg.get("time_anchor"):
            return _parse_iso(str(msg["time_anchor"]))
    return ""


def build_documents(
    record: dict[str, Any], *, max_documents: int, max_chars: int
) -> list[EvalDocument]:
    """Chat sessions + a user-profile document, ready to ingest."""
    documents: list[EvalDocument] = []

    profile = record.get("user_profile") or {}
    if isinstance(profile, dict):
        parts = [f"{k}: {v}" for k, v in profile.items() if v]
        if parts:
            documents.append(
                EvalDocument(title="user-profile", text=_clip("\n".join(parts), max_chars))
            )

    chat = record.get("chat") or []
    for index, session in enumerate(chat[: max(0, max_documents - len(documents))]):
        if not isinstance(session, list):
            continue
        text = _clip(_session_text(session), max_chars)
        if not text:
            continue
        documents.append(
            EvalDocument(
                title=f"session-{index}", text=text, anchor_date=_session_anchor(session)
            )
        )
    return documents


def _reference_date(documents: list[EvalDocument]) -> str:
    parsed = [
        dt for doc in documents if doc.anchor_date and (dt := _parse_date(doc.anchor_date))
    ]
    return max(parsed).strftime("%d %B %Y") if parsed else ""


def _build_cases(record: dict[str, Any], conv_id: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
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
        )
        cases = _build_cases(record, conv_id)
        if not cases:
            continue
        units.append(
            IngestUnit(
                group_id=conv_id,
                documents=documents,
                cases=cases,
                reference_date=_reference_date(documents),
            )
        )
    return units
