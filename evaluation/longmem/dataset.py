"""LongMemEval dataset loading and document normalization.

LongMemEval stores each question with its own haystack as three parallel arrays
(``haystack_session_ids`` / ``haystack_dates`` / ``haystack_sessions``) that we
zip by index into one :class:`EvalDocument` per session. Each question is its own
ingest unit (``group_id == question_id``) because the haystack is per-question.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dateparser import parse as _parse_date

from evaluation.common.datamodels import EvalCase, EvalDocument, IngestUnit
from evaluation.longmem.config import LongMemConfig

# LongMemEval dates look like "2023/05/30 (Tue) 23:40"; strip the weekday
# parenthetical so dateparser reads the timestamp cleanly.
_WEEKDAY_PAREN = re.compile(r"\([^)]*\)")


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist (LongMemEval dataset missing).")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be a JSON array of question records")
    return [row for row in payload if isinstance(row, dict)]


def _parse_iso(raw: str) -> str:
    """ISO datetime from a LongMemEval date string, or '' when unparseable."""
    cleaned = _WEEKDAY_PAREN.sub("", raw or "").strip()
    if not cleaned:
        return ""
    dt = _parse_date(cleaned)
    return dt.isoformat() if dt else ""


def _clip(text: str, max_chars: int) -> str:
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def _session_text(session: list[dict[str, Any]]) -> str:
    """Join a session's messages into one ``role: content`` blob."""
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


def build_documents(
    record: dict[str, Any], *, max_documents: int, max_chars: int
) -> list[EvalDocument]:
    """Zip the parallel haystack arrays into ingestable per-session documents."""
    ids = record.get("haystack_session_ids") or []
    dates = record.get("haystack_dates") or []
    sessions = record.get("haystack_sessions") or []
    documents: list[EvalDocument] = []
    for index, session in enumerate(sessions[:max_documents]):
        if not isinstance(session, list):
            continue
        text = _clip(_session_text(session), max_chars)
        if not text:
            continue
        title = str(ids[index]) if index < len(ids) else f"session-{index}"
        anchor = _parse_iso(dates[index]) if index < len(dates) else ""
        documents.append(EvalDocument(title=title, text=text, anchor_date=anchor))
    return documents


def _reference_date(documents: list[EvalDocument]) -> str:
    """Human-readable date of the newest session (anchors relative time)."""
    parsed = [
        dt for doc in documents if doc.anchor_date and (dt := _parse_date(doc.anchor_date))
    ]
    return max(parsed).strftime("%d %B %Y") if parsed else ""


def load_units(config: LongMemConfig, *, limit: int, offset: int) -> list[IngestUnit]:
    """Load the selected question window; one ingest unit per question."""
    records = _load_records(config.data_root / config.questions_file)
    selected = records[offset : offset + limit]

    units: list[IngestUnit] = []
    for record in selected:
        qid = str(record.get("question_id") or "")
        if not qid:
            raise ValueError("LongMemEval record has no question_id")
        question = str(record.get("question") or "")
        if not question:
            raise ValueError(f"LongMemEval case {qid} has no question")
        documents = build_documents(
            record,
            max_documents=config.max_documents,
            max_chars=config.max_chars_per_document,
        )
        case = EvalCase(
            case_id=qid,
            group_id=qid,
            question=question,
            gold=str(record.get("answer") or ""),
            category=str(record.get("question_type") or ""),
            question_date=_parse_iso(str(record.get("question_date") or "")),
        )
        units.append(
            IngestUnit(
                group_id=qid,
                documents=documents,
                cases=[case],
                reference_date=_reference_date(documents),
            )
        )
    return units
