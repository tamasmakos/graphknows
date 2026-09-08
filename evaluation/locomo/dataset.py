"""LoCoMo dataset loading and document normalization.

Reads the converted LoCoMo JSONL (see ``evaluation/scripts/convert_locomo.py``),
groups questions by dialogue, and normalizes each dialogue's sessions into
``EvalDocument`` objects ready for ingestion.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dateparser import parse as _parse_date

from evaluation.common.datamodels import EvalCase, EvalDocument, IngestUnit
from evaluation.common.dataset import clip, reference_date
from evaluation.locomo.config import LocomoConfig

_CASE_ID_KEYS = ("id", "question_id", "case_id")
_QUESTION_KEYS = ("question", "query", "prompt")
_CATEGORY_KEYS = ("category", "type", "task_type", "question_type")


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read benchmark rows from a JSONL file (one JSON object per line)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run scripts/convert_locomo.py first.")
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"{path} must contain JSONL rows or a JSON array of objects")


def first_string(row: dict[str, Any], keys: tuple[str, ...], default: str = "") -> str:
    """Return the first present, non-null key value coerced to str."""
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return default


def dialogue_id(case_id: str) -> str:
    """Extract the dialogue prefix from a LoCoMo case id.

    ``"conv-26-q0001"`` → ``"conv-26"``; an id without a ``-qNNNN`` suffix is
    its own group.
    """
    idx = case_id.rfind("-q")
    if idx > 0 and case_id[idx + 2 :].isdigit():
        return case_id[:idx]
    return case_id


def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""
    role = first_string(message, ("role", "speaker", "author"), "message")
    content = first_string(message, ("content", "text", "utterance", "message"))
    timestamp = first_string(message, ("timestamp", "created_at", "time"))
    prefix = f"{timestamp} | {role}" if timestamp else role
    return f"{prefix}: {content}" if content else ""


def _session_anchor(session: dict[str, Any]) -> str:
    """ISO anchor datetime from the first timestamped message in a session.

    Used as the ingest ``anchor_date`` so relative dates ("yesterday", "last
    Saturday") resolve against the conversation's own time. Returns "" when no
    parseable timestamp is present.
    """
    for msg in session.get("messages", []):
        if not isinstance(msg, dict):
            continue
        ts = first_string(msg, ("timestamp", "created_at", "time"))
        if not ts:
            continue
        dt = _parse_date(ts)
        return dt.isoformat() if dt else ""
    return ""


_EVIDENCE_ID = re.compile(r"D(\d+):(\d+)")


def _evidence_turns(row: dict[str, Any]) -> list[str]:
    """Gold evidence turn CONTENTS for a LoCoMo row (``['D1:3']`` -> the utterance).

    LoCoMo ships the turn-ids that answer each question (1536/1540 rows, 27% of
    them multi-hop) and nothing used them. ``D<doc>:<turn>`` is 1-indexed on
    both axes. Contents — not the "ts | Speaker:" prefix — because the content
    is what survives chunking verbatim (verified: 97/97 evidence turns are an
    exact substring of some CHUNK).

    Scoring only: these never reach the retriever or the answerer.
    """
    # str(list) works for both the list and the stringified-list dumps.
    ids = [
        (int(m.group(1)), int(m.group(2)))
        for m in _EVIDENCE_ID.finditer(str(row.get("evidence") or ""))
    ]
    sessions = row.get("sessions")
    if not isinstance(sessions, list):
        return []
    out: list[str] = []
    for doc_no, turn_no in ids:
        try:
            content = sessions[doc_no - 1]["messages"][turn_no - 1].get("content")
        except (IndexError, KeyError, TypeError, AttributeError):
            continue
        if text := " ".join(str(content or "").split()):
            out.append(text)
    return out


def _session_turns(session: dict[str, Any]) -> list[dict[str, str]]:
    """A session's messages as ingestable turns, structure intact.

    ``name`` and ``timestamp`` travel as FIELDS rather than being flattened into
    the content: the speaker becomes ``CHUNK.speaker`` and anchors the turn's
    first-person clauses, and the timestamp is what its relative dates resolve
    against. Flattening either is what mints greetings as entities and dates a
    2023 conversation to ingestion time.
    """
    turns: list[dict[str, str]] = []
    for msg in session.get("messages", []):
        if not isinstance(msg, dict):
            continue
        content = first_string(msg, ("content", "text", "utterance", "message")).strip()
        if not content:
            continue
        turns.append(
            {
                # LoCoMo speakers are people, not chat roles; the participant
                # name goes to `name`, which is what becomes CHUNK.speaker.
                "role": "user",
                "content": content,
                "name": first_string(msg, ("speaker", "role", "author")),
                "timestamp": first_string(msg, ("timestamp", "created_at", "time")),
            }
        )
    return turns


def build_documents(
    row: dict[str, Any], *, max_documents: int, max_chars: int, turn_fed: bool = False
) -> list[EvalDocument]:
    """Normalize a LoCoMo row's ``sessions`` into ingestable units.

    ``turn_fed`` keeps each utterance separate so ingestion runs through the
    buffer-then-flush path a live agent uses; otherwise the session is joined
    into one document, as the original harness did.
    """
    sessions = row.get("sessions")
    if not isinstance(sessions, list):
        return []
    documents: list[EvalDocument] = []
    for index, session in enumerate(sessions[:max_documents]):
        if not isinstance(session, dict):
            continue
        title = first_string(session, ("title", "id", "session_id"), f"session-{index}")
        text = "\n".join(filter(None, (_message_text(msg) for msg in session.get("messages", []))))
        text = clip(text, max_chars)
        turns = _session_turns(session) if turn_fed else []
        if text or turns:
            documents.append(
                EvalDocument(
                    title=title,
                    text=text,
                    anchor_date=_session_anchor(session),
                    turns=turns,
                )
            )
    return documents


def load_units(config: LocomoConfig, *, limit: int, offset: int) -> list[IngestUnit]:
    """Load the selected question window and group it into dialogues.

    Questions are grouped by dialogue id, preserving order. Each dialogue's
    documents are built once from its first question's row (all questions in a
    LoCoMo dialogue embed the same sessions).
    """
    rows = read_rows(config.data_root / config.questions_file)
    selected = rows[offset : offset + limit]

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        dlg = dialogue_id(first_string(row, _CASE_ID_KEYS, ""))
        groups.setdefault(dlg, []).append(row)

    units: list[IngestUnit] = []
    for dlg_id, dlg_rows in groups.items():
        documents = build_documents(
            dlg_rows[0],
            max_documents=config.max_documents,
            max_chars=config.max_chars_per_document,
            turn_fed=config.turn_fed,
        )
        cases = [
            EvalCase(
                case_id=first_string(r, _CASE_ID_KEYS, f"{dlg_id}-unknown"),
                group_id=dlg_id,
                question=first_string(r, _QUESTION_KEYS),
                gold=str(r.get("answer") or r.get("gold") or r.get("target") or ""),
                category=first_string(r, _CATEGORY_KEYS),
                extras={"evidence": _evidence_turns(r)},
            )
            for r in dlg_rows
        ]
        for case in cases:
            if not case.question:
                raise ValueError(f"LoCoMo case {case.case_id} has no question/query/prompt")
        units.append(
            IngestUnit(
                group_id=dlg_id,
                documents=documents,
                cases=cases,
                reference_date=reference_date(documents),
            )
        )
    return units
