"""LongMemEval dataset: download, sampling, and haystack normalization.

Each question row carries its own ``haystack_sessions`` (lists of user/assistant
turns) with per-session dates. One question = one ingest unit. Every turn line
is prefixed with the session date so retrieved chunks keep their temporal
anchor after server-side chunking.
"""

from __future__ import annotations

import json
import random
from typing import Any

from dateparser import parse as _parse_date

from evaluation.common.datamodels import EvalCase, EvalDocument, IngestUnit
from evaluation.common.download import download_file
from evaluation.longmemeval.config import LongMemEvalConfig

QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)

_SAMPLING_SEED = 42  # fixed so a --per-type slice is the same set across runs


def load_dataset(config: LongMemEvalConfig) -> list[dict[str, Any]]:
    """Load (downloading on first use) the 500-question dataset."""
    path = config.data_root / config.dataset_file
    if not path.exists():
        download_file(config.dataset_url, path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) < 500:
        raise ValueError(f"{path} looks truncated ({len(rows) if isinstance(rows, list) else 0} rows; expected ≥500)")
    return rows


def sample_questions(
    rows: list[dict[str, Any]],
    *,
    per_type: int | None,
    question_types: list[str] | None,
) -> list[dict[str, Any]]:
    """Stratified per-type sample (deterministic), optionally type-filtered.

    ``per_type=None`` keeps every question of the selected types.
    """
    wanted = set(question_types or QUESTION_TYPES)
    filtered = [r for r in rows if r.get("question_type") in wanted]
    if per_type is None:
        return sorted(filtered, key=lambda r: str(r.get("question_id")))
    rng = random.Random(_SAMPLING_SEED)
    selected: list[dict[str, Any]] = []
    for qtype in sorted(wanted):
        of_type = [r for r in filtered if r.get("question_type") == qtype]
        take = min(per_type, len(of_type))
        selected.extend(rng.sample(of_type, take))
    return sorted(selected, key=lambda r: str(r.get("question_id")))


def _session_date_label(raw: str) -> tuple[str, str]:
    """Return ``(iso_anchor, human_prefix)`` for a haystack date string."""
    parsed = _parse_date(raw or "")
    if parsed is None:
        return "", ""
    return parsed.isoformat(), parsed.strftime("%Y/%m/%d")


def _split_chunks(lines: list[str], max_chars: int) -> list[str]:
    """Split turn lines into ≤max_chars texts without truncating any line."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def build_documents(row: dict[str, Any], *, max_chars: int) -> list[EvalDocument]:
    """One (or more, when oversized) document per haystack session, date-sorted."""
    sessions = row.get("haystack_sessions") or []
    dates = row.get("haystack_dates") or []
    session_ids = row.get("haystack_session_ids") or []

    entries: list[tuple[str, str, str, list[Any]]] = []  # (iso, prefix, sid, turns)
    for index, turns in enumerate(sessions):
        raw_date = str(dates[index]) if index < len(dates) else ""
        iso, prefix = _session_date_label(raw_date)
        sid = str(session_ids[index]) if index < len(session_ids) else f"session-{index}"
        entries.append((iso, prefix, sid, turns if isinstance(turns, list) else []))
    entries.sort(key=lambda e: e[0])  # chronological ingestion, like real usage

    documents: list[EvalDocument] = []
    for iso, prefix, sid, turns in entries:
        lines = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "user"))
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"[{prefix}] {role}: {content}" if prefix else f"{role}: {content}")
        if not lines:
            continue
        for part_idx, text in enumerate(_split_chunks(lines, max_chars)):
            title = sid if part_idx == 0 else f"{sid}-part{part_idx + 1}"
            documents.append(EvalDocument(title=title, text=text, anchor_date=iso))
    return documents


def load_units(
    config: LongMemEvalConfig,
    *,
    per_type: int | None,
    question_types: list[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[IngestUnit]:
    """Build one ingest unit per sampled question."""
    rows = sample_questions(
        load_dataset(config), per_type=per_type, question_types=question_types
    )
    if limit is not None:
        rows = rows[offset : offset + limit]

    units: list[IngestUnit] = []
    for row in rows:
        qid = str(row.get("question_id"))
        question_date = str(row.get("question_date", ""))
        case = EvalCase(
            case_id=qid,
            group_id=qid,
            question=str(row.get("question", "")),
            gold=str(row.get("answer", "")),
            category=str(row.get("question_type", "")),
            question_date=question_date,
            extras={"is_abstention": qid.endswith("_abs")},
        )
        units.append(
            IngestUnit(
                group_id=qid,
                documents=build_documents(row, max_chars=config.max_chars_per_document),
                cases=[case],
                reference_date=question_date,
            )
        )
    return units
