"""Repository-transcript loading: harness JSONL in, ingest units out.

Each gold record names the transcript it is answered from, so questions sharing
a transcript ingest it once (``group_id`` is the transcript's stem). The files
are agent-harness JSONL — the shape
:mod:`graphknows.ingestion.parsers.transcript` reads — flattened here to the
``role``/``content``/``timestamp`` messages the eval harness ingests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.common.datamodels import EvalCase, EvalDocument, IngestUnit
from evaluation.common.dataset import clip, message_turns, reference_date
from evaluation.repo.config import RepoConfig

# Record types that carry something said. Everything else in a transcript
# (summaries, system notices) is harness bookkeeping, not evidence.
_KEPT_RECORD_TYPES = frozenset({"user", "assistant"})


def _block_text(block: Any) -> str:
    """What one content block says, or '' when it says nothing citable."""
    if isinstance(block, str):
        return block.strip()
    if not isinstance(block, dict):
        return ""
    return str(block.get("text") or block.get("thinking") or "").strip()


def _message(record: dict[str, Any]) -> dict[str, str] | None:
    """One transcript record as a chat message, or ``None`` when it holds none.

    A shape this does not recognise is skipped rather than raised on: a
    transcript is an append-only log written by a versioned harness, so an
    unknown record must cost one message, never the row.
    """
    if record.get("type") not in _KEPT_RECORD_TYPES:
        return None
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    blocks = [content] if isinstance(content, str) else content
    if not isinstance(blocks, list):
        return None
    text = "\n".join(said for block in blocks if (said := _block_text(block)))
    if not text:
        return None
    return {
        "role": str(message.get("role") or record["type"]),
        "content": text,
        "timestamp": str(record.get("timestamp") or ""),
    }


def _messages(path: Path) -> list[dict[str, str]]:
    """Every citable message of a transcript, in order."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist (repository transcript missing).")
    messages: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and (msg := _message(record)):
            messages.append(msg)
    return messages


def _transcript_path(config: RepoConfig, transcript: str) -> Path:
    """Where a named transcript is read from.

    The committed fixture keeps CI deterministic; every other transcript is the
    user's own session, read from the harness's directory at run time so no
    session content is committed.
    """
    committed = config.data_root / transcript
    return committed if committed.exists() else config.sessions_root / transcript


def _document(config: RepoConfig, transcript: str) -> EvalDocument:
    """One transcript as an ingestable document (turn-fed when configured)."""
    messages = _messages(_transcript_path(config, transcript))
    anchor = messages[0]["timestamp"] if messages else ""
    text = clip(
        "\n".join(f"{m['role']}: {m['content']}" for m in messages), config.max_chars_per_document
    )
    turns = message_turns(messages, timestamp=anchor) if config.turn_fed else []
    return EvalDocument(title=Path(transcript).stem, text=text, anchor_date=anchor, turns=turns)


def _case(record: dict[str, Any], group_id: str) -> EvalCase:
    """One gold record as a benchmark question, keyed by its question kind."""
    case_id = str(record.get("case_id") or "")
    question = str(record.get("question") or "")
    if not (case_id and question):
        raise ValueError(f"repo gold record {record!r} needs a case_id and a question")
    return EvalCase(
        case_id=case_id,
        group_id=group_id,
        question=question,
        gold=str(record.get("answer") or ""),
        category=str(record.get("kind") or ""),
        extras={"evidence": [str(e) for e in (record.get("evidence") or [])]},
    )


def _load_gold(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist (repository gold set missing).")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    return [record for record in records if isinstance(record, dict)]


def load_units(config: RepoConfig, *, limit: int, offset: int) -> list[IngestUnit]:
    """Load the selected question window; one ingest unit per transcript."""
    records = _load_gold(config.data_root / config.questions_file)[offset : offset + limit]

    units: dict[str, IngestUnit] = {}
    for record in records:
        transcript = str(record.get("transcript") or "")
        if not transcript:
            raise ValueError(f"repo gold case {record.get('case_id')!r} names no transcript")
        group_id = Path(transcript).stem
        if group_id not in units:
            document = _document(config, transcript)
            units[group_id] = IngestUnit(
                group_id=group_id,
                documents=[document],
                reference_date=reference_date([document]),
            )
        units[group_id].cases.append(_case(record, group_id))
    return list(units.values())
