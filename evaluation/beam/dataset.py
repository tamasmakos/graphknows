"""BEAM dataset: HuggingFace download/cache, chat parsing, unit building.

The published dataset stores chat in three shapes (2D turn lists up to 1M,
plan-keyed session dicts for 10M, and ``{"turns": ...}`` batch dicts) and may
store ``probing_questions`` as a Python-repr string — all handled here. Each
conversation becomes one ingest unit; chat batches are split into ≤max_chars
documents (never truncated) with each turn line prefixed by the batch's time
anchor so retrieved chunks keep their temporal anchor.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any

from dateparser import parse as _parse_date

from evaluation.beam.config import CHAT_SIZES, HF_DATASET, HF_DATASET_10M, BeamConfig
from evaluation.common.datamodels import EvalCase, EvalDocument, IngestUnit

log = logging.getLogger(__name__)

QUESTION_TYPES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)


def _parse_probing_questions(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(raw)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def load_split(config: BeamConfig, chat_size: str) -> list[dict[str, Any]]:
    """Load one size bucket, downloading from HuggingFace on first use."""
    if chat_size not in CHAT_SIZES:
        raise ValueError(f"chat_size must be one of {CHAT_SIZES}, got {chat_size!r}")
    cache_path = config.data_root / f"beam_{chat_size}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    try:
        from datasets import load_dataset as hf_load
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The BEAM dataset needs the `datasets` package — install the eval "
            "extras: pip install -e '.[eval]'"
        ) from exc

    log.info("Downloading BEAM %s split from HuggingFace …", chat_size)
    if chat_size == "10M":
        ds = hf_load(HF_DATASET_10M, split="10M")
    else:
        ds = hf_load(HF_DATASET, split=chat_size)

    conversations: list[dict[str, Any]] = []
    for idx, item in enumerate(ds):
        conversations.append(
            {
                "conversation_id": item.get("conversation_id", f"{chat_size}_{idx}"),
                "conversation_seed": item.get("conversation_seed", {}),
                "user_profile": item.get("user_profile", {}),
                "chat": item.get("chat", []),
                "probing_questions": _parse_probing_questions(
                    item.get("probing_questions", "{}")
                ),
            }
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(conversations, ensure_ascii=False), encoding="utf-8")
    log.info("Cached %d conversations → %s", len(conversations), cache_path)
    return conversations


def _unwrap_batch_dicts(batch_dicts: list[dict]) -> list[list[dict]]:
    """Unwrap a list of batch dicts (with ``turns`` key) into flat turn lists."""
    batches: list[list[dict]] = []
    for batch in batch_dicts:
        turns = batch.get("turns", [])
        flat: list[dict] = []
        for item in turns:
            if isinstance(item, list):
                flat.extend(item)
            elif isinstance(item, dict):
                flat.append(item)
        batches.append(flat)
    return batches


def parse_chat(chat_data: Any) -> list[list[dict]]:
    """Parse chat data into batches of turn dicts (all three storage shapes)."""
    if not chat_data:
        return []

    # List of dicts with a "turns" key → unwrap directly.
    if isinstance(chat_data, list) and isinstance(chat_data[0], dict) and "turns" in chat_data[0]:
        return _unwrap_batch_dicts(chat_data)

    # 10M plan-based format: list of session dicts mapping plan keys to batch lists.
    if isinstance(chat_data, list) and isinstance(chat_data[0], dict):
        first_session = chat_data[0]
        sample_val = next(iter(first_session.values()), None)
        is_plan_format = (
            isinstance(sample_val, list)
            and sample_val
            and isinstance(sample_val[0], dict)
            and "turns" in sample_val[0]
        )
        if is_plan_format:
            batches: list[list[dict]] = []
            for session in chat_data:
                if not isinstance(session, dict):
                    continue
                plan_keys = sorted(
                    session.keys(),
                    key=lambda k: int(k.split("-")[-1]) if k.split("-")[-1].isdigit() else 0,
                )
                for plan_key in plan_keys:
                    plan_batches = session[plan_key]
                    if plan_batches:
                        batches.extend(_unwrap_batch_dicts(plan_batches))
            return batches
        # Single flat list of turn dicts.
        if "role" in first_session or "content" in first_session:
            return [chat_data]
        return []

    # Already a 2D list.
    if isinstance(chat_data, list) and isinstance(chat_data[0], list):
        return chat_data

    return []


def _batch_anchor(turns: list[dict]) -> tuple[str, str]:
    """``(iso_anchor, human_prefix)`` from the first turn carrying a time anchor."""
    for turn in turns:
        anchor = turn.get("time_anchor")
        if not anchor:
            continue
        parsed = _parse_date(str(anchor).replace("-", " "))
        if parsed is not None:
            return parsed.isoformat(), parsed.strftime("%Y/%m/%d")
    return "", ""


def _split_lines(lines: list[str], max_chars: int) -> list[str]:
    """Pack lines into ≤max_chars texts without truncating any line."""
    texts: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) + 1 > max_chars:
            texts.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        texts.append("\n".join(current))
    return texts


def build_documents(chat_data: Any, *, max_chars: int) -> list[EvalDocument]:
    """Normalize a conversation's chat into ingestable, date-anchored documents."""
    documents: list[EvalDocument] = []
    for batch_idx, turns in enumerate(parse_chat(chat_data)):
        iso, prefix = _batch_anchor(turns)
        lines = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "user"))
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"[{prefix}] {role}: {content}" if prefix else f"{role}: {content}")
        for part_idx, text in enumerate(_split_lines(lines, max_chars)):
            documents.append(
                EvalDocument(
                    title=f"batch-{batch_idx:04d}-part{part_idx}",
                    text=text,
                    anchor_date=iso,
                )
            )
    return documents


def extract_questions(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the type-keyed ``probing_questions`` dict (10 types x 2)."""
    pq = conversation.get("probing_questions") or {}
    questions: list[dict[str, Any]] = []
    for q_type in QUESTION_TYPES:
        type_questions = pq.get(q_type, [])
        if isinstance(type_questions, dict):
            type_questions = [type_questions]
        if not isinstance(type_questions, list):
            continue
        for q in type_questions:
            if isinstance(q, dict):
                questions.append({**q, "question_type": q_type})
            elif isinstance(q, str):
                questions.append({"question_type": q_type, "question_text": q, "rubric": []})
    return questions


def rubric_nuggets(question: dict[str, Any]) -> list[str]:
    """Rubric nugget descriptions for one question."""
    rubric = question.get("rubric", {})
    if isinstance(rubric, dict):
        nuggets = rubric.get("nuggets", [])
        return [n.get("description", str(n)) if isinstance(n, dict) else str(n) for n in nuggets]
    if isinstance(rubric, list):
        return [str(n) for n in rubric]
    return [str(rubric)] if rubric else []


def parse_conversation_ranges(spec: str) -> list[int]:
    """Parse ``"0-4,10,90-99"`` into a sorted index list."""
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    return sorted(indices)


def load_units(
    config: BeamConfig,
    *,
    chat_sizes: list[str],
    conversations: str,
    question_types: list[str] | None = None,
) -> list[IngestUnit]:
    """Build one ingest unit per selected conversation."""
    wanted_types = set(question_types or QUESTION_TYPES)
    indices = parse_conversation_ranges(conversations)
    units: list[IngestUnit] = []
    for size in chat_sizes:
        rows = load_split(config, size)
        for idx in indices:
            if idx >= len(rows):
                continue
            row = rows[idx]
            group_id = f"{size}-{idx:03d}"
            documents = build_documents(
                row.get("chat"), max_chars=config.max_chars_per_document
            )
            cases = []
            for q_idx, question in enumerate(extract_questions(row)):
                if question["question_type"] not in wanted_types:
                    continue
                nuggets = rubric_nuggets(question)
                cases.append(
                    EvalCase(
                        case_id=f"{group_id}-q{q_idx:02d}-{question['question_type']}",
                        group_id=group_id,
                        question=str(question.get("question") or question.get("question_text", "")),
                        gold=" | ".join(nuggets),
                        category=question["question_type"],
                        extras={
                            "nuggets": nuggets,
                            "difficulty": question.get("difficulty", ""),
                        },
                    )
                )
            reference = max((d.anchor_date for d in documents if d.anchor_date), default="")
            units.append(
                IngestUnit(
                    group_id=group_id,
                    documents=documents,
                    cases=cases,
                    reference_date=reference,
                )
            )
    return units
