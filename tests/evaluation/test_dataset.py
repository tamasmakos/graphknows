"""Unit tests for LoCoMo dataset loading and normalization."""

from __future__ import annotations

import json

from evaluation.config import EvalConfig
from evaluation.ingestion.locomo import build_documents, dialogue_id, load_dialogues

_ROW = {
    "id": "conv-26-q0001",
    "category": "single_hop",
    "question": "Where did Alex go?",
    "answer": "Prague",
    "sessions": [
        {
            "session_id": "conv-26-s1",
            "messages": [
                {"role": "Alex", "content": "Flying to Berlin.", "timestamp": "1 May 2023"},
                {"role": "Alex", "content": "Then Prague."},
            ],
        }
    ],
}


def test_dialogue_id_strips_question_suffix() -> None:
    assert dialogue_id("conv-26-q0001") == "conv-26"
    assert dialogue_id("locomo-smoke-001") == "locomo-smoke-001"


def test_build_documents_joins_messages_and_anchors_date() -> None:
    docs = build_documents(_ROW, max_documents=50, max_chars=4000)
    assert len(docs) == 1
    assert "Alex: Flying to Berlin." in docs[0].text
    assert "Then Prague." in docs[0].text
    assert docs[0].anchor_date.startswith("2023-05-01")


def test_load_dialogues_groups_questions(tmp_path) -> None:
    questions = tmp_path / "questions.jsonl"
    second = {**_ROW, "id": "conv-26-q0002", "question": "When?", "answer": "May"}
    questions.write_text(
        "\n".join(json.dumps(r) for r in (_ROW, second)), encoding="utf-8"
    )
    config = EvalConfig(data_root=tmp_path)

    dialogues = load_dialogues(config, limit=2, offset=0)

    assert len(dialogues) == 1
    assert dialogues[0].dialogue_id == "conv-26"
    assert len(dialogues[0].cases) == 2
    assert len(dialogues[0].documents) == 1
