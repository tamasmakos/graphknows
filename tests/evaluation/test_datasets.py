"""Dataset loaders: locomo grouping, longmemeval sampling, beam chat parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from evaluation.beam.config import BeamConfig
from evaluation.beam.dataset import (
    build_documents as beam_documents,
)
from evaluation.beam.dataset import (
    extract_questions,
    parse_chat,
    parse_conversation_ranges,
    rubric_nuggets,
)
from evaluation.beam.dataset import (
    load_units as load_beam,
)
from evaluation.locomo.config import LocomoConfig
from evaluation.locomo.dataset import load_units as load_locomo
from evaluation.longmemeval.dataset import (
    build_documents as lme_documents,
)
from evaluation.longmemeval.dataset import (
    sample_questions,
)


def _locomo_row(case_id: str, question: str = "What happened?") -> dict[str, Any]:
    return {
        "id": case_id,
        "category": "temporal",
        "question": question,
        "answer": "7 May 2023",
        "sessions": [
            {
                "session_id": "s1",
                "messages": [
                    {
                        "role": "Caroline",
                        "content": "I went yesterday.",
                        "timestamp": "1:56 pm on 8 May, 2023",
                    }
                ],
            }
        ],
    }


class TestLocomoDataset:
    def test_groups_by_dialogue_and_sets_reference_date(self, tmp_path: Path) -> None:
        rows = [_locomo_row("conv-1-q0001"), _locomo_row("conv-1-q0002"), _locomo_row("conv-2-q0001")]
        path = tmp_path / "questions.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        config = LocomoConfig(data_root=tmp_path)

        units = load_locomo(config, limit=10, offset=0)
        assert [u.group_id for u in units] == ["conv-1", "conv-2"]
        assert len(units[0].cases) == 2
        assert units[0].reference_date == "08 May 2023"
        assert "1:56 pm on 8 May, 2023 | Caroline:" in units[0].documents[0].text

    def test_missing_question_raises(self, tmp_path: Path) -> None:
        row = _locomo_row("conv-1-q0001")
        del row["question"]
        path = tmp_path / "questions.jsonl"
        path.write_text(json.dumps(row), encoding="utf-8")
        with pytest.raises(ValueError, match="no question"):
            load_locomo(LocomoConfig(data_root=tmp_path), limit=1, offset=0)


class TestLongMemEvalDataset:
    _ROWS: ClassVar[list[dict[str, Any]]] = [
        {"question_id": f"q{i}_{t}", "question_type": t, "question": "?", "answer": "a"}
        for t in ("multi-session", "knowledge-update")
        for i in range(4)
    ]

    def test_stratified_sampling_deterministic(self) -> None:
        first = sample_questions(self._ROWS, per_type=2, question_types=None)
        second = sample_questions(self._ROWS, per_type=2, question_types=None)
        assert first == second
        assert len(first) == 4  # 2 per type x 2 types

    def test_type_filter(self) -> None:
        only = sample_questions(self._ROWS, per_type=None, question_types=["multi-session"])
        assert {r["question_type"] for r in only} == {"multi-session"}

    def test_documents_are_date_prefixed_and_sorted(self) -> None:
        row = {
            "haystack_sessions": [
                [{"role": "user", "content": "later session"}],
                [{"role": "user", "content": "earlier session"}],
            ],
            "haystack_dates": ["2023/06/10 (Sat) 10:00", "2023/05/01 (Mon) 21:05"],
            "haystack_session_ids": ["s-later", "s-earlier"],
        }
        docs = lme_documents(row, max_chars=8000)
        assert [d.title for d in docs] == ["s-earlier", "s-later"]
        assert docs[0].text.startswith("[2023/05/01] user:")
        assert docs[0].anchor_date.startswith("2023-05-01")

    def test_oversized_session_splits_without_truncation(self) -> None:
        row = {
            "haystack_sessions": [[{"role": "user", "content": "x" * 300}] * 10],
            "haystack_dates": ["2023/05/01 (Mon) 21:05"],
            "haystack_session_ids": ["s1"],
        }
        docs = lme_documents(row, max_chars=1000)
        assert len(docs) > 1
        total = sum(len(d.text) for d in docs)
        assert total >= 10 * 300  # nothing dropped


class TestBeamDataset:
    def test_parse_chat_2d_list(self) -> None:
        chat = [[{"role": "user", "content": "hi"}], [{"role": "assistant", "content": "yo"}]]
        assert parse_chat(chat) == chat

    def test_parse_chat_batch_dicts(self) -> None:
        chat = [{"turns": [{"role": "user", "content": "hi"}]}]
        assert parse_chat(chat) == [[{"role": "user", "content": "hi"}]]

    def test_parse_chat_plan_format(self) -> None:
        chat = [
            {
                "plan-2": [{"turns": [{"role": "user", "content": "second"}]}],
                "plan-1": [{"turns": [{"role": "user", "content": "first"}]}],
            }
        ]
        batches = parse_chat(chat)
        assert [t[0]["content"] for t in batches] == ["first", "second"]

    def test_documents_carry_time_anchor(self) -> None:
        chat = [[{"role": "user", "content": "hello", "time_anchor": "2024-03-05"}]]
        docs = beam_documents(chat, max_chars=8000)
        assert docs[0].text.startswith("[2024/03/05] user: hello")
        assert docs[0].anchor_date.startswith("2024-03-05")

    def test_extract_questions_and_nuggets(self) -> None:
        conv = {
            "probing_questions": {
                "abstention": [
                    {
                        "question_text": "Did I ever mention a boat?",
                        "rubric": {"nuggets": [{"description": "declines to answer"}]},
                    }
                ],
                "summarization": ["Summarize the project."],
            }
        }
        questions = extract_questions(conv)
        assert {q["question_type"] for q in questions} == {"abstention", "summarization"}
        abstention = next(q for q in questions if q["question_type"] == "abstention")
        assert rubric_nuggets(abstention) == ["declines to answer"]

    def test_conversation_ranges(self) -> None:
        assert parse_conversation_ranges("0-2,5,9-10") == [0, 1, 2, 5, 9, 10]

    def test_load_units_reads_question_and_rubric(self, tmp_path: Path) -> None:
        # Mirrors the real HF schema: field is `question`, rubric is a list of strings.
        rows = [
            {
                "conversation_id": "c0",
                "chat": [[{"role": "user", "content": "hi", "time_anchor": "2024-03-05"}]],
                "probing_questions": {
                    "information_extraction": [
                        {"question": "When does my first sprint end?",
                         "rubric": ["LLM response should state: March 29"]}
                    ]
                },
            }
        ]
        (tmp_path / "beam_100K.json").write_text(json.dumps(rows), encoding="utf-8")
        config = BeamConfig(data_root=tmp_path)
        units = load_beam(
            config, chat_sizes=["100K"], conversations="0",
            question_types=["information_extraction"],
        )
        case = units[0].cases[0]
        assert case.question == "When does my first sprint end?"
        assert case.gold == "LLM response should state: March 29"
        assert case.extras["nuggets"] == ["LLM response should state: March 29"]
