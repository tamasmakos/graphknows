"""Dataset loaders: locomo grouping and reference-date handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.locomo.config import LocomoConfig
from evaluation.locomo.dataset import load_units as load_locomo
from evaluation.longmem.config import LongMemConfig
from evaluation.longmem.dataset import load_units as load_longmem


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
        rows = [
            _locomo_row("conv-1-q0001"),
            _locomo_row("conv-1-q0002"),
            _locomo_row("conv-2-q0001"),
        ]
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


def _longmem_record(qid: str = "q1") -> dict[str, Any]:
    return {
        "question_id": qid,
        "question_type": "single-session-user",
        "question": "What degree did I graduate with?",
        "question_date": "2023/05/30 (Tue) 23:40",
        "answer": "Business Administration",
        "answer_session_ids": ["s2"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/25 (Thu) 10:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "just chatting about the weather"}],
            [
                {
                    "role": "user",
                    "content": "I graduated with Business Administration.",
                    "has_answer": True,
                },
                {"role": "assistant", "content": "Congrats on the degree!", "has_answer": True},
            ],
        ],
    }


class TestLongMemDataset:
    def _write(self, tmp_path: Path, records: list[dict[str, Any]]) -> LongMemConfig:
        path = tmp_path / "longmemeval_s_cleaned.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        return LongMemConfig(data_root=tmp_path)

    def test_one_unit_per_question_with_evidence_and_reference_date(self, tmp_path: Path) -> None:
        config = self._write(tmp_path, [_longmem_record("q1")])
        units = load_longmem(config, limit=10, offset=0)

        assert len(units) == 1
        unit = units[0]
        assert unit.group_id == "q1"
        assert len(unit.cases) == 1
        case = unit.cases[0]
        # has_answer messages become turn-level gold evidence (whitespace-normalized).
        assert case.extras["evidence"] == [
            "I graduated with Business Administration.",
            "Congrats on the degree!",
        ]
        assert case.category == "single-session-user"
        # Reference date is the newest haystack session.
        assert unit.reference_date == "25 May 2023"

    def test_no_has_answer_flags_yields_empty_evidence(self, tmp_path: Path) -> None:
        record = _longmem_record("q2")
        for session in record["haystack_sessions"]:
            for msg in session:
                msg.pop("has_answer", None)
        units = load_longmem(self._write(tmp_path, [record]), limit=1, offset=0)
        assert units[0].cases[0].extras["evidence"] == []
