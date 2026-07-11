"""Dataset loaders: locomo grouping and reference-date handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.locomo.config import LocomoConfig
from evaluation.locomo.dataset import load_units as load_locomo


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
