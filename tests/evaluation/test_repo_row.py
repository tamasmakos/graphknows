"""The repository-transcript row: harness JSONL in, pipeline-shaped units out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.common.baseline import ROW_MODULES
from evaluation.common.datamodels import EvalCase, IngestUnit, RetrievedPassage
from evaluation.common.pipeline import BenchmarkAdapter
from evaluation.repo.adapter import RepoAdapter
from evaluation.repo.config import RepoConfig
from evaluation.repo.dataset import load_units

_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost_usd": 0.0}


class FakeLLM:
    """Returns scripted completions in order; records every prompt."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    async def generate(self, system: str, user: str, **_: Any) -> tuple[str, dict]:
        self.prompts.append((system, user))
        return self._replies.pop(0), dict(_USAGE)


class FakeJudge:
    """Scripted two-stage judge: returns queued scores, records (q, gold, gen)."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self.calls: list[tuple[str, str, str]] = []

    async def judge_two_stage(
        self, question: str, gold: str, generated: str
    ) -> tuple[float, dict[str, Any]]:
        self.calls.append((question, gold, generated))
        score = self._scores.pop(0)
        return score, {"semantic_f1": score, "judge_reasoning": "reason"}


@pytest.fixture(autouse=True)
def _no_judge_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHKNOWS_JUDGE_CACHE", "0")


def _transcript_record(role: str, text: str, timestamp: str) -> dict[str, Any]:
    return {
        "type": role,
        "timestamp": timestamp,
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    }


def _corpus(tmp_path: Path, gold: list[dict[str, Any]]) -> RepoConfig:
    """A data root holding one transcript plus the given gold records."""
    lines = [
        json.dumps(_transcript_record("user", "Where does resolve live?", "2025-01-02T10:00:00Z")),
        "{ not json",  # a half-written line costs one message, not the row
        json.dumps({"type": "summary", "summary": "bookkeeping, not evidence"}),
        json.dumps(
            _transcript_record(
                "assistant", "Settled: resolve() lives in identity.py.", "2025-01-02T10:01:00Z"
            )
        ),
    ]
    (tmp_path / "session-a.jsonl").write_text("\n".join(lines), encoding="utf-8")
    (tmp_path / "gold.jsonl").write_text(
        "\n".join(json.dumps(record) for record in gold), encoding="utf-8"
    )
    return RepoConfig(data_root=tmp_path)


def _gold_record(case_id: str, kind: str = "decision_location") -> dict[str, Any]:
    return {
        "case_id": case_id,
        "transcript": "session-a.jsonl",
        "kind": kind,
        "question": "Where was resolve() placed?",
        "answer": "identity.py",
        "evidence": ["Settled: resolve() lives in identity.py."],
    }


def _case() -> EvalCase:
    return EvalCase(
        case_id="c1",
        group_id="session-a",
        question="Where was resolve() placed?",
        gold="identity.py",
        category="decision_location",
    )


class TestRepoDataset:
    def test_questions_sharing_a_transcript_ingest_it_once(self, tmp_path: Path) -> None:
        config = _corpus(tmp_path, [_gold_record("c1"), _gold_record("c2", "change_rationale")])

        units = load_units(config, limit=10, offset=0)

        assert [unit.group_id for unit in units] == ["session-a"]
        unit = units[0]
        assert len(unit.documents) == 1
        assert [case.case_id for case in unit.cases] == ["c1", "c2"]
        assert [case.category for case in unit.cases] == ["decision_location", "change_rationale"]

    def test_document_keeps_said_turns_and_drops_harness_bookkeeping(self, tmp_path: Path) -> None:
        config = _corpus(tmp_path, [_gold_record("c1")])

        document = load_units(config, limit=1, offset=0)[0].documents[0]

        assert "user: Where does resolve live?" in document.text
        assert "Settled: resolve() lives in identity.py." in document.text
        assert "bookkeeping" not in document.text
        assert document.anchor_date == "2025-01-02T10:00:00Z"

    def test_evidence_travels_to_the_case_for_evidence_recall(self, tmp_path: Path) -> None:
        config = _corpus(tmp_path, [_gold_record("c1")])

        case = load_units(config, limit=1, offset=0)[0].cases[0]

        assert case.extras["evidence"] == ["Settled: resolve() lives in identity.py."]

    def test_window_slices_the_gold_questions(self, tmp_path: Path) -> None:
        config = _corpus(tmp_path, [_gold_record("c1"), _gold_record("c2")])

        units = load_units(config, limit=1, offset=1)

        assert [case.case_id for unit in units for case in unit.cases] == ["c2"]

    def test_missing_gold_set_is_reported_as_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="gold set missing"):
            load_units(RepoConfig(data_root=tmp_path), limit=1, offset=0)


class TestRepoAdapter:
    def test_runs_through_the_shared_pipeline_and_joins_the_panel(self) -> None:
        assert issubclass(RepoAdapter, BenchmarkAdapter)
        assert ROW_MODULES["repo"] == "evaluation.repo.cli"

    @pytest.mark.asyncio
    async def test_answer_extracts_after_marker(self) -> None:
        answerer = FakeLLM(["Thinking...\nANSWER: identity.py"])
        adapter = RepoAdapter(RepoConfig(), answerer, FakeJudge([]))
        unit = IngestUnit(group_id="session-a", reference_date="2 January 2025")
        passages = [RetrievedPassage(text="Settled: resolve() lives in identity.py.")]

        answer, usage = await adapter.answer(_case(), unit, passages)

        assert answer == "identity.py"
        assert usage["gen_total_tokens"] == 15
        prompt = answerer.prompts[0][1]
        assert "2 January 2025" in prompt and "Where was resolve() placed?" in prompt

    @pytest.mark.asyncio
    async def test_judge_scores_map_to_labels(self) -> None:
        judge = FakeJudge([1.0, 0.0])
        adapter = RepoAdapter(RepoConfig(), FakeLLM([]), judge)

        score, metrics = await adapter.judge_case(_case(), "identity.py")
        assert score == 1.0 and metrics["judge_label"] == "CORRECT"
        assert metrics["semantic_f1"] == 1.0

        score, metrics = await adapter.judge_case(_case(), "retriever.py")
        assert score == 0.0 and metrics["judge_label"] == "WRONG"
        assert judge.calls[0][1] == "identity.py"

    def test_lexical_metrics_score_the_gold_string(self) -> None:
        adapter = RepoAdapter(RepoConfig(), FakeLLM([]), FakeJudge([]))

        assert adapter.lexical_metrics(_case(), "identity.py")["token_f1"] == 1.0
