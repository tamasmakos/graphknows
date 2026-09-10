"""The committed repository gold set covers three question kinds (FR-044)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.common.datamodels import EvalCase, IngestUnit
from evaluation.repo.config import RepoConfig
from evaluation.repo.dataset import load_units

KINDS = frozenset({"decision_location", "change_rationale", "test_coverage"})
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def units() -> list[IngestUnit]:
    config = RepoConfig(data_root=REPO_ROOT / "evaluation/data/repo")
    assert (config.data_root / "README.md").exists(), "the gold set records its provenance"
    return load_units(config, limit=1000, offset=0)


@pytest.fixture(scope="module")
def cases(units: list[IngestUnit]) -> list[EvalCase]:
    return [case for unit in units for case in unit.cases]


class TestRepoGoldSet:
    def test_every_kind_is_covered(self, cases: list[EvalCase]) -> None:
        assert {case.category for case in cases} >= KINDS

    def test_no_case_asks_an_unlisted_kind(self, cases: list[EvalCase]) -> None:
        assert {case.category for case in cases} <= KINDS

    def test_every_case_is_named_once_and_answered(self, cases: list[EvalCase]) -> None:
        assert len({case.case_id for case in cases}) == len(cases)
        assert all(case.question and case.gold for case in cases)

    def test_every_case_cites_evidence_its_document_actually_says(
        self, units: list[IngestUnit]
    ) -> None:
        for unit in units:
            text = unit.documents[0].text
            for case in unit.cases:
                assert case.extras["evidence"], f"{case.case_id} cites no evidence"
                for sentence in case.extras["evidence"]:
                    assert sentence in text, f"{case.case_id} cites text {unit.group_id} lacks"

    def test_every_answer_names_a_path_that_exists(self, cases: list[EvalCase]) -> None:
        paths = [word.strip(".,;") for case in cases for word in case.gold.split() if "/" in word]
        assert paths, "answers name the files they point at"
        assert all((REPO_ROOT / path).exists() for path in paths)
