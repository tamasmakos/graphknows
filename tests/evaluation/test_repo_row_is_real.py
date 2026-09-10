"""The repository row runs on real agent transcripts, not written-for-it prose (FR-044)."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from evaluation.repo.config import RepoConfig, claude_code_sessions_root
from evaluation.repo.dataset import load_units

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "evaluation/data/repo"
# The fields a harness record carries in the wild; a transcript written for the
# test would have none of them.
HARNESS_FIELDS = frozenset(
    {"uuid", "parentUuid", "sessionId", "cwd", "gitBranch", "isSidechain", "version", "timestamp"}
)


def _records(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture(scope="module")
def fixtures() -> list[Path]:
    transcripts = sorted((DATA_ROOT / "transcripts").glob("*.jsonl"))
    assert len(transcripts) == 1, "one committed fixture keeps CI deterministic"
    return transcripts


@pytest.fixture(scope="module")
def gold() -> list[dict]:
    return _records(DATA_ROOT / "gold.jsonl")


class TestCommittedFixture:
    def test_every_said_record_carries_the_harness_shape(self, fixtures: list[Path]) -> None:
        said = [r for r in _records(fixtures[0]) if r.get("type") in {"user", "assistant"}]
        assert said, "the fixture holds turns to answer from"
        assert all(set(record) >= HARNESS_FIELDS for record in said)
        assert max(len(record) for record in said) >= 16, "the real sixteen-field record shape"

    def test_the_fixture_keeps_the_bookkeeping_records_around_the_turns(
        self, fixtures: list[Path]
    ) -> None:
        types = {record.get("type") for record in _records(fixtures[0])}
        assert types - {"user", "assistant"}, "the loader meets bookkeeping in the wild"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
    )


class TestGoldIsDerivedFromTheTree:
    def test_every_answer_names_a_commit_this_repository_has(self, gold: list[dict]) -> None:
        if _git("rev-parse", "--is-shallow-repository").stdout.strip() != "false":
            pytest.skip("a shallow checkout carries no history to check the answers against")
        for case in gold:
            commit = str(case.get("commit") or "")
            assert commit, f"{case['case_id']} names no commit"
            assert commit in case["answer"], f"{case['case_id']} answers without its commit"
            # Reachability, not mere existence: `cat-file -e` answers yes for any
            # object lying around locally, so a commit that is dangling here and
            # absent from a fresh clone passes on the author's machine and fails
            # in CI. An ancestor of HEAD is what every checkout of this branch has.
            reachable = _git("merge-base", "--is-ancestor", commit, "HEAD")
            assert reachable.returncode == 0, (
                f"{case['case_id']} names {commit}, which is not an ancestor of HEAD -- "
                "a fresh clone of this branch would not have it"
            )


class TestSessionsAreReadAtRunTime:
    def test_a_transcript_that_is_not_committed_comes_from_the_session_directory(
        self, tmp_path: Path, fixtures: list[Path]
    ) -> None:
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        (sessions / "live.jsonl").write_bytes(fixtures[0].read_bytes())
        (tmp_path / "gold.jsonl").write_text(
            json.dumps(
                {
                    "case_id": "live",
                    "transcript": "live.jsonl",
                    "kind": "decision_location",
                    "question": "Where does the decoder live?",
                    "answer": "graphknows/ingestion/extraction/llm/decoder.py",
                }
            ),
            encoding="utf-8",
        )
        config = RepoConfig(data_root=tmp_path, sessions_root=sessions)

        document = load_units(config, limit=1, offset=0)[0].documents[0]

        assert "replaces the core inventory" in document.text

    def test_the_session_directory_defaults_to_the_harness_directory_for_this_checkout(
        self,
    ) -> None:
        root = claude_code_sessions_root()

        assert root.parent == Path.home() / ".claude" / "projects"
        assert root.name.endswith(re.sub(r"[^A-Za-z0-9]", "-", Path.cwd().name))
