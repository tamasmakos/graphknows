"""Tests for the public evaluation CLI surface."""

from __future__ import annotations

import pytest

from evaluation.run_eval import _resolve_file_backed_root, build_parser


def test_run_parser_accepts_multiple_supported_benchmarks() -> None:
    args = build_parser().parse_args(["run", "locomo", "beam", "--limit", "2"])

    assert args.action == "run"
    assert args.benchmarks == ["locomo", "beam"]
    assert args.limit == 2


def test_individual_benchmark_subcommands_are_shorthand() -> None:
    args = build_parser().parse_args(
        ["longmemeval", "--longmemeval-max-questions", "3", "--promote-to-ltm"]
    )

    assert args.action == "longmemeval"
    assert args.benchmarks == ["longmemeval"]
    assert args.longmemeval_max_questions == 3
    assert args.promote_to_ltm is True


def test_old_benchmark_names_are_not_public_cli_choices() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "memoryarena"])


def test_file_backed_defaults_fall_back_to_tracked_smoke_fixtures(tmp_path) -> None:
    data_root, data_source = _resolve_file_backed_root(
        str(tmp_path / "missing-locomo"),
        "questions.jsonl",
        "evaluation/fixtures/locomo",
    )

    assert data_source == "smoke_fixture"
    assert data_root.as_posix().endswith("evaluation/fixtures/locomo")
