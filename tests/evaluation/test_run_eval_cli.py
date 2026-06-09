"""Tests for the public evaluation CLI surface."""

from __future__ import annotations

import pytest

from evaluation.run_eval import build_parser


def test_run_parser_accepts_multiple_supported_benchmarks() -> None:
    args = build_parser().parse_args(["run", "locomo", "beam", "--limit", "2"])

    assert args.action == "run"
    assert args.benchmarks == ["locomo", "beam"]
    assert args.limit == 2


def test_individual_benchmark_subcommands_are_shorthand() -> None:
    args = build_parser().parse_args(["longmemeval", "--longmemeval-max-questions", "3"])

    assert args.action == "longmemeval"
    assert args.benchmarks == ["longmemeval"]
    assert args.longmemeval_max_questions == 3


def test_old_benchmark_names_are_not_public_cli_choices() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "memoryarena"])
