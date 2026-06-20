"""Unified entrypoint for the supported graphknows memory benchmarks."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from evaluation.core.benchmark_registry import (
    BENCHMARK_ORDER,
    SUPPORTED_BENCHMARKS,
    expand_benchmark_selection,
)
from evaluation.core.client import GraphknowsMCPClient
from evaluation.core.file_backed import FileBackedBenchmarkOptions, run_file_backed_benchmark
from evaluation.core.locomo import run_locomo_benchmark
from evaluation.core.longmemeval import run_longmemeval_small
from evaluation.core.reporting import BenchmarkReportPaths, write_benchmark_report


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S")


def _add_common_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--command", default="graphknows-mcp")
    parser.add_argument("--run-id", default=f"eval-{_timestamp()}")
    parser.add_argument("--output-dir", default="evaluation/results")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--no-purge", action="store_true")
    parser.add_argument("--recall-scope", choices=["stm", "ltm", "both"], default="stm")
    parser.add_argument(
        "--promote-to-ltm",
        action="store_true",
        help="Flush each case from STM to LTM before recall; required to exercise llm_assisted relation extraction.",
    )
    parser.add_argument(
        "--relation-extraction-mode",
        choices=["gliner2", "llm_assisted", "off"],
        default="gliner2",
    )

    parser.add_argument("--locomo-data-root", default="evaluation/data/locomo")
    parser.add_argument("--locomo-questions", default="questions.jsonl")
    parser.add_argument("--locomo-max-documents", type=int, default=50)
    parser.add_argument("--locomo-max-chars-per-document", type=int, default=4000)
    parser.add_argument(
        "--generate-answers",
        action="store_true",
        help="Add DSPy answer generation step (retrieve → generate → token F1). "
        "Makes scores paper-comparable. Requires OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--generation-model",
        default="",
        help="OpenRouter model for answer generation (e.g. moonshotai/kimi-k2). "
        "Defaults to OPENROUTER_MODEL env var or moonshotai/kimi-k2.",
    )
    parser.add_argument(
        "--purge-ltm",
        action="store_true",
        help="Purge all LTM data before the eval run starts. Prevents cross-run contamination.",
    )

    parser.add_argument("--longmemeval-data-root", default="evaluation/data/longmemeval-v2")
    parser.add_argument("--longmemeval-question-id", default="")
    parser.add_argument("--longmemeval-max-questions", type=int, default=1)
    parser.add_argument("--longmemeval-max-trajectories", type=int, default=5)
    parser.add_argument("--longmemeval-max-states", type=int, default=8)
    parser.add_argument("--longmemeval-max-chars-per-trajectory", type=int, default=4000)
    parser.add_argument(
        "--longmemeval-ingest-strategy",
        choices=["raw", "trajectory_capsules", "batched_capsules"],
        default="batched_capsules",
    )

    parser.add_argument("--beam-data-root", default="evaluation/data/beam")
    parser.add_argument("--beam-scale", choices=["1m", "10m"], default="1m")
    parser.add_argument("--beam-questions", default="questions.jsonl")
    parser.add_argument("--beam-max-documents", type=int, default=200)
    parser.add_argument("--beam-max-chars-per-document", type=int, default=8000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("list", help="List supported benchmarks.")

    run = subparsers.add_parser("run", help="Run one or more supported benchmarks.")
    run.add_argument(
        "benchmarks",
        nargs="+",
        choices=[*BENCHMARK_ORDER, "all"],
        help="Benchmarks to run; use 'all' for LoCoMo, LongMemEval, and BEAM.",
    )
    _add_common_run_options(run)

    for name in BENCHMARK_ORDER:
        single = subparsers.add_parser(
            name, help=f"Run only {SUPPORTED_BENCHMARKS[name].display_name}."
        )
        single.set_defaults(benchmarks=[name])
        _add_common_run_options(single)

    all_parser = subparsers.add_parser("all", help="Run LoCoMo, LongMemEval, and BEAM.")
    all_parser.set_defaults(benchmarks=["all"])
    _add_common_run_options(all_parser)

    return parser


def _print_supported_benchmarks() -> None:
    for name in BENCHMARK_ORDER:
        spec = SUPPORTED_BENCHMARKS[name]
        categories = ", ".join(spec.categories)
        print(f"{spec.name}: {spec.display_name} ({spec.question_count})")
        print(f"  {spec.description}")
        print(f"  Categories: {categories}")
        print(f"  Data: {spec.data_requirement}")


def _benchmark_run_id(base_run_id: str, benchmark: str, multi: bool) -> str:
    if multi and not base_run_id.endswith(f"-{benchmark}"):
        return f"{base_run_id}-{benchmark}"
    return base_run_id


def _benchmark_output_dir(base_output_dir: str, benchmark: str) -> Path:
    return Path(base_output_dir) / benchmark


def _evaluation_env(relation_extraction_mode: str) -> dict[str, str]:
    env = {"RELATION_EXTRACTION_MODE": relation_extraction_mode}
    # Propagate GRAPHKNOWS_ONTOLOGY_FILE only if already set in the environment.
    # When unset, DSPyExtractor uses its open-domain relation type defaults.
    import os as _os

    if _os.environ.get("GRAPHKNOWS_ONTOLOGY_FILE"):
        env["GRAPHKNOWS_ONTOLOGY_FILE"] = _os.environ["GRAPHKNOWS_ONTOLOGY_FILE"]
    return env


def _resolve_file_backed_root(
    data_root: str,
    questions_file: str,
    fixture_root: str,
) -> tuple[Path, str]:
    root = Path(data_root)
    if (root / questions_file).exists():
        return root, "local_data"
    fallback = Path(fixture_root)
    if (fallback / questions_file).exists():
        print(f"{root / questions_file} not found; using smoke fixture {fallback / questions_file}")
        return fallback, "smoke_fixture"
    return root, "missing"


async def _run_locomo(args: argparse.Namespace, *, run_id: str) -> BenchmarkReportPaths:
    data_root, data_source = _resolve_file_backed_root(
        args.locomo_data_root,
        args.locomo_questions,
        "evaluation/fixtures/locomo",
    )
    async with GraphknowsMCPClient(
        args.command,
        request_timeout_s=900,
        env=_evaluation_env(args.relation_extraction_mode),
    ) as client:
        cases = await run_locomo_benchmark(
            client,
            FileBackedBenchmarkOptions(
                benchmark="locomo",
                data_root=data_root,
                questions_file=args.locomo_questions,
                run_id=run_id,
                top_k=args.top_k,
                offset=args.offset,
                limit=args.limit,
                max_documents=args.locomo_max_documents,
                max_chars_per_document=args.locomo_max_chars_per_document,
                recall_scope=args.recall_scope,
                promote_to_ltm=args.promote_to_ltm,
                session_prefix="eval-locomo",
                purge=not args.no_purge,
                relation_extraction_mode=args.relation_extraction_mode,
                generate_answers=getattr(args, "generate_answers", False),
                generation_model=getattr(args, "generation_model", ""),
                pre_run_purge_ltm=getattr(args, "purge_ltm", False),
            ),
        )
    return write_benchmark_report(
        benchmark="locomo",
        run_id=run_id,
        output_dir=_benchmark_output_dir(args.output_dir, "locomo"),
        cases=[case.to_record() for case in cases],
        metadata={
            "benchmark": SUPPORTED_BENCHMARKS["locomo"].display_name,
            "categories": list(SUPPORTED_BENCHMARKS["locomo"].categories),
            "data_root": str(data_root),
            "data_source": data_source,
            "questions_file": args.locomo_questions,
            "recall_scope": args.recall_scope,
            "promote_to_ltm": args.promote_to_ltm,
            "relation_extraction_mode": args.relation_extraction_mode,
        },
    )


_LME_QUESTION_TYPES = frozenset(
    {
        "single_session_user",
        "single_session_assistant",
        "single_session_preference",
        "knowledge_update",
        "temporal_reasoning",
        "multi_session",
    }
)


def _lme_data_is_valid(data_root: Path) -> bool:
    """Return True only when data_root contains proper LME conversational-memory questions."""
    import json as _json

    questions_path = data_root / "questions.jsonl"
    haystacks_path = data_root / "haystacks" / "lme_v2_small.json"
    if not questions_path.exists() or not haystacks_path.exists():
        return False
    try:
        first_line = next(
            (
                line
                for line in questions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
            None,
        )
        if not first_line:
            return False
        row = _json.loads(first_line)
        return str(row.get("question_type", "")).lower() in _LME_QUESTION_TYPES
    except Exception:
        return False


async def _run_longmemeval(args: argparse.Namespace, *, run_id: str) -> BenchmarkReportPaths:
    data_root = Path(args.longmemeval_data_root)
    use_lme_runner = _lme_data_is_valid(data_root)
    if not use_lme_runner:
        fixture_root = Path("evaluation/fixtures/longmemeval")
        if fixture_root.exists() and (fixture_root / "questions.jsonl").exists():
            print(
                f"{data_root} does not contain valid LME conversational-memory data; "
                f"using smoke fixture {fixture_root / 'questions.jsonl'}"
            )
            data_root = fixture_root
        else:
            print(
                f"WARNING: {data_root} does not contain valid LME data "
                f"and no fixture found at {fixture_root}"
            )

    async with GraphknowsMCPClient(
        args.command,
        request_timeout_s=900,
        env=_evaluation_env(args.relation_extraction_mode),
    ) as client:
        if use_lme_runner:
            cases = await run_longmemeval_small(
                client,
                data_root=data_root,
                question_id=args.longmemeval_question_id or None,
                max_questions=args.longmemeval_max_questions,
                run_id=run_id,
                top_k=args.top_k,
                max_trajectories=args.longmemeval_max_trajectories,
                max_states=args.longmemeval_max_states,
                max_chars_per_trajectory=args.longmemeval_max_chars_per_trajectory,
                ingest_strategy=args.longmemeval_ingest_strategy,
                recall_scope=args.recall_scope,
                promote_to_ltm=args.promote_to_ltm,
                purge=not args.no_purge,
            )
        else:
            cases = await run_file_backed_benchmark(
                client,
                FileBackedBenchmarkOptions(
                    benchmark="longmemeval",
                    data_root=data_root,
                    questions_file="questions.jsonl",
                    run_id=run_id,
                    top_k=args.top_k,
                    offset=args.offset,
                    limit=args.longmemeval_max_questions,
                    max_documents=args.longmemeval_max_trajectories,
                    max_chars_per_document=args.longmemeval_max_chars_per_trajectory,
                    recall_scope=args.recall_scope,
                    promote_to_ltm=args.promote_to_ltm,
                    session_prefix="eval-longmemeval",
                    purge=not args.no_purge,
                    relation_extraction_mode=args.relation_extraction_mode,
                ),
            )
    return write_benchmark_report(
        benchmark="longmemeval",
        run_id=run_id,
        output_dir=_benchmark_output_dir(args.output_dir, "longmemeval"),
        cases=[case.to_record() for case in cases],
        metadata={
            "benchmark": SUPPORTED_BENCHMARKS["longmemeval"].display_name,
            "categories": list(SUPPORTED_BENCHMARKS["longmemeval"].categories),
            "data_root": str(data_root),
            "ingest_strategy": args.longmemeval_ingest_strategy,
            "recall_scope": args.recall_scope,
            "promote_to_ltm": args.promote_to_ltm,
            "relation_extraction_mode": args.relation_extraction_mode,
        },
    )


async def _run_beam(args: argparse.Namespace, *, run_id: str) -> BenchmarkReportPaths:
    data_root, data_source = _resolve_file_backed_root(
        str(Path(args.beam_data_root) / args.beam_scale),
        args.beam_questions,
        f"evaluation/fixtures/beam/{args.beam_scale}",
    )
    async with GraphknowsMCPClient(
        args.command,
        request_timeout_s=1800,
        env=_evaluation_env(args.relation_extraction_mode),
    ) as client:
        cases = await run_file_backed_benchmark(
            client,
            FileBackedBenchmarkOptions(
                benchmark="beam",
                data_root=data_root,
                questions_file=args.beam_questions,
                run_id=run_id,
                top_k=args.top_k,
                offset=args.offset,
                limit=args.limit,
                max_documents=args.beam_max_documents,
                max_chars_per_document=args.beam_max_chars_per_document,
                recall_scope=args.recall_scope,
                promote_to_ltm=args.promote_to_ltm,
                session_prefix=f"eval-beam-{args.beam_scale}",
                purge=not args.no_purge,
                relation_extraction_mode=args.relation_extraction_mode,
            ),
        )
    return write_benchmark_report(
        benchmark="beam",
        run_id=run_id,
        output_dir=_benchmark_output_dir(args.output_dir, "beam"),
        cases=[case.to_record() for case in cases],
        metadata={
            "benchmark": SUPPORTED_BENCHMARKS["beam"].display_name,
            "categories": list(SUPPORTED_BENCHMARKS["beam"].categories),
            "scale": args.beam_scale,
            "data_root": str(data_root),
            "data_source": data_source,
            "questions_file": args.beam_questions,
            "recall_scope": args.recall_scope,
            "promote_to_ltm": args.promote_to_ltm,
            "relation_extraction_mode": args.relation_extraction_mode,
        },
    )


async def _run_one(name: str, args: argparse.Namespace, *, run_id: str) -> BenchmarkReportPaths:
    if name == "locomo":
        return await _run_locomo(args, run_id=run_id)
    if name == "longmemeval":
        return await _run_longmemeval(args, run_id=run_id)
    if name == "beam":
        return await _run_beam(args, run_id=run_id)
    raise ValueError(f"unsupported benchmark: {name}")


async def _run_selected(args: argparse.Namespace) -> int:
    selected = expand_benchmark_selection(list(args.benchmarks))
    multi = len(selected) > 1
    for name in selected:
        run_id = _benchmark_run_id(args.run_id, name, multi=multi)
        paths = await _run_one(name, args, run_id=run_id)
        print(paths.json_path)
    return 0


async def amain(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "list":
        _print_supported_benchmarks()
        return 0
    return await _run_selected(args)


def main() -> None:
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
