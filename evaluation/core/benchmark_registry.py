"""Registry for the supported graphknows memory benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    """Metadata for one supported evaluation benchmark."""

    name: str
    display_name: str
    question_count: str
    categories: tuple[str, ...]
    description: str
    data_requirement: str


SUPPORTED_BENCHMARKS: dict[str, BenchmarkSpec] = {
    "locomo": BenchmarkSpec(
        name="locomo",
        display_name="LoCoMo",
        question_count="1,540",
        categories=("single-hop", "multi-hop", "open-domain", "temporal memory recall"),
        description=(
            "Multi-session conversational memory recall benchmark for reproducible "
            "single-hop, multi-hop, open-domain, and temporal questions."
        ),
        data_requirement="local LoCoMo JSON/JSONL export under evaluation/data/locomo",
    ),
    "longmemeval": BenchmarkSpec(
        name="longmemeval",
        display_name="LongMemEval",
        question_count="500",
        categories=(
            "single-session user recall",
            "single-session assistant recall",
            "single-session preference recall",
            "knowledge update",
            "temporal reasoning",
            "multi-session recall",
        ),
        description=(
            "LongMemEval-V2 memory scenarios, including demanding knowledge update "
            "and multi-session recall tasks."
        ),
        data_requirement=(
            "LongMemEval-V2 data under evaluation/data/longmemeval-v2; missing files "
            "are downloaded by the existing adapter"
        ),
    ),
    "beam": BenchmarkSpec(
        name="beam",
        display_name="BEAM",
        question_count="1M and 10M token scales",
        categories=(
            "preference following",
            "instruction following",
            "information extraction",
            "knowledge update",
            "multi-session reasoning",
            "summarization",
            "temporal reasoning",
            "event ordering",
            "abstention",
            "contradiction resolution",
        ),
        description=(
            "Production-scale memory benchmark for context volumes that cannot be "
            "solved by expanding the context window alone."
        ),
        data_requirement="local BEAM JSON/JSONL export under evaluation/data/beam/{1m,10m}",
    ),
}

BENCHMARK_ORDER: tuple[str, ...] = ("locomo", "longmemeval", "beam")


def expand_benchmark_selection(selection: list[str]) -> list[str]:
    """Expand CLI benchmark selections into ordered, de-duplicated benchmark names."""
    names = BENCHMARK_ORDER if "all" in selection else tuple(selection)
    expanded: list[str] = []
    for name in names:
        if name not in SUPPORTED_BENCHMARKS:
            raise ValueError(f"unsupported benchmark: {name}")
        if name not in expanded:
            expanded.append(name)
    return expanded
