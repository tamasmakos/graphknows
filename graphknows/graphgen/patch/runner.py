"""Patch runner orchestration for ontology backfill phases."""

from __future__ import annotations

import argparse
from typing import Callable, Dict, Iterable, List

from neo4j import GraphDatabase

from .phases import (
    run_legacy_phase,
    run_structural_phase,
    run_unknown_phase,
    run_gliner2_fallback_phase,
)
from .report import format_patch_report

DEFAULT_PHASE_ORDER = ["structural", "legacy", "unknown", "gliner2_fallback"]

PhaseRunner = Callable[[object, bool, int], Dict[str, object]]

PHASE_RUNNERS: Dict[str, PhaseRunner] = {
    "structural": run_structural_phase,
    "legacy": run_legacy_phase,
    "unknown": run_unknown_phase,
    "gliner2_fallback": run_gliner2_fallback_phase,
}


def _normalize_phases(phases: list[str] | None) -> List[str]:
    if not phases:
        return list(DEFAULT_PHASE_ORDER)

    requested = [phase.strip().lower() for phase in phases if phase and phase.strip()]
    unknown = [phase for phase in requested if phase not in PHASE_RUNNERS]
    if unknown:
        raise ValueError(f"Unknown patch phases: {', '.join(sorted(set(unknown)))}")

    requested_set = set(requested)
    return [phase for phase in DEFAULT_PHASE_ORDER if phase in requested_set]


def _distribution_from_result(phase_result: Dict[str, object]) -> Dict[str, int]:
    distribution = phase_result.get("updated_by_class")
    if isinstance(distribution, dict):
        return {str(k): int(v) for k, v in distribution.items()}

    updated_by_type = phase_result.get("updated_by_type")
    if isinstance(updated_by_type, dict):
        return {str(k): int(v) for k, v in updated_by_type.items()}

    if phase_result.get("phase") == "unknown":
        return {
            "reclassified": int(phase_result.get("reclassified", 0) or 0),
            "left_unknown": int(phase_result.get("left_unknown", 0) or 0),
        }

    return {}


def run_patch(
    neo4j_uri: str = "bolt://172.18.0.1:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "testpassword123",
    phases: list[str] | None = None,
    dry_run: bool = False,
    batch_size: int = 500,
    taxonomy=None,
) -> dict:
    """Run selected ontology patch phases in deterministic order.

    Args:
        neo4j_uri: Bolt URI for Neo4j.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
        phases: Subset of phase names to run (all phases if None).
        dry_run: Preview changes without writing to Neo4j.
        batch_size: Max nodes processed per batch per phase.
        taxonomy: Optional ``TaxonomyConfig`` passed to phases that support it
            (currently: ``legacy``, ``gliner2_fallback``).
    """
    selected_phases = _normalize_phases(phases)
    summary: Dict[str, object] = {
        "dry_run": dry_run,
        "batch_size": batch_size,
        "phases": [],
        "totals": {"before": 0, "after": 0, "updated": 0},
    }

    with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
        for phase_name in selected_phases:
            runner = PHASE_RUNNERS[phase_name]
            # Pass taxonomy to phases that accept it
            import inspect

            sig = inspect.signature(runner)
            if "taxonomy" in sig.parameters:
                phase_result = runner(
                    driver, dry_run=dry_run, batch_size=batch_size, taxonomy=taxonomy
                )
            else:
                phase_result = runner(driver, dry_run=dry_run, batch_size=batch_size)
            phase_result["ontology_class_distribution"] = _distribution_from_result(phase_result)
            summary["phases"].append(phase_result)
            summary["totals"]["before"] += int(phase_result.get("before", 0) or 0)
            summary["totals"]["after"] += int(phase_result.get("after", 0) or 0)
            summary["totals"]["updated"] += int(phase_result.get("updated", 0) or 0)

    return summary


def _parse_phase_list(raw_phases: str) -> list[str]:
    return [token.strip() for token in raw_phases.split(",") if token.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ontology patch phases.")
    parser.add_argument("--neo4j-uri", default="bolt://172.18.0.1:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="testpassword123")
    parser.add_argument(
        "--phases",
        default=",".join(DEFAULT_PHASE_ORDER),
        help="Comma-separated phases (structural,legacy,unknown)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    parser.add_argument("--batch-size", type=int, default=500)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    result = run_patch(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        phases=_parse_phase_list(args.phases),
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )
    print(format_patch_report(result))
