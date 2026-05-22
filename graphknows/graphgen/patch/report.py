"""Human-readable reporting helpers for patch execution."""

from __future__ import annotations

from typing import Dict, List


def _format_distribution(distribution: Dict[str, int]) -> str:
    if not distribution:
        return "none"
    parts = [f"{key}={value}" for key, value in sorted(distribution.items())]
    return ", ".join(parts)


def format_patch_report(summary: dict) -> str:
    """Format run_patch summary into line-oriented text report."""
    dry_run = bool(summary.get("dry_run", False))
    action_label = "would update" if dry_run else "updated"

    lines: List[str] = []
    lines.append(f"Patch run (dry_run={dry_run}, batch_size={summary.get('batch_size')})")

    for phase in summary.get("phases", []):
        phase_name = phase.get("phase", "unknown")
        before = int(phase.get("before", 0) or 0)
        after = int(phase.get("after", 0) or 0)
        updated = int(phase.get("updated", 0) or 0)
        distribution = phase.get("ontology_class_distribution")
        if not isinstance(distribution, dict):
            distribution = {}

        lines.append(f"- {phase_name}: {action_label}={updated}, before={before}, after={after}")
        lines.append(f"  ontology_class distribution: {_format_distribution(distribution)}")

    totals = summary.get("totals", {})
    lines.append(
        "Totals: "
        f"{action_label}={int(totals.get('updated', 0) or 0)}, "
        f"before={int(totals.get('before', 0) or 0)}, "
        f"after={int(totals.get('after', 0) or 0)}"
    )

    return "\n".join(lines)
