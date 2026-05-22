"""Phase 1: deterministic structural node classification."""

from __future__ import annotations

from typing import Dict, List

from neo4j import Driver

STRUCTURAL_MAPPINGS = {
    "CHUNK": "STRUCTURAL_CHUNK",
    # SEGMENT is now the temporal pipeline-run container (E6); old text-subdivision
    # nodes that were written as SEGMENT before the PASSAGE rename are back-compat mapped
    # to STRUCTURAL_PASSAGE so the patch can normalise legacy data.
    "SEGMENT": "STRUCTURAL_PASSAGE",
    # New label for text subdivisions (formerly SEGMENT)
    "PASSAGE": "STRUCTURAL_PASSAGE",
    "DOC": "STRUCTURAL_DOCUMENT",
    "TOPIC": "TOPIC",
    "SUBTOPIC": "SUBTOPIC",
}


def _collect_count(driver: Driver, node_type: str, ontology_class: str) -> int:
    """Count structural nodes still missing ontology_class for a type."""
    query = """
    MATCH (n)
    WHERE n.node_type = $node_type AND n.ontology_class IS NULL
    RETURN count(n) AS c
    """
    with driver.session() as session:
        return int(
            session.run(query, node_type=node_type, ontology_class=ontology_class).single()["c"]
        )


def _update_batch(driver: Driver, node_type: str, ontology_class: str, batch_size: int) -> int:
    """Update one batch and return updated node count."""
    query = """
    MATCH (n)
    WHERE n.node_type = $node_type AND n.ontology_class IS NULL
    WITH n LIMIT $batch_size
    SET n.ontology_class = $ontology_class
    RETURN count(n) AS updated
    """
    with driver.session() as session:
        return int(
            session.run(
                query,
                node_type=node_type,
                ontology_class=ontology_class,
                batch_size=batch_size,
            ).single()["updated"]
        )


def run_structural_phase(driver: Driver, dry_run: bool, batch_size: int) -> Dict[str, object]:
    """Run deterministic structural node classification.

    Args:
        driver: Neo4j driver.
        dry_run: If True, no writes are performed.
        batch_size: Maximum updates per transaction.

    Returns:
        Phase report dictionary.
    """
    before: Dict[str, int] = {}
    updated_by_type: Dict[str, int] = {}
    skipped: List[str] = []

    for node_type, ontology_class in STRUCTURAL_MAPPINGS.items():
        pending = _collect_count(driver, node_type, ontology_class)
        before[node_type] = pending
        if pending == 0:
            skipped.append(node_type)
            updated_by_type[node_type] = 0
            continue

        if dry_run:
            updated_by_type[node_type] = pending
            continue

        total_updated = 0
        while True:
            updated = _update_batch(driver, node_type, ontology_class, batch_size)
            total_updated += updated
            if updated == 0:
                break
        updated_by_type[node_type] = total_updated

    after: Dict[str, int] = {}
    for node_type, ontology_class in STRUCTURAL_MAPPINGS.items():
        if dry_run:
            after[node_type] = 0
        else:
            after[node_type] = _collect_count(driver, node_type, ontology_class)

    total_before = sum(before.values())
    total_updated = sum(updated_by_type.values())
    total_after = sum(after.values())

    return {
        "phase": "structural",
        "before": total_before,
        "after": total_after,
        "updated": total_updated,
        "updated_by_type": updated_by_type,
        "dry_run": dry_run,
        "skipped": skipped,
    }
