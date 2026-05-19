"""Phase 3: UNKNOWN ontology class reclassification."""

from __future__ import annotations

from typing import Dict, List

from neo4j import Driver

from ..ontology_classifier import classify_name


def _unknown_count(driver: Driver) -> int:
    """Count unknown ENTITY_CONCEPT nodes."""
    query = """
    MATCH (n)
    WHERE n.node_type = 'ENTITY_CONCEPT' AND n.ontology_class = 'UNKNOWN'
    RETURN count(n) AS c
    """
    with driver.session() as session:
        return int(session.run(query).single()["c"])


def _fetch_unknown_nodes(driver: Driver, batch_size: int) -> List[Dict[str, object]]:
    """Fetch a batch of unknown nodes to classify."""
    query = """
    MATCH (n)
    WHERE n.node_type = 'ENTITY_CONCEPT' AND n.ontology_class = 'UNKNOWN'
    WITH n LIMIT $batch_size
    RETURN id(n) AS internal_id, coalesce(n.name, '') AS name
    """
    with driver.session() as session:
        return [
            {"internal_id": int(row["internal_id"]), "name": str(row["name"])}
            for row in session.run(query, batch_size=batch_size)
        ]


def _fetch_unknown_names_page(driver: Driver, batch_size: int, offset: int) -> List[str]:
    """Fetch names of unknown nodes using stable pagination."""
    query = """
    MATCH (n)
    WHERE n.node_type = 'ENTITY_CONCEPT' AND n.ontology_class = 'UNKNOWN'
    RETURN coalesce(n.name, '') AS name
    ORDER BY id(n)
    SKIP $offset
    LIMIT $batch_size
    """
    with driver.session() as session:
        return [
            str(row["name"]) for row in session.run(query, offset=offset, batch_size=batch_size)
        ]


def _apply_unknown_updates(driver: Driver, rows: List[Dict[str, object]]) -> int:
    """Apply classification results to unknown node batch."""
    query = """
    UNWIND $rows AS row
    MATCH (n)
    WHERE id(n) = row.internal_id
    SET n.ontology_class = row.ontology_class
    SET n.classification_confidence = row.classification_confidence
    RETURN count(n) AS updated
    """
    with driver.session() as session:
        return int(session.run(query, rows=rows).single()["updated"])


def run_unknown_phase(driver: Driver, dry_run: bool, batch_size: int) -> Dict[str, object]:
    """Run UNKNOWN node reclassification phase.

    Args:
        driver: Neo4j driver.
        dry_run: If True, no writes are performed.
        batch_size: Maximum nodes per transaction.

    Returns:
        Phase report dictionary.
    """
    before = _unknown_count(driver)
    if dry_run:
        reclassified = 0
        left_unknown = 0
        offset = 0
        while True:
            names = _fetch_unknown_names_page(driver, batch_size=batch_size, offset=offset)
            if not names:
                break
            offset += len(names)
            for name in names:
                result = classify_name(name, default_to_concept=False)
                if result is None:
                    left_unknown += 1
                else:
                    reclassified += 1
        return {
            "phase": "unknown",
            "before": before,
            "after": left_unknown,
            "updated": before,
            "reclassified": reclassified,
            "left_unknown": left_unknown,
            "dry_run": True,
            "skipped": [],
        }

    updated = 0
    reclassified = 0
    left_unknown = 0

    while True:
        nodes = _fetch_unknown_nodes(driver, batch_size)
        if not nodes:
            break

        rows: List[Dict[str, object]] = []
        for node in nodes:
            result = classify_name(str(node["name"]), default_to_concept=False)
            if result is None:
                rows.append(
                    {
                        "internal_id": node["internal_id"],
                        "ontology_class": "UNCLASSIFIED",
                        "classification_confidence": "low",
                    }
                )
                left_unknown += 1
            else:
                rows.append(
                    {
                        "internal_id": node["internal_id"],
                        "ontology_class": result,
                        "classification_confidence": "rule_based",
                    }
                )
                reclassified += 1

        updated += _apply_unknown_updates(driver, rows)

    after = _unknown_count(driver)
    return {
        "phase": "unknown",
        "before": before,
        "after": after,
        "updated": updated,
        "reclassified": reclassified,
        "left_unknown": left_unknown,
        "dry_run": False,
        "skipped": [],
    }
