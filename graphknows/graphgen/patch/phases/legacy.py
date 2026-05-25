"""Phase 2: legacy node migration for unlabeled historical nodes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

from neo4j import Driver

from ..ontology_classifier import classify_name

LEGACY_LABELS = ("Entity", "Chunk", "Topic")

NODE_TYPE_FROM_LABEL = {
    "Entity": "ENTITY_CONCEPT",
    "Chunk": "CHUNK",
    "Topic": "TOPIC",
}

GRAPH_TYPE_FROM_LABEL = {
    "Entity": "entity_relation",
    "Chunk": "lexical_graph",
    "Topic": "topic",
}


@dataclass
class LegacyNode:
    """In-memory representation of a legacy node."""

    internal_id: int
    label: str
    name: str
    node_id: Optional[str]
    node_type: Optional[str]
    graph_type: Optional[str]
    ontology_class: Optional[str]
    legacy_type: Optional[str]


def _fetch_legacy_nodes(driver: Driver, batch_size: int) -> List[LegacyNode]:
    """Fetch a batch of legacy nodes missing node_type."""
    query = """
    MATCH (n)
    WHERE n.node_type IS NULL AND (n:Entity OR n:Chunk OR n:Topic)
    WITH n LIMIT $batch_size
    RETURN
      id(n) AS internal_id,
      labels(n) AS labels,
      coalesce(n.name, '') AS name,
      n.id AS node_id,
      n.node_type AS node_type,
      n.graph_type AS graph_type,
      n.ontology_class AS ontology_class,
      n.type AS legacy_type
    """
    nodes: List[LegacyNode] = []
    with driver.session() as session:
        for row in session.run(query, batch_size=batch_size):
            labels = row["labels"]
            preferred = next((label for label in LEGACY_LABELS if label in labels), None)
            if preferred is None:
                continue
            nodes.append(
                LegacyNode(
                    internal_id=int(row["internal_id"]),
                    label=preferred,
                    name=str(row["name"]),
                    node_id=row["node_id"],
                    node_type=row["node_type"],
                    graph_type=row["graph_type"],
                    ontology_class=row["ontology_class"],
                    legacy_type=row["legacy_type"],
                )
            )
    return nodes


def _derive_ontology_class(node: LegacyNode) -> str:
    """Determine ontology_class from legacy type and name heuristics."""
    if node.legacy_type:
        return str(node.legacy_type).strip().upper()

    if node.label in {"Chunk", "Topic"}:
        if node.label == "Chunk":
            return "STRUCTURAL_CHUNK"
        return "TOPIC"

    guessed = classify_name(node.name)
    if guessed:
        return guessed
    return "CONCEPT"


def _derive_id(name: str, existing_id: Optional[str]) -> str:
    """Generate deterministic 12-char id when missing."""
    if existing_id:
        return str(existing_id)
    digest = hashlib.md5(name.encode("utf-8"), usedforsecurity=False).hexdigest()  # noqa: S324
    return digest[:12]


def _migrate_nodes(driver: Driver, nodes: List[LegacyNode]) -> int:
    """Apply migration updates for a node batch."""
    payload = []
    for node in nodes:
        payload.append(
            {
                "internal_id": node.internal_id,
                "node_type": NODE_TYPE_FROM_LABEL[node.label],
                "node_id": _derive_id(node.name, node.node_id),
                "graph_type": node.graph_type or GRAPH_TYPE_FROM_LABEL[node.label],
                "ontology_class": node.ontology_class or _derive_ontology_class(node),
            }
        )

    query = """
    UNWIND $rows AS row
    MATCH (n)
    WHERE id(n) = row.internal_id
    SET n.node_type = row.node_type
    SET n.id = row.node_id
    SET n.graph_type = row.graph_type
    SET n.ontology_class = row.ontology_class
    RETURN count(n) AS updated
    """
    with driver.session() as session:
        return int(session.run(query, rows=payload).single()["updated"])


def _preview_distribution(nodes: List[LegacyNode]) -> Dict[str, int]:
    """Build ontology class distribution for a node batch."""
    distribution: Dict[str, int] = {}
    for node in nodes:
        value = node.ontology_class or _derive_ontology_class(node)
        distribution[value] = distribution.get(value, 0) + 1
    return distribution


def _legacy_count(driver: Driver) -> int:
    """Count current legacy nodes in scope."""
    query = """
    MATCH (n)
    WHERE n.node_type IS NULL AND (n:Entity OR n:Chunk OR n:Topic)
    RETURN count(n) AS c
    """
    with driver.session() as session:
        return int(session.run(query).single()["c"])


def run_legacy_phase(driver: Driver, dry_run: bool, batch_size: int) -> Dict[str, object]:
    """Run legacy node migration.

    Args:
        driver: Neo4j driver.
        dry_run: If True, no writes are performed.
        batch_size: Maximum processed nodes per batch.

    Returns:
        Phase report dictionary.
    """
    before = _legacy_count(driver)
    if dry_run:
        return {
            "phase": "legacy",
            "before": before,
            "after": 0,
            "updated": before,
            "scanned": before,
            "dry_run": True,
            "skipped": [],
        }

    updated = 0
    scanned = 0
    updated_by_class: Dict[str, int] = {}

    while True:
        nodes = _fetch_legacy_nodes(driver, batch_size)
        if not nodes:
            break

        scanned += len(nodes)
        batch_dist = _preview_distribution(nodes)
        for key, value in batch_dist.items():
            updated_by_class[key] = updated_by_class.get(key, 0) + value
        updated += _migrate_nodes(driver, nodes)

    after = _legacy_count(driver)
    if after < 0:
        after = 0

    return {
        "phase": "legacy",
        "before": before,
        "after": after,
        "updated": updated,
        "scanned": scanned,
        "updated_by_class": updated_by_class,
        "dry_run": dry_run,
        "skipped": [],
    }
