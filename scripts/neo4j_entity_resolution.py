"""
Neo4j Entity Resolution Script.

This script performs entity resolution directly on the Neo4j graph:
1. Merges known EU/European Union alias nodes into canonical forms.
2. Applies BlockingResolver (string similarity) to find additional aliases.
3. Normalises RELATION predicate labels to canonical forms.

Usage:
    PYTHONPATH=/root/graphknows python scripts/neo4j_entity_resolution.py

Google-style docstrings, type hints, no bare except, 100-char lines.
"""

import logging
import sys
from typing import Dict, List, Tuple

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError

# Import existing resolution machinery
sys.path.insert(0, "/root/graphknows/services")
sys.path.insert(0, "/root/graphknows/services/graphknows")
from graphgen.pipeline.graph_cleaning.resolution import BlockingResolver, EntityRecord

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEO4J_URI = "bolt://172.18.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword123"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("neo4j_entity_resolution")

# ---------------------------------------------------------------------------
# Task A-1: Static known alias map
# All keys (aliases) → canonical names
# ---------------------------------------------------------------------------

KNOWN_ALIASES: Dict[str, str] = {
    # EU family
    "EU": "European Union",
    "the EU": "European Union",
    "the Union": "European Union",
    "European union": "European Union",
    # Member States
    "EU member states": "EU Member States",
    "Member States": "EU Member States",
    "the Member States": "EU Member States",
    # Parliament
    "MEPs": "Members of the European Parliament",
    "MEP": "Members of the European Parliament",
    "EP": "European Parliament",
    # Commission
    "EC": "European Commission",
    "the Commission": "European Commission",
    # EEAS / HR
    "EEAS": "European External Action Service",
    "HR/VP": "High Representative / Vice President",
    # People
    "von der Leyen": "Ursula von der Leyen",
    "President von der Leyen": "Ursula von der Leyen",
    # NATO
    "North Atlantic Treaty Organization": "NATO",
}

# ---------------------------------------------------------------------------
# Task B: Predicate normalisation map
# ---------------------------------------------------------------------------

PREDICATE_MAP: Dict[str, str] = {
    "RECEIVES_ACTION_PLAN_FROM": "RECEIVES_FROM",
    "GETS_PLAN_FROM": "RECEIVES_FROM",
    "IS_GIVEN_PLAN_BY": "RECEIVES_FROM",
    "SUPPORTS_WITH": "SUPPORTS",
    "EXPRESSES_SUPPORT_FOR": "SUPPORTS",
    "REAFFIRMS_SUPPORT_FOR": "SUPPORTS",
    "CONTINUES_TO_SUPPORT": "SUPPORTS",
    "CALLS_ON": "CALLS_FOR",
    "URGES": "CALLS_FOR",
    "WELCOMES_THE": "WELCOMES",
    "COMMITS_TO": "COMMITTED_TO",
    "PLEDGES_TO": "COMMITTED_TO",
    "IMPOSED": "IMPOSES",
    "HAS_IMPOSED": "IMPOSES",
    "IN_NEGOTIATIONS_WITH": "NEGOTIATES_WITH",
    "WORKS_WITH": "COOPERATES_WITH",
    "DISCUSSED": "DISCUSSES",
    "OPPOSED": "OPPOSES",
    "CONDEMNED": "CONDEMNS",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_entities(driver: Driver) -> List[Dict[str, str]]:
    """Load all Entity nodes from Neo4j.

    Args:
        driver: Active Neo4j driver.

    Returns:
        List of dicts with keys 'name' and 'label'.
    """
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN e.name AS name, e.label AS label")
        return [{"name": r["name"], "label": r["label"] or ""} for r in result if r["name"]]


def build_blocking_aliases(
    entities: List[Dict[str, str]],
    threshold: float = 0.88,
) -> Dict[str, str]:
    """Run BlockingResolver to find string-similar alias pairs.

    Uses only the name+label (no embeddings) for the first pass.

    Args:
        entities: List of {'name': str, 'label': str} dicts from Neo4j.
        threshold: Similarity threshold (default 0.88).

    Returns:
        Mapping alias_name → canonical_name (longest name in cluster).
    """
    records: List[EntityRecord] = [
        EntityRecord(id=e["name"], text=e["name"], type=e["label"])
        for e in entities
    ]

    resolver = BlockingResolver(similarity_threshold=threshold)
    resolver.add_records(records)
    resolved_map: Dict[str, str] = resolver.resolve()  # original_id -> canonical_text

    aliases: Dict[str, str] = {}
    for original, canonical in resolved_map.items():
        if original != canonical:
            aliases[original] = canonical
            logger.info(f"  BlockingResolver: '{original}' → '{canonical}'")

    logger.info(f"BlockingResolver found {len(aliases)} alias pairs.")
    return aliases


def merge_alias_maps(*maps: Dict[str, str]) -> Dict[str, str]:
    """Merge multiple alias→canonical dicts; KNOWN_ALIASES takes priority.

    Args:
        *maps: Alias maps to merge (earlier maps take priority).

    Returns:
        Combined alias map.
    """
    combined: Dict[str, str] = {}
    for m in reversed(maps):  # lower-priority first
        combined.update(m)
    return combined


# ---------------------------------------------------------------------------
# Neo4j merge operations
# ---------------------------------------------------------------------------


def node_exists(session, name: str) -> bool:
    """Check whether an Entity node with the given name exists."""
    result = session.run(
        "MATCH (e:Entity {name: $name}) RETURN count(e) AS n", name=name
    )
    return result.single()["n"] > 0


def redirect_edges(session, alias: str, canonical: str) -> int:
    """Redirect all edges from the alias node to the canonical node.

    Args:
        session: Neo4j session.
        alias: Name of the alias Entity node to dissolve.
        canonical: Name of the canonical Entity node to keep.

    Returns:
        Total number of edge operations performed.
    """
    ops = 0

    # --- Incoming MENTIONS (Chunk → alias) ---
    res = session.run(
        """
        MATCH (c:Chunk)-[r:MENTIONS]->(alias:Entity {name: $alias})
        MATCH (canon:Entity {name: $canonical})
        MERGE (c)-[:MENTIONS]->(canon)
        DELETE r
        RETURN count(r) AS n
        """,
        alias=alias,
        canonical=canonical,
    )
    ops += res.single()["n"]

    # --- Outgoing CO_OCCURS (alias → b) ---
    res = session.run(
        """
        MATCH (alias:Entity {name: $alias})-[r:CO_OCCURS]->(b:Entity)
        MATCH (canon:Entity {name: $canonical})
        MERGE (canon)-[:CO_OCCURS {weight: r.weight, via_chunks: r.via_chunks}]->(b)
        DELETE r
        RETURN count(r) AS n
        """,
        alias=alias,
        canonical=canonical,
    )
    ops += res.single()["n"]

    # --- Incoming CO_OCCURS (b → alias) ---
    res = session.run(
        """
        MATCH (b:Entity)-[r:CO_OCCURS]->(alias:Entity {name: $alias})
        MATCH (canon:Entity {name: $canonical})
        MERGE (b)-[:CO_OCCURS {weight: r.weight, via_chunks: r.via_chunks}]->(canon)
        DELETE r
        RETURN count(r) AS n
        """,
        alias=alias,
        canonical=canonical,
    )
    ops += res.single()["n"]

    # --- Outgoing RELATION (alias → b) ---
    res = session.run(
        """
        MATCH (alias:Entity {name: $alias})-[r:RELATION]->(b:Entity)
        MATCH (canon:Entity {name: $canonical})
        MERGE (canon)-[:RELATION {predicate: r.predicate, chunk_id: r.chunk_id}]->(b)
        DELETE r
        RETURN count(r) AS n
        """,
        alias=alias,
        canonical=canonical,
    )
    ops += res.single()["n"]

    # --- Incoming RELATION (b → alias) ---
    res = session.run(
        """
        MATCH (b:Entity)-[r:RELATION]->(alias:Entity {name: $alias})
        MATCH (canon:Entity {name: $canonical})
        MERGE (b)-[:RELATION {predicate: r.predicate, chunk_id: r.chunk_id}]->(canon)
        DELETE r
        RETURN count(r) AS n
        """,
        alias=alias,
        canonical=canonical,
    )
    ops += res.single()["n"]

    # --- Outgoing IN_COMMUNITY (alias → t) ---
    res = session.run(
        """
        MATCH (alias:Entity {name: $alias})-[r:IN_COMMUNITY]->(t:Topic)
        MATCH (canon:Entity {name: $canonical})
        MERGE (canon)-[:IN_COMMUNITY]->(t)
        DELETE r
        RETURN count(r) AS n
        """,
        alias=alias,
        canonical=canonical,
    )
    ops += res.single()["n"]

    return ops


def delete_alias_node(session, alias: str) -> None:
    """Delete the alias Entity node (must have no remaining edges).

    Args:
        session: Neo4j session.
        alias: Name of the Entity node to delete.
    """
    session.run("MATCH (e:Entity {name: $alias}) DETACH DELETE e", alias=alias)


def apply_alias_map(driver: Driver, alias_map: Dict[str, str]) -> Dict[str, int]:
    """Apply the alias→canonical map to Neo4j.

    For each pair: redirect all edges, then delete the alias node.

    Args:
        driver: Active Neo4j driver.
        alias_map: Mapping alias_name → canonical_name.

    Returns:
        Stats dict with keys 'merged', 'skipped', 'edges_redirected'.
    """
    stats = {"merged": 0, "skipped": 0, "edges_redirected": 0}

    with driver.session() as session:
        for alias, canonical in alias_map.items():
            if alias == canonical:
                continue
            if not node_exists(session, alias):
                logger.debug(f"  Alias node not found, skipping: '{alias}'")
                stats["skipped"] += 1
                continue
            if not node_exists(session, canonical):
                logger.info(
                    f"  Canonical '{canonical}' not in graph; skipping alias '{alias}'"
                )
                stats["skipped"] += 1
                continue

            logger.info(f"  Merging '{alias}' → '{canonical}'")
            try:
                n_edges = redirect_edges(session, alias, canonical)
                stats["edges_redirected"] += n_edges
                delete_alias_node(session, alias)
                stats["merged"] += 1
                logger.info(f"     {n_edges} edge(s) redirected, alias node deleted")
            except Neo4jError as exc:
                logger.error(f"     Neo4j error merging '{alias}': {exc}")

    return stats


# ---------------------------------------------------------------------------
# Task B: Predicate normalisation
# ---------------------------------------------------------------------------


def normalise_predicates(driver: Driver, predicate_map: Dict[str, str]) -> int:
    """Rename RELATION predicates to canonical forms.

    Args:
        driver: Active Neo4j driver.
        predicate_map: Mapping old_predicate → new_predicate.

    Returns:
        Total number of relationship properties updated.
    """
    total_updated = 0
    with driver.session() as session:
        for old, new in predicate_map.items():
            result = session.run(
                """
                MATCH ()-[r:RELATION]->()
                WHERE r.predicate = $old
                SET r.predicate = $new
                RETURN count(r) AS updated
                """,
                old=old,
                new=new,
            )
            n = result.single()["updated"]
            if n:
                logger.info(f"  Predicate '{old}' → '{new}': {n} relation(s) updated")
                total_updated += n

    logger.info(f"Predicate normalisation complete. Total updated: {total_updated}")
    return total_updated


# ---------------------------------------------------------------------------
# Task C: Verification
# ---------------------------------------------------------------------------


def verify_results(driver: Driver) -> None:
    """Print verification stats after entity resolution.

    Args:
        driver: Active Neo4j driver.
    """
    with driver.session() as session:
        # EU / European Union edges
        eu_edges = session.run(
            """
            MATCH (e:Entity {name: 'European Union'})-[r]-()
            RETURN type(r) AS rel_type, count(r) AS n
            """
        ).data()
        logger.info(f"'European Union' edges: {eu_edges}")

        # Total entity count
        total = session.run("MATCH (n:Entity) RETURN count(n) AS n").single()["n"]
        logger.info(f"Total Entity nodes after resolution: {total}")

        # Sample RELATION triplets
        sample = session.run(
            """
            MATCH (a:Entity)-[r:RELATION]->(b:Entity)
            RETURN a.name AS src, r.predicate AS pred, b.name AS dst
            LIMIT 15
            """
        ).data()
        logger.info("Sample RELATION triplets:")
        for row in sample:
            logger.info(f"  ({row['src']})-[{row['pred']}]->({row['dst']})")

        # Edge type summary
        edge_summary = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS n ORDER BY n DESC"
        ).data()
        logger.info(f"Edge type summary: {edge_summary}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Neo4j entity resolution script."""
    logger.info("=" * 70)
    logger.info("Neo4j Entity Resolution — starting")
    logger.info("=" * 70)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        # ------------------------------------------------------------------
        # Task A-1: Load entities
        # ------------------------------------------------------------------
        logger.info("Loading Entity nodes from Neo4j …")
        entities = load_entities(driver)
        logger.info(f"  Loaded {len(entities)} entity nodes.")

        entity_names = {e["name"] for e in entities}

        # ------------------------------------------------------------------
        # Task A-2: Static alias map — filter to aliases that actually exist
        # ------------------------------------------------------------------
        static_aliases: Dict[str, str] = {
            alias: canon
            for alias, canon in KNOWN_ALIASES.items()
            if alias in entity_names and alias != canon
        }
        logger.info(f"Static KNOWN_ALIASES applicable: {len(static_aliases)}")

        # ------------------------------------------------------------------
        # Task A-3: BlockingResolver for additional string-similar aliases
        # ------------------------------------------------------------------
        logger.info("Running BlockingResolver (threshold=0.88) …")
        blocking_aliases = build_blocking_aliases(entities, threshold=0.88)

        # ------------------------------------------------------------------
        # Task A-4: Merge alias maps (static takes priority)
        # ------------------------------------------------------------------
        # merge_alias_maps: later maps override earlier; we want static priority
        full_alias_map = merge_alias_maps(static_aliases, blocking_aliases)
        # But we need static to win, so overlay static on top:
        full_alias_map.update(static_aliases)

        logger.info(f"Combined alias map size: {len(full_alias_map)} pairs")

        # ------------------------------------------------------------------
        # Apply alias merges to Neo4j
        # ------------------------------------------------------------------
        logger.info("Applying alias merges to Neo4j …")
        merge_stats = apply_alias_map(driver, full_alias_map)
        logger.info(f"Merge stats: {merge_stats}")

        # ------------------------------------------------------------------
        # Task B: Predicate normalisation
        # ------------------------------------------------------------------
        logger.info("Normalising RELATION predicates …")
        pred_updated = normalise_predicates(driver, PREDICATE_MAP)
        logger.info(f"  {pred_updated} predicate(s) normalised.")

        # ------------------------------------------------------------------
        # Task C: Verify
        # ------------------------------------------------------------------
        logger.info("Verifying results …")
        verify_results(driver)

        logger.info("=" * 70)
        logger.info("Entity resolution complete ")
        logger.info("=" * 70)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
