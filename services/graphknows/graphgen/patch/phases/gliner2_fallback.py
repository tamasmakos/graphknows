"""Phase: gliner2_fallback — reclassify UNCLASSIFIED nodes with GLiNER2.

Nodes left with ``ontology_class='UNCLASSIFIED'`` after the structural,
legacy, and unknown phases are run through GLiNER2 NER on their *name*
field.  The top-scoring label is written back together with metadata
(``classification_method='gliner2'``, ``classification_confidence``).

This phase is intentionally last in the default phase order so that earlier,
cheaper classifiers get first pass.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from neo4j import Driver

if TYPE_CHECKING:
    from graphgen.config.taxonomy import TaxonomyConfig

logger = logging.getLogger(__name__)

_BATCH_QUERY = """
MATCH (n)
WHERE n.ontology_class = 'UNCLASSIFIED'
  AND n.name IS NOT NULL
RETURN n.name AS name, elementId(n) AS eid
LIMIT $batch_size
"""

_UPDATE_QUERY = """
MATCH (n)
WHERE elementId(n) = $eid
SET n.ontology_class = $ontology_class,
    n.classification_method = 'gliner2',
    n.classification_confidence = $confidence
"""

_COUNT_QUERY = """
MATCH (n) WHERE n.ontology_class = 'UNCLASSIFIED' RETURN count(n) AS c
"""

_DIST_QUERY = """
MATCH (n)
WHERE n.ontology_class IS NOT NULL AND n.ontology_class <> 'UNCLASSIFIED'
  AND n.classification_method = 'gliner2'
RETURN n.ontology_class AS cls, count(n) AS cnt
"""


def run_gliner2_fallback_phase(
    driver: Driver,
    dry_run: bool,
    batch_size: int,
    taxonomy: "Optional[TaxonomyConfig]" = None,
) -> Dict[str, Any]:
    """Reclassify UNCLASSIFIED nodes using GLiNER2.

    Args:
        driver: Connected Neo4j driver.
        dry_run: If True no writes are performed.
        batch_size: Number of nodes fetched and processed per batch.
        taxonomy: Optional domain taxonomy; supplies the label list for GLiNER2.
            Falls back to the default eu-politics taxonomy if None.

    Returns:
        Phase report dict with keys: phase, before, after, reclassified,
        left_unknown, distribution, dry_run.
    """
    # --- Resolve taxonomy and label set ---
    if taxonomy is None:
        try:
            from graphgen.config.taxonomy import get_default_taxonomy
            taxonomy = get_default_taxonomy()
        except Exception as exc:
            logger.warning("Could not load default taxonomy: %s. Using fallback labels.", exc)

    if taxonomy is not None:
        entity_labels = taxonomy.entity_label_keys()
    else:
        entity_labels = ["PERSON", "ORGANIZATION", "LOCATION", "POLICY", "EVENT"]

    logger.info(
        "gliner2_fallback: label set = %s (dry_run=%s)", entity_labels, dry_run
    )

    # --- Pre-flight count ---
    with driver.session() as s:
        before = int(s.run(_COUNT_QUERY).single()["c"])

    if before == 0:
        return {
            "phase": "gliner2_fallback",
            "before": 0,
            "after": 0,
            "reclassified": 0,
            "left_unknown": 0,
            "distribution": {},
            "dry_run": dry_run,
        }

    # --- Load GLiNER2 model ---
    try:
        from graphgen.pipeline.entity_relation.extraction import get_gliner2_model
        model = get_gliner2_model()
    except Exception as exc:
        logger.error("gliner2_fallback: could not load GLiNER2 model: %s", exc)
        return {
            "phase": "gliner2_fallback",
            "before": before,
            "after": before,
            "reclassified": 0,
            "left_unknown": before,
            "distribution": {},
            "dry_run": dry_run,
            "error": str(exc),
        }

    # --- Process in batches ---
    reclassified = 0
    total_seen = 0

    while True:
        with driver.session() as s:
            rows = list(s.run(_BATCH_QUERY, batch_size=batch_size))

        if not rows:
            break

        total_seen += len(rows)
        names = [r["name"] for r in rows]
        eids = [r["eid"] for r in rows]

        # Run GLiNER2 on each name synchronously (names are short strings)
        updates: List[Dict[str, Any]] = []
        for name, eid in zip(names, eids):
            try:
                result = model.extract_entities(
                    name,
                    entity_labels,
                    0.3,   # lower threshold for short strings
                    True,  # format_results
                    True,  # include_confidence
                    True,  # include_spans
                )
                entities_map = result.get("entities") or {}
                best_label: Optional[str] = None
                best_conf = 0.0
                for label, spans in entities_map.items():
                    for span in spans:
                        conf = float(span.get("confidence", 0.0))
                        if conf > best_conf:
                            best_conf = conf
                            best_label = label.upper()
                if best_label:
                    updates.append({"eid": eid, "ontology_class": best_label, "confidence": best_conf})
            except Exception as exc:
                logger.debug("GLiNER2 inference failed for %r: %s", name, exc)

        if not dry_run:
            with driver.session() as s:
                for upd in updates:
                    s.run(
                        _UPDATE_QUERY,
                        eid=upd["eid"],
                        ontology_class=upd["ontology_class"],
                        confidence=upd["confidence"],
                    )
            reclassified += len(updates)
        else:
            reclassified += len(updates)

        # If we got fewer rows than batch_size we've exhausted the set
        if len(rows) < batch_size:
            break

    # --- Post-flight count ---
    with driver.session() as s:
        after = int(s.run(_COUNT_QUERY).single()["c"])

    # --- Distribution of gliner2-assigned classes ---
    distribution: Dict[str, int] = {}
    if not dry_run:
        with driver.session() as s:
            for row in s.run(_DIST_QUERY):
                distribution[row["cls"]] = int(row["cnt"])

    return {
        "phase": "gliner2_fallback",
        "before": before,
        "after": after,
        "reclassified": reclassified,
        "left_unknown": after,
        "distribution": distribution,
        "dry_run": dry_run,
    }
