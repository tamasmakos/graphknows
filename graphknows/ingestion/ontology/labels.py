"""GLiNER label resolution helpers (flat copy from utils/labels.py).

Public symbols:
- :data:`DEFAULT_LABELS` — fallback label list.
- :func:`resolve_entity_labels` — resolve GLiNER labels from config.
"""

from __future__ import annotations

import logging
from typing import Any

from graphknows.ingestion.domain.normalization import standardize_label

logger = logging.getLogger(__name__)

DEFAULT_LABELS = [
    "Person",
    "Organization",
    "Location",
    "Policy",
    "Political Party",
    "Concept",
    "Event",
]


def resolve_entity_labels(extraction_config: dict[str, Any]) -> list[str]:
    """Resolve GLiNER labels from *extraction_config*.

    Supports:
    1. Manual labels from ``gliner_labels`` / ``entity_labels`` key.
    2. Ontology-derived labels when ``ontology.enabled`` is true.
    3. Merged combination of both.
    """
    if hasattr(extraction_config, "model_dump"):
        extraction_config = extraction_config.model_dump()

    manual_labels = (
        extraction_config.get("entity_labels")
        or extraction_config.get("gliner_labels")
        or DEFAULT_LABELS
    )

    ontology_config = extraction_config.get("ontology", {})
    if hasattr(ontology_config, "model_dump"):
        ontology_config = ontology_config.model_dump()

    ontology_enabled = ontology_config.get("enabled", False)

    result: list[str] = []
    if not ontology_enabled:
        logger.debug(f"Using {len(manual_labels)} manual GLiNER labels")
        result = manual_labels
    else:
        ontology_labels = _extract_ontology_labels(ontology_config)
        if not ontology_labels:
            logger.warning(
                "Ontology extraction enabled but no labels extracted, using manual labels"
            )
            result = manual_labels
        else:
            if ontology_config.get("merge_with_manual", True):
                combined = list(set(manual_labels + ontology_labels))
                logger.info(
                    f"Merged {len(manual_labels)} manual + {len(ontology_labels)} ontology"
                    f" = {len(combined)} total labels"
                )
                result = sorted(combined)
            else:
                logger.info(f"Using {len(ontology_labels)} ontology-derived labels")
                result = ontology_labels

    if not result:
        logger.warning(f"No entity labels resolved; using {len(DEFAULT_LABELS)} default labels.")
        result = DEFAULT_LABELS

    result = [standardize_label(label) for label in result]
    return sorted(set(result))


def _extract_ontology_labels(ontology_config: dict[str, Any]) -> list[str]:
    """Extract labels from ontology based on *ontology_config*."""
    try:
        from graphknows.ingestion.ontology.rdf_parser import OntologyLabelExtractor

        ontology_dir = ontology_config.get("ontology_dir", "input/ontology/cdm-4.13.2")
        namespace_filter = ontology_config.get("namespace_filter")
        include_local_names = ontology_config.get("include_local_names", True)
        top_level_only = ontology_config.get("top_level_only", True)
        min_subclasses = ontology_config.get("min_subclasses", 0)

        logger.info(
            f"Extracting labels from ontology: {ontology_dir}"
            f" (top_level={top_level_only}, min_children={min_subclasses})"
        )

        extractor = OntologyLabelExtractor(
            ontology_dir=ontology_dir,
            namespace_filter=namespace_filter,
        )
        labels = extractor.extract_labels(
            include_local_names=include_local_names,
            top_level_only=top_level_only,
            min_subclasses=min_subclasses,
        )
        logger.info(f"Extracted {len(labels)} labels from ontology")

    except Exception as e:
        logger.error(f"Failed to extract ontology labels: {e}", exc_info=True)
        return []
    else:
        return labels
