"""GLiNER2-based entity extraction helpers.

Provides:
- :func:`get_gliner2_model` — lazy singleton model loader.
- :func:`get_max_concurrent` — concurrency setting from config.
- :func:`extract_relations_with_llm_async` — DSPy relation extraction wrapper.
- :func:`_extract_entities_for_chunk` — GLiNER2 Schema API NER for one chunk.
- :func:`_build_relation_eligible_entities` — filter entities for DSPy pass.
- :func:`_build_entity_surface_metadata` — enrich entity surface forms.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from graphknows.graphgen.extraction.canonicalization import (
    classify_surface_form,
    normalize_surface_form,
)
from graphknows.graphgen.extraction.ontology import (
    build_top_level_label_space,
    refine_entities_with_ontology,
)
from graphknows.graphgen.diagnostics import diagnostics_enabled, write_diagnostic_json
from graphknows.graphgen.utils import standardize_label

logger = logging.getLogger(__name__)

GLINER2_MODEL = None


def get_gliner2_model(config: dict[str, Any] | None = None):
    """Return the global GLiNER2 model singleton, loading it on first call.

    Args:
        config: Optional pipeline config dict with an ``extraction.gliner_model`` key.

    Returns:
        Loaded GLiNER2 model instance.

    Raises:
        RuntimeError: If the model cannot be loaded (OOM, missing weights, …).
    """
    global GLINER2_MODEL
    if GLINER2_MODEL is None:
        try:
            from gliner2 import GLiNER2

            model_name = "fastino/gliner2-base-v1"
            if config:
                extraction_cfg = config.get("extraction", {})
                if hasattr(extraction_cfg, "model_dump"):
                    extraction_cfg = extraction_cfg.model_dump()
                model_name = extraction_cfg.get("gliner_model", model_name)
            logger.info(f"Loading GLiNER2 model ({model_name})...")
            GLINER2_MODEL = GLiNER2.from_pretrained(model_name)
            logger.info("GLiNER2 model loaded.")
        except Exception as e:
            logger.error(f"Failed to load GLiNER2: {e}")
            raise RuntimeError(
                f"GLiNER2 model could not be loaded ({type(e).__name__}: {e}). "
                "Check RAM availability — model requires ~2 GB. No silent fallback."
            ) from e
    return GLINER2_MODEL


def get_max_concurrent(config: dict[str, Any], default: int = 8) -> int:
    """Return max_concurrent_chunks from *config* with a fallback *default*."""
    extraction_cfg = config.get("extraction", {})
    if hasattr(extraction_cfg, "model_dump"):
        extraction_cfg = extraction_cfg.model_dump()
    return extraction_cfg.get("max_concurrent_chunks", default)


async def extract_relations_with_llm_async(
    text: str,
    extractor: Any,
    keywords: list[str] | None = None,
    entities: list[str] | None = None,
    abstract_concepts: list[str] | None = None,
    allowed_relation_types: list[str] | None = None,
    candidate_relations: list[dict[str, Any]] | None = None,
) -> tuple[list[tuple[str, str, str, dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    """Extract relations using the configured DSPy extractor."""
    if not extractor or not text:
        return [], [], {}

    try:
        result = await extractor.extract_relations(
            text=text,
            keywords=keywords,
            entities=entities,
            abstract_concepts=abstract_concepts,
            allowed_relation_types=allowed_relation_types,
            candidate_relations=candidate_relations,
        )
        if isinstance(result, tuple):
            if len(result) == 3:
                relations, nodes, diagnostics = result
                return relations or [], nodes or [], diagnostics or {}
            if len(result) == 2:
                relations, nodes = result
                return relations or [], nodes or [], {}
        logger.warning("Extractor returned unexpected result shape; treating as empty extraction")
        return [], [], {}
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"extract_relations_with_llm_async failed: {e}") from e


async def _extract_entities_for_chunk(
    text: str,
    config: dict[str, Any],
    labels: list[str],
    return_diagnostics: bool = False,
    taxonomy: Any | None = None,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run GLiNER2 NER for a single chunk via the Schema API.

    Uses ``model.create_schema()`` → ``model.extract()`` to perform
    taxonomy-guided entity recognition and candidate relation extraction in a
    single model forward pass.

    The returned diagnostics dict always contains a ``candidate_relations`` key
    so callers can forward them to the DSPy verification pass.
    """
    extraction_config = config.get("extraction", {})
    if hasattr(extraction_config, "model_dump"):
        extraction_config = extraction_config.model_dump()

    diagnostics: dict[str, Any] = {
        "backend": "gliner2",
        "label_count": len(labels),
        "label_sample": labels[:20],
        "candidate_relations": [],
    }

    try:
        model = get_gliner2_model(config)
        if not model:
            raise RuntimeError(
                "GLiNER2 model failed to load (OOM or missing weights). "
                "Check available RAM — model requires ~2 GB."
            )

        threshold = extraction_config.get("gliner_threshold", 0.5)

        if taxonomy is None:
            raise RuntimeError(
                "No taxonomy supplied to _extract_entities_for_chunk. "
                "GLiNER2 Schema API requires a loaded TaxonomyConfig."
            )

        descriptions = {
            k: v.description.strip()
            for k, v in taxonomy.entity_groups.items()
            if hasattr(v, "description") and v.description.strip()
        }
        thresholds = {
            k: v.min_ner_confidence
            for k, v in taxonomy.entity_groups.items()
            if hasattr(v, "min_ner_confidence")
        }
        entity_spec = {
            label.lower(): {
                "description": descriptions.get(label, label.lower()),
                "threshold": thresholds.get(label) or threshold,
            }
            for label in taxonomy.entity_groups
        }
        gliner_schema = model.create_schema()
        gliner_schema.entities(entity_spec)

        if hasattr(taxonomy, "relation_types") and taxonomy.relation_types:
            rel_descs: dict[str, str] = {}
            for rt in taxonomy.relation_types:
                if isinstance(rt, dict):
                    rel_descs.update(rt)
                else:
                    rel_descs[str(rt)] = str(rt).replace("_", " ").lower()
            if rel_descs:
                gliner_schema.relations(rel_descs)

        raw = await asyncio.to_thread(
            model.extract,
            text,
            gliner_schema,
            threshold,
            True,  # format_results
            True,  # include_confidence
            True,  # include_spans
        )

        child_to_group: dict[str, str] = {}
        for group_label, group_cfg in taxonomy.entity_groups.items():
            child_to_group[group_label.lower()] = group_label
            for child in getattr(group_cfg, "children", []):
                child_to_group[child.lower()] = group_label

        flat_entities: list[dict[str, Any]] = []
        for raw_label, spans in (raw.get("entities") or {}).items():
            canonical = child_to_group.get(raw_label.lower(), raw_label.upper())
            for span in spans or []:
                flat_entities.append(
                    {
                        "text": span.get("text", ""),
                        "label": standardize_label(canonical),
                        "ontology_label": standardize_label(canonical),
                        "confidence": float(span.get("confidence", 0.0)),
                        "start": span.get("start"),
                        "end": span.get("end"),
                        "backend": "gliner2",
                    }
                )

        candidate_relations: list[dict[str, Any]] = []
        for rel_type, triplets in (raw.get("relation_extraction") or {}).items():
            for triplet in triplets or []:
                head = triplet.get("head", {})
                tail = triplet.get("tail", {})
                src_text = head.get("text", "")
                tgt_text = tail.get("text", "")
                if not src_text or not tgt_text or src_text == tgt_text:
                    continue
                candidate_relations.append(
                    {
                        "source": src_text,
                        "relation": rel_type.upper(),
                        "target": tgt_text,
                        "confidence": min(
                            float(head.get("confidence", 0.0)),
                            float(tail.get("confidence", 0.0)),
                        ),
                        "extractor": "gliner2",
                    }
                )

        label_space = build_top_level_label_space(labels or [], taxonomy=taxonomy)
        refined_entities = refine_entities_with_ontology(
            flat_entities, label_space, taxonomy=taxonomy
        )
        diagnostics.update(
            {
                "candidate_relations": candidate_relations,
                "raw_entities": flat_entities,
                "entities": refined_entities,
                "threshold": threshold,
                "schema_mode": "taxonomy",
            }
        )
        return (refined_entities, diagnostics) if return_diagnostics else refined_entities

    except Exception as e:
        raise RuntimeError(f"GLiNER2 extraction failed for chunk: {e}") from e


def _build_relation_eligible_entities(
    gliner_entities: list[dict[str, Any]],
    taxonomy: Any | None = None,
) -> list[str]:
    """Filter and rank NER entities for relation-extraction eligibility."""
    if taxonomy is not None:
        known_labels = set(taxonomy.entity_groups.keys())
    else:
        known_labels = {"PERSON", "ORGANIZATION", "LOCATION", "EVENT", "POLICY"}

    def _min_conf(label: str) -> float:
        if taxonomy is not None:
            return taxonomy.get_min_ner_confidence(label)
        return {"EVENT": 0.5, "ORGANIZATION": 0.6, "LOCATION": 0.6}.get(label, 0.0)

    eligible = []
    fallback = []

    for entity in gliner_entities or []:
        text = standardize_label(entity.get("text", ""))
        ontology_label = standardize_label(entity.get("ontology_label") or entity.get("label", ""))
        confidence = float(entity.get("confidence", 0.0) or 0.0)

        if not text:
            continue

        fallback.append(text)

        if len(text) <= 3 and "_" not in text and ontology_label in known_labels:
            continue
        if ontology_label == "PERSON" and "_" not in text:
            continue
        if confidence < _min_conf(ontology_label):
            continue

        eligible.append(text)

    deduped_eligible = list(dict.fromkeys(eligible))
    if deduped_eligible:
        return deduped_eligible
    return list(dict.fromkeys(fallback))


def _build_entity_surface_metadata(gliner_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich GLiNER2 entity spans with surface-form classification metadata."""
    metadata = []
    for entity in gliner_entities or []:
        text = entity.get("text", "")
        metadata.append(
            {
                "text": text,
                "canonical_form": normalize_surface_form(text),
                "surface_form_class": classify_surface_form(text),
                "ontology_label": entity.get("ontology_label") or entity.get("label"),
                "confidence": entity.get("confidence"),
                "merge_candidate": classify_surface_form(text) == "named_entity",
            }
        )
    return metadata
