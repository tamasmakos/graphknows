"""GLiNER2 entity extractor for the memory ingestion pipeline.

Wraps the GLiNER2 Schema API using the project taxonomy, matching the thesis
pipeline: ontology-guided extraction with per-group confidence thresholds.

Returns dicts with keys ``name``, ``type``, ``score`` — the contract expected
by ``MemoryIngestor``.

No silent fallbacks. If GLiNER2 is not installed, the API key is missing, or
the model fails to load, a ``RuntimeError`` is raised immediately.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_BASE_THRESHOLD: float = 0.35


class GLiNER2EntityExtractor:
    """Taxonomy-driven GLiNER2 entity extractor.

    Loads the model once on first call to ``extract()`` (lazy load per
    dev-conventions). Raises ``RuntimeError`` on any hard failure — no silent
    fallbacks, no empty-list returns on model errors.

    Args:
        taxonomy_path: Path to taxonomy YAML file. When omitted, loads the
            default EU-politics taxonomy via ``get_default_taxonomy()``.
    """

    def __init__(self, taxonomy_path: str | None = None) -> None:
        from graphknows.graphgen.config.taxonomy import get_default_taxonomy, load_taxonomy

        self._taxonomy = load_taxonomy(taxonomy_path) if taxonomy_path else get_default_taxonomy()
        self._gliner: Any | None = None
        self._gliner_schema: Any | None = None
        self._child_to_group: dict[str, str] = {}

        # Build entity spec from taxonomy (cheap — no model load here)
        self._entity_spec: dict[str, dict[str, Any]] = {
            label.lower(): {
                "description": grp.description.strip() or label.lower().replace("_", " "),
                "threshold": max(grp.min_ner_confidence, _BASE_THRESHOLD),
            }
            for label, grp in self._taxonomy.entity_groups.items()
            if grp.description.strip()
        }
        self._rel_descs: dict[str, str] = {
            rt: rt.replace("_", " ").lower() for rt in self._taxonomy.relation_types
        }

    def _ensure_model_loaded(self) -> None:
        """Load GLiNER2 model on first extraction call.

        Raises:
            RuntimeError: If the model cannot be loaded (missing package, OOM,
                missing weights). No silent fallback — callers must fix the
                environment.
        """
        if self._gliner is not None:
            return

        import contextlib
        import sys

        from graphknows.graphgen.pipeline.entity_relation.extraction import get_gliner2_model

        log.info("Loading GLiNER2 model (first extraction call)...")
        with contextlib.redirect_stdout(sys.stderr):
            model = get_gliner2_model()

        if model is None:
            raise RuntimeError(
                "GLiNER2 model failed to load — check RAM (~2 GB required) and "
                "HuggingFace access. No silent fallback."
            )

        schema = model.create_schema()
        schema.entities(self._entity_spec)
        if self._rel_descs:
            schema.relations(self._rel_descs)

        # Build child-label → canonical group mapping
        child_to_group: dict[str, str] = {}
        for group_label, grp in self._taxonomy.entity_groups.items():
            child_to_group[group_label.lower()] = group_label
            for child in grp.children:
                child_to_group[child.lower()] = group_label

        self._gliner = model
        self._gliner_schema = schema
        self._child_to_group = child_to_group
        log.info("GLiNER2 model loaded (taxonomy=%s)", self._taxonomy.name)

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Extract entities from ``text`` using GLiNER2 Schema API + taxonomy.

        Args:
            text: Raw text segment to process.

        Returns:
            List of entity dicts with keys:
                - ``name`` (str): entity surface form
                - ``type`` (str): canonical taxonomy group label (e.g. PERSON)
                - ``score`` (float): confidence score

        Raises:
            RuntimeError: If the model cannot be loaded.
            Exception: Any GLiNER2 extraction error propagates — no silent
                swallow. The caller (``MemoryIngestor``) must handle failures.
        """
        self._ensure_model_loaded()

        raw = self._gliner.extract(
            text,
            self._gliner_schema,
            _BASE_THRESHOLD,
            True,  # format_results
            True,  # include_confidence
            True,  # include_spans
        )

        entities: list[dict[str, Any]] = []
        for raw_label, spans in (raw.get("entities") or {}).items():
            canonical = self._child_to_group.get(raw_label.lower(), raw_label.upper())
            min_conf = self._taxonomy.get_min_ner_confidence(canonical)
            eff_threshold = max(min_conf, _BASE_THRESHOLD)
            for span in spans or []:
                conf = float(span.get("confidence", 0.0))
                if conf < eff_threshold:
                    continue
                name = span.get("text", "").strip()
                if not name:
                    continue
                entities.append(
                    {
                        "name": name,
                        "type": canonical,
                        "score": conf,
                    }
                )

        return entities
