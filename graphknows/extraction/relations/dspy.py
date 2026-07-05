"""DSPy-based relation extractor — graphgen extraction primitives.

Merges the former ``extractors.py`` and ``dspy_module.py`` into one module.

Public symbols:
- :class:`BaseExtractor` — abstract relation-extractor interface.
- :class:`DSPyExtractor` — DSPy + OpenRouter implementation.
- :func:`get_extractor` — factory that returns a ``DSPyExtractor``.
- :class:`GraphExtractorModule` — DSPy module for triple extraction.

Extract-then-map
~~~~~~~~~~~~~~~~
After the LLM produces free-form predicate labels, each label is mapped to
the closest controlled-vocabulary property from the loaded RDF/OWL ontology
using :class:`~graphknows.extraction.ontology.relation_filter.OntologyPredicateMapper`.
Triples whose predicate cannot be mapped above the similarity threshold are
dropped.  This two-stage approach (generate then constrain) lets the LLM
surface near-miss labels ("is friends with" → ``friend_of``) while still
enforcing ontology discipline.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

try:
    import dspy
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
    raise ModuleNotFoundError(
        "llm_assisted extraction requires the 'assisted' extra. "
        "Install it with:  pip install 'graphknows[assisted]'"
    ) from exc
from pydantic import BaseModel, Field

from graphknows.config.llm import _get_openrouter_api_key, build_lm
from graphknows.extraction.ontology.loader import RdfOntologyManifest, load_rdf_ontology_file
from graphknows.extraction.ontology.relation_filter import OntologyPredicateMapper
from graphknows.extraction.utils import standardize_label

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ontology context enrichment
# ---------------------------------------------------------------------------


def _build_enriched_ontology_context(manifest: RdfOntologyManifest) -> str:
    """Build a SKOS-enriched context string from an RDF/OWL ontology manifest.

    Extends the basic ``{"classes": [...], "properties": [...]}`` context from
    :func:`~graphknows.extraction.ontology.loader.load_rdf_ontology_file` with
    per-property domain/range and SKOS ``prefLabel``/``definition`` where
    available.  This gives the DSPy LLM structured signal about *what each
    relation means* and *which entity types it connects*.

    Falls back to the manifest's raw context string when rdflib is unavailable
    or the ontology file cannot be parsed.
    """
    import json

    try:
        import rdflib
        from rdflib import OWL, RDF, RDFS
        from rdflib.namespace import SKOS

        path = manifest.file_path
        graph = rdflib.Graph()
        graph.parse(path)

        def _label(node: object) -> str:
            lbl = graph.value(node, RDFS.label)  # type: ignore[arg-type]
            if lbl:
                return str(lbl)
            s = str(node)
            if "#" in s:
                return s.rsplit("#", 1)[-1]
            if "/" in s:
                return s.rstrip("/").rsplit("/", 1)[-1]
            return s

        property_nodes = (
            set(graph.subjects(RDF.type, OWL.ObjectProperty))
            | set(graph.subjects(RDF.type, OWL.DatatypeProperty))
            | set(graph.subjects(RDF.type, RDF.Property))
        )

        enriched_props: list[dict] = []
        for prop in property_nodes:
            prop_label = _label(prop)
            if not prop_label:
                continue
            entry: dict = {"label": prop_label}
            domain = graph.value(prop, RDFS.domain)  # type: ignore[arg-type]
            if domain:
                entry["domain"] = _label(domain)
            rng = graph.value(prop, RDFS.range)  # type: ignore[arg-type]
            if rng:
                entry["range"] = _label(rng)
            pref = graph.value(prop, SKOS.prefLabel)  # type: ignore[arg-type]
            if pref:
                entry["prefLabel"] = str(pref)
            defn = graph.value(prop, SKOS.definition)  # type: ignore[arg-type]
            if defn:
                entry["definition"] = str(defn)
            enriched_props.append(entry)

        enriched_props.sort(key=lambda p: p["label"])

        base = json.loads(manifest.context)
        base["properties_detail"] = enriched_props
        return json.dumps(base, ensure_ascii=False)

    except Exception as exc:
        log.debug("Could not build enriched ontology context: %s — using base context", exc)
        return manifest.context


# ---------------------------------------------------------------------------
# DSPy signature & module
# ---------------------------------------------------------------------------


class Triplet(BaseModel):
    source: str = Field(description="The source entity")
    source_type: str = Field(description="The ontology class of the source entity")
    relation: str = Field(description="The relationship between source and target")
    target: str = Field(description="The target entity")
    target_type: str = Field(description="The ontology class of the target entity")
    confidence: float = Field(description="Confidence score (0.0-1.0) of this extraction")
    evidence: str = Field(description="Verbatim text snippet that supports this relation")
    reasoning: str = Field(
        default="",
        description="Short explanation of how the evidence supports this relation.",
    )


class EntityRelation(dspy.Signature):
    """Extract RDF-constrained relation triplets from text using the provided ontology.

    Rules:
    - The *relation* field MUST be one of the labels in ontology_relation_types.
      Prefer the closest ontology label — if the text says "is friends with",
      emit "friend_of". Do NOT invent new relation labels.
    - The *source_type* and *target_type* fields MUST be one of the labels in
      ontology_classes.  Use the domain/range annotations in ontology_context to
      choose correct types (e.g. friend_of: domain=Person, range=Person).
    - Be generous: include all plausible relations clearly supported by the text.
    - Set confidence >= 0.5 for relations directly evidenced by the text.
    """

    text: str = dspy.InputField(desc="Input text to analyse.")
    ontology_context: str = dspy.InputField(
        desc=(
            "JSON context from the RDF/OWL ontology.  Contains 'classes', 'properties', and "
            "'properties_detail' (list of {label, domain, range, definition} dicts) — use the "
            "domain/range fields to assign correct source_type/target_type values."
        )
    )
    ontology_classes: list[str] = dspy.InputField(
        desc=(
            "Allowed entity class labels from the RDF/OWL ontology.  "
            "Use ONLY these as source_type and target_type values."
        )
    )
    ontology_relation_types: list[str] = dspy.InputField(
        desc=(
            "Exhaustive list of permitted relation labels from the RDF/OWL ontology properties. "
            "The relation field in every output triplet MUST be one of these exact strings. "
            "Choose the closest matching label — do NOT invent new ones."
        )
    )
    triplets: list[Triplet] = dspy.OutputField(
        desc=(
            "Verified relation triplets grounded in the text and constrained by the RDF/OWL "
            "ontology.  Every triplet's relation must be one of ontology_relation_types, and "
            "source_type/target_type must be one of ontology_classes."
        )
    )


class GraphExtractorModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.Predict(EntityRelation)

    def forward(
        self,
        text: str,
        ontology_context: str,
        ontology_classes: list[str],
        ontology_relation_types: list[str],
    ):
        return self.extract(
            text=text,
            ontology_context=ontology_context,
            ontology_classes=ontology_classes,
            ontology_relation_types=ontology_relation_types,
        )


def _model_or_mapping_get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _coerce_triplets(prediction: Any) -> list[Any]:
    triplets = getattr(prediction, "triplets", None)
    if triplets is None:
        raise ValueError("DSPy prediction did not include triplets")
    return triplets or []


def _is_empty_lm_response_error(exc: Exception) -> bool:
    """Return true for provider responses that contain no parseable extraction."""
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "empty or null response",
            "failed to parse the lm response",
            "did not include triplets",
        )
    )


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BaseExtractor(ABC):
    """Abstract base for graph relation extractors."""

    @abstractmethod
    async def extract_relations(
        self,
        text: str,
    ) -> tuple[list[tuple[str, str, str, dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
        """Extract (source, relation, target, props) triplets from *text*."""

    async def close(self) -> None:
        """Clean up resources."""


# ---------------------------------------------------------------------------
# DSPy implementation
# ---------------------------------------------------------------------------

_OPEN_DOMAIN_CLASSES = [
    "PERSON",
    "ORGANIZATION",
    "PROJECT",
    "SYSTEM",
    "TEAM",
    "LOCATION",
    "PRODUCT",
    "SERVICE",
    "TECHNOLOGY",
    "CONCEPT",
    "EVENT",
    "DOCUMENT",
]

_OPEN_DOMAIN_RELATION_TYPES = [
    "WORKS_AT",
    "WORKS_FOR",
    "EMPLOYED_BY",
    "MEMBER_OF",
    "JOINED",
    "MANAGES",
    "MANAGED_BY",
    "REPORTS_TO",
    "LEADS",
    "LED_BY",
    "OWNS",
    "OWNED_BY",
    "OPERATED_BY",
    "OPERATES",
    "FUNDS",
    "FUNDED_BY",
    "SPONSORS",
    "SPONSORED_BY",
    "PART_OF",
    "CONTAINS",
    "BELONGS_TO",
    "LOCATED_IN",
    "CREATED_BY",
    "DEVELOPED_BY",
    "AUTHORED_BY",
    "USES",
    "USED_BY",
    "DEPENDS_ON",
    "SUPPORTS",
    "SUBMITTED_TO",
    "APPROVED_BY",
    "REVIEWED_BY",
    "ASSOCIATED_WITH",
    "RELATED_TO",
    "CONNECTED_TO",
    "SUPPLIES",
    "SUPPLIED_BY",
    "CONTRACTED_WITH",
    "ASSIGNED_TO",
    "RESPONSIBLE_FOR",
]


class DSPyExtractor(BaseExtractor):
    """DSPy + OpenRouter relation extractor."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        llm_config = config.get("llm", {})
        if hasattr(llm_config, "model_dump"):
            llm_config = llm_config.model_dump()
        ontology_config = config.get("ontology", {})
        if hasattr(ontology_config, "model_dump"):
            ontology_config = ontology_config.model_dump()

        self._load_ontology(ontology_config)

        model = (
            llm_config.get("extraction_model")
            or llm_config.get("base_model")
            or llm_config.get("model")
        )

        api_key = llm_config.get("api_key") or _get_openrouter_api_key(config)
        if not api_key:
            raise ValueError(
                "llm_assisted extraction requires an LLM API key. "
                "Set GRAPHKNOWS_LLM_API_KEY in your environment or .env."
            )

        self._configure_lm(api_key, model, llm_config.get("api_base", ""))

    def _load_ontology(self, ontology_config: dict[str, Any]) -> None:
        """Load RDF ontology labels, falling back to open-domain defaults.

        Also builds an :class:`OntologyPredicateMapper` for the extract-then-map
        step.  When an ontology file is configured, the mapper constrains
        free-form LLM predicates to the ontology's controlled vocabulary.  When
        no ontology file is given, the mapper is built from the open-domain
        relation type list so extract-then-map still applies (with relaxed
        similarity matching because the open-domain list is large).
        """
        ontology_file = str(ontology_config.get("file") or "").strip()
        if ontology_file:
            self._ontology: RdfOntologyManifest | None = load_rdf_ontology_file(ontology_file)
            self._ontology_classes = [
                standardize_label(label) for label in self._ontology.class_labels
            ]
            self._ontology_relation_types = [
                standardize_label(label) for label in self._ontology.property_labels
            ]
            if not self._ontology_classes or not self._ontology_relation_types:
                raise ValueError(f"RDF ontology file produced no DSPy labels: {ontology_file}")
            # Build the extract-then-map predicate mapper from the ontology labels.
            # The raw (non-standardized) property labels are used so the mapper
            # can match surface forms like "is friends with" → "friend_of".
            self._predicate_mapper: OntologyPredicateMapper = OntologyPredicateMapper(
                self._ontology.property_labels
            )
            # Build the enriched SKOS context for the DSPy LLM prompt.
            self._ontology_enriched_context: str = _build_enriched_ontology_context(self._ontology)
        else:
            self._ontology = None
            self._ontology_classes = _OPEN_DOMAIN_CLASSES
            self._ontology_relation_types = _OPEN_DOMAIN_RELATION_TYPES
            self._predicate_mapper = OntologyPredicateMapper(
                _OPEN_DOMAIN_RELATION_TYPES,
                min_similarity=0.35,  # Slightly higher threshold for the open-domain list
            )
            self._ontology_enriched_context = ""
            log.info("DSPyExtractor: no ontology file — using open-domain relation types")

    def _configure_lm(self, api_key: str, model: str, api_base: str = "") -> None:
        """Build the DSPy LM instance and module.

        Uses dspy.settings.context(lm=self._lm) inside _extract_sync instead of
        dspy.configure() to avoid mutating the global thread-local from a different
        thread than the one that will actually execute the module call.
        """
        log.info("Configuring DSPy LM with model %s", model)
        self._lm = build_lm(
            model,
            api_key=api_key,
            api_base=api_base,
            temperature=self.config.get("llm", {}).get("temperature", 0.0),
            max_tokens=self.config.get("llm", {}).get("max_tokens", 2048),
            num_retries=0,
        )
        self.module = GraphExtractorModule()
        log.info("Initialized DSPy extractor with model %s", model)

    def _extract_triplet_fields(self, triplet: Any) -> dict[str, Any]:
        """Pull raw fields from a DSPy Triplet object or dict."""
        return {
            "source": _model_or_mapping_get(triplet, "source"),
            "relation": _model_or_mapping_get(triplet, "relation"),
            "target": _model_or_mapping_get(triplet, "target"),
            "source_type": _model_or_mapping_get(triplet, "source_type"),
            "target_type": _model_or_mapping_get(triplet, "target_type"),
            "evidence": _model_or_mapping_get(triplet, "evidence", ""),
        }

    def _triplet_is_valid(
        self, raw: dict[str, Any], allowed_relations: set[str]
    ) -> tuple[bool, str | None]:
        """Return (is_valid, drop_reason). Pure predicate, no side effects."""
        if not (raw.get("source") and raw.get("relation") and raw.get("target")):
            return False, "missing_triplet_field"
        if standardize_label(raw["relation"]) not in allowed_relations:
            return False, "relation_type_not_in_rdf_ontology"
        return True, None

    def _apply_extract_then_map(
        self, raw: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Map a raw triplet's predicate to the closest ontology property.

        This is the extract-then-map step: the LLM is allowed to emit free-form
        predicates, and we post-hoc map each to the closest controlled-vocabulary
        property using :class:`OntologyPredicateMapper`.

        Returns:
            ``(mapped_raw, drop_reason)`` where ``mapped_raw`` is the triplet
            dict with the ``relation`` field replaced by the canonical label,
            and ``drop_reason`` is set (and ``mapped_raw`` is ``None``) when
            the predicate cannot be mapped above the similarity threshold.
        """
        original_relation = raw.get("relation", "")
        # Guard: _predicate_mapper may not be set if the extractor was created
        # via __new__ without __init__ (e.g. in unit tests with manual attribute setup).
        predicate_mapper = getattr(self, "_predicate_mapper", None)
        if predicate_mapper is None:
            # No mapper — pass the relation through unchanged (no constraint).
            return dict(raw), None
        canonical, score = predicate_mapper.map_predicate(original_relation)
        if canonical is None:
            return None, f"predicate_unmappable(score={score:.3f})"
        mapped = dict(raw)
        mapped["relation"] = canonical
        if canonical != original_relation:
            mapped["_mapped_from"] = original_relation
            mapped["_map_score"] = round(score, 4)
            log.debug(
                "extract-then-map: '%s' → '%s' (score=%.3f)", original_relation, canonical, score
            )
        return mapped, None

    def _normalize_triplet(
        self, raw: dict[str, Any]
    ) -> tuple[str, str, str, str, str, dict[str, Any]]:
        """Standardize labels and build the props dict for a valid triplet.

        Entity names (source/target) preserve original casing so they match
        GLiNER2 extractions during entity resolution. Relation types are still
        standardized to UPPER_CASE_WITH_UNDERSCORES for Cypher compatibility.
        """
        source = (raw["source"] or "").strip()
        relation = standardize_label(raw["relation"])
        target = (raw["target"] or "").strip()
        source_type = standardize_label(raw.get("source_type") or "") or "ENTITY"
        target_type = standardize_label(raw.get("target_type") or "") or "ENTITY"
        props = {
            "evidence": raw.get("evidence", ""),
            "source_type": source_type,
            "target_type": target_type,
        }
        return source, relation, target, source_type, target_type, props

    async def extract_relations(
        self,
        text: str,
    ) -> tuple[list[tuple[str, str, str, dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
        """Extract relations using DSPy."""
        ontology_classes = self._ontology_classes
        ontology_relation_types = self._ontology_relation_types

        try:
            lm = self._lm
            # Use the SKOS-enriched context when available; it carries per-property
            # domain/range/definition so the LLM makes better typing decisions.
            # Fallback: use the raw manifest context or empty string (supports tests
            # that build DSPyExtractor via __new__ without calling __init__).
            ontology_context = getattr(self, "_ontology_enriched_context", None) or (
                self._ontology.context if self._ontology is not None else ""
            )

            def _extract_sync() -> Any:
                with dspy.settings.context(lm=lm):
                    return self.module(
                        text=text,
                        ontology_context=ontology_context,
                        ontology_classes=ontology_classes,
                        ontology_relation_types=ontology_relation_types,
                    )

            prediction = await asyncio.to_thread(_extract_sync)
            triplets = _coerce_triplets(prediction)

            relations: list[tuple[str, str, str, dict[str, Any]]] = []
            nodes_data: list[dict[str, Any]] = []
            seen_nodes: set[str] = set()
            raw_triplets: list[dict[str, Any]] = []
            triplet_decisions: list[dict[str, Any]] = []
            allowed_norm = set(ontology_relation_types)

            for triplet in triplets:
                try:
                    raw = self._extract_triplet_fields(triplet)
                except Exception as item_exc:
                    log.debug("Skipping malformed triplet item: %s", item_exc)
                    triplet_decisions.append({"kept": False, "drop_reason": "malformed_item"})
                    continue
                raw_triplets.append(raw)

                # --- Step 1: require source, relation, target fields ----------
                if not (raw.get("source") and raw.get("relation") and raw.get("target")):
                    triplet_decisions.append(
                        {**raw, "kept": False, "drop_reason": "missing_triplet_field"}
                    )
                    continue

                # --- Step 2: extract-then-map --------------------------------
                # Map the free-form LLM predicate to the closest controlled-
                # vocabulary property.  If unmappable, drop the triple.
                mapped_raw, map_drop_reason = self._apply_extract_then_map(raw)
                if mapped_raw is None:
                    triplet_decisions.append(
                        {**raw, "kept": False, "drop_reason": map_drop_reason or "extract_then_map_failed"}
                    )
                    continue

                # --- Step 3: final ontology membership check -----------------
                # After mapping, the canonical relation *should* always be in
                # allowed_norm, but we verify to guard against edge cases.
                valid, drop_reason = self._triplet_is_valid(mapped_raw, allowed_norm)
                if not valid:
                    triplet_decisions.append(
                        {**mapped_raw, "kept": False, "drop_reason": drop_reason}
                    )
                    continue

                triplet_decisions.append({**mapped_raw, "kept": True, "drop_reason": None})
                source, relation, target, source_type, target_type, props = self._normalize_triplet(
                    mapped_raw
                )
                # Surface the mapping metadata in edge properties when the
                # predicate was remapped from a free-form LLM label.
                if "_mapped_from" in mapped_raw:
                    props["_mapped_from"] = mapped_raw["_mapped_from"]
                    props["_map_score"] = mapped_raw.get("_map_score")
                relations.append((source, relation, target, props))

                if source not in seen_nodes:
                    nodes_data.append({"id": source, "type": source_type, "properties": {}})
                    seen_nodes.add(source)
                if target not in seen_nodes:
                    nodes_data.append({"id": target, "type": target_type, "properties": {}})
                    seen_nodes.add(target)

            return (
                relations,
                nodes_data,
                {
                    "raw_triplets": raw_triplets,
                    "triplet_decisions": triplet_decisions,
                    "ontology_classes": ontology_classes,
                    "ontology_relation_types": ontology_relation_types,
                },
            )

        except Exception as e:
            if _is_empty_lm_response_error(e):
                log.warning(
                    "DSPy relation extraction returned no parseable output; "
                    "treating chunk as having no assisted relations: %s",
                    e,
                )
                return (
                    [],
                    [],
                    {
                        "raw_triplets": [],
                        "triplet_decisions": [],
                        "ontology_classes": ontology_classes,
                        "ontology_relation_types": ontology_relation_types,
                        "error_type": "empty_lm_response",
                        "error": str(e),
                    },
                )
            # Pydantic ValidationError during DSPy list parsing: one or more
            # triplets were malformed.  Log and return empty so the chunk is not
            # lost entirely from the ingest pipeline.  Individual malformed items
            # inside a successfully-parsed list are already handled per-item above.
            try:
                from pydantic import ValidationError as _PydanticValidationError

                if isinstance(e, _PydanticValidationError):
                    log.warning(
                        "DSPy pydantic validation error — skipping malformed triplets: %s",
                        e,
                    )
                    return (
                        [],
                        [],
                        {
                            "raw_triplets": [],
                            "triplet_decisions": [],
                            "ontology_classes": ontology_classes,
                            "ontology_relation_types": ontology_relation_types,
                            "error_type": "validation_error",
                            "error": str(e),
                        },
                    )
            except ImportError:
                pass
            raise RuntimeError(f"DSPy relation extraction failed: {e}") from e


def get_extractor(config: dict[str, Any]) -> BaseExtractor:
    """Return the sole supported extractor: DSPy."""
    return DSPyExtractor(config)
