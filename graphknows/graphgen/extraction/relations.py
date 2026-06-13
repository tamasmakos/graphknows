"""DSPy-based relation extractor — graphgen extraction primitives.

Merges the former ``extractors.py`` and ``dspy_module.py`` into one module.

Public symbols:
- :class:`BaseExtractor` — abstract relation-extractor interface.
- :class:`DSPyExtractor` — DSPy + OpenRouter implementation.
- :func:`get_extractor` — factory that returns a ``DSPyExtractor``.
- :class:`GraphExtractorModule` — DSPy module for triple extraction.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import dspy
from pydantic import BaseModel, Field

from graphknows.config.llm import (
    _get_openrouter_api_key,
    normalize_openrouter_model,
)
from graphknows.graphgen.utils import standardize_label
from graphknows.ingestion.ontology.loader import RdfOntologyManifest, load_rdf_ontology_file

log = logging.getLogger(__name__)

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
    """Extract RDF-constrained relation triplets from text.

    Rules:
    - The *relation* field MUST be chosen from ontology_relation_types.
      Do NOT invent new relation labels.
    - Be generous: include all plausible relations supported by the text.
    - Set confidence >= 0.5 for relations clearly supported by the text.
    """

    text: str = dspy.InputField(desc="Input text to analyse.")
    ontology_context: str = dspy.InputField(
        desc="Compact JSON context parsed directly from the configured RDF/OWL ontology file."
    )
    ontology_classes: list[str] = dspy.InputField(
        desc="Allowed entity class labels parsed directly from the RDF/OWL ontology file."
    )
    ontology_relation_types: list[str] = dspy.InputField(
        desc=(
            "Exhaustive list of permitted relation labels parsed from RDF/OWL properties. "
            "The relation field in every output triplet MUST be one of these."
        )
    )
    triplets: list[Triplet] = dspy.OutputField(
        desc=(
            "Verified relation triplets grounded in the text and RDF/OWL ontology. "
            "Every triplet relation must be one of ontology_relation_types."
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
    "PERSON", "ORGANIZATION", "PROJECT", "SYSTEM", "TEAM", "LOCATION",
    "PRODUCT", "SERVICE", "TECHNOLOGY", "CONCEPT", "EVENT", "DOCUMENT",
]

_OPEN_DOMAIN_RELATION_TYPES = [
    "WORKS_AT", "WORKS_FOR", "EMPLOYED_BY", "MEMBER_OF", "JOINED",
    "MANAGES", "MANAGED_BY", "REPORTS_TO", "LEADS", "LED_BY",
    "OWNS", "OWNED_BY", "OPERATED_BY", "OPERATES",
    "FUNDS", "FUNDED_BY", "SPONSORS", "SPONSORED_BY",
    "PART_OF", "CONTAINS", "BELONGS_TO", "LOCATED_IN",
    "CREATED_BY", "DEVELOPED_BY", "AUTHORED_BY",
    "USES", "USED_BY", "DEPENDS_ON", "SUPPORTS",
    "SUBMITTED_TO", "APPROVED_BY", "REVIEWED_BY",
    "ASSOCIATED_WITH", "RELATED_TO", "CONNECTED_TO",
    "SUPPLIES", "SUPPLIED_BY", "CONTRACTED_WITH",
    "ASSIGNED_TO", "RESPONSIBLE_FOR",
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
            or "moonshotai/kimi-k2"
        )

        api_key = _get_openrouter_api_key(config)
        if not api_key:
            raise ValueError(
                "DSPyExtractor requires an OpenRouter API key. "
                "Set OPENROUTER_API_KEY in your .env file. "
                "Current value is empty — check /root/graphknows/.env."
            )

        self._configure_lm(api_key, model)

    def _load_ontology(self, ontology_config: dict[str, Any]) -> None:
        """Load RDF ontology labels, falling back to open-domain defaults."""
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
                raise ValueError(
                    f"RDF ontology file produced no DSPy labels: {ontology_file}"
                )
        else:
            self._ontology = None
            self._ontology_classes = _OPEN_DOMAIN_CLASSES
            self._ontology_relation_types = _OPEN_DOMAIN_RELATION_TYPES
            log.info("DSPyExtractor: no ontology file — using open-domain relation types")

    def _configure_lm(self, api_key: str, model: str) -> None:
        """Build the DSPy LM instance and module.

        Uses dspy.settings.context(lm=self._lm) inside _extract_sync instead of
        dspy.configure() to avoid mutating the global thread-local from a different
        thread than the one that will actually execute the module call.
        """
        or_model = normalize_openrouter_model(model)
        log.info("Configuring DSPy LM for OpenRouter with model %s", or_model)
        self._lm = dspy.LM(
            model=or_model,
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=2048,
            num_retries=0,
        )
        self.module = GraphExtractorModule()
        log.info("Initialized DSPy extractor with model %s", or_model)

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

    def _normalize_triplet(
        self, raw: dict[str, Any]
    ) -> tuple[str, str, str, str, str, dict[str, Any]]:
        """Standardize labels and build the props dict for a valid triplet."""
        source = standardize_label(raw["source"])
        relation = standardize_label(raw["relation"])
        target = standardize_label(raw["target"])
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
            ontology_context = self._ontology.context if self._ontology is not None else ""

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
                raw = self._extract_triplet_fields(triplet)
                raw_triplets.append(raw)

                valid, drop_reason = self._triplet_is_valid(raw, allowed_norm)
                if not valid:
                    triplet_decisions.append({**raw, "kept": False, "drop_reason": drop_reason})
                    continue

                triplet_decisions.append({**raw, "kept": True, "drop_reason": None})
                source, relation, target, source_type, target_type, props = (
                    self._normalize_triplet(raw)
                )
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
            raise RuntimeError(f"DSPy relation extraction failed: {e}") from e


def get_extractor(config: dict[str, Any]) -> BaseExtractor:
    """Return the sole supported extractor: DSPy."""
    return DSPyExtractor(config)
