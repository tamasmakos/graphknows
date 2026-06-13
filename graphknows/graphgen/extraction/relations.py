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
        ontology_file = str(ontology_config.get("file") or "").strip()
        if ontology_file:
            self._ontology: RdfOntologyManifest | None = load_rdf_ontology_file(ontology_file)
            self._ontology_classes = [standardize_label(label) for label in self._ontology.class_labels]
            self._ontology_relation_types = [
                standardize_label(label) for label in self._ontology.property_labels
            ]
            if not self._ontology_classes or not self._ontology_relation_types:
                raise ValueError(f"RDF ontology file produced no DSPy labels: {ontology_file}")
        else:
            # No ontology — use a broad open-domain set of semantic relation types.
            self._ontology = None
            self._ontology_classes = [
                "PERSON", "ORGANIZATION", "PROJECT", "SYSTEM", "TEAM", "LOCATION",
                "PRODUCT", "SERVICE", "TECHNOLOGY", "CONCEPT", "EVENT", "DOCUMENT",
            ]
            self._ontology_relation_types = [
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
            log.info("DSPyExtractor: no ontology file — using open-domain relation types")

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

        or_model = normalize_openrouter_model(model)
        log.info("Configuring DSPy LM for OpenRouter with model %s", or_model)
        # Store the LM instance on self — do NOT call dspy.configure() here.
        # dspy.configure() mutates a global thread-local and will raise
        # "dspy.settings can only be changed by the thread that initially configured it"
        # when extract_relations runs _extract_sync in asyncio.to_thread (a different thread).
        # Instead we use dspy.settings.context(lm=self._lm) inside _extract_sync so the
        # LM is scoped to whichever thread actually executes the module call.
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
                    prediction = self.module(
                        text=text,
                        ontology_context=ontology_context,
                        ontology_classes=ontology_classes,
                        ontology_relation_types=ontology_relation_types,
                    )
                return prediction

            prediction = await asyncio.to_thread(_extract_sync)
            triplets = _coerce_triplets(prediction)

            relations: list[tuple[str, str, str, dict[str, Any]]] = []
            nodes_data: list[dict[str, Any]] = []
            seen_nodes: set[str] = set()
            raw_triplets: list[dict[str, Any]] = []
            triplet_decisions: list[dict[str, Any]] = []

            if triplets:
                for triplet in triplets:
                    source = _model_or_mapping_get(triplet, "source")
                    relation = _model_or_mapping_get(triplet, "relation")
                    target = _model_or_mapping_get(triplet, "target")
                    source_type = _model_or_mapping_get(triplet, "source_type")
                    target_type = _model_or_mapping_get(triplet, "target_type")
                    evidence = _model_or_mapping_get(triplet, "evidence", "")

                    raw_triplet: dict[str, Any] = {
                        "source": source,
                        "relation": relation,
                        "target": target,
                        "source_type": source_type,
                        "target_type": target_type,
                        "evidence": evidence,
                    }
                    raw_triplets.append(raw_triplet)

                    if not (source and relation and target):
                        triplet_decisions.append(
                            {
                                **raw_triplet,
                                "kept": False,
                                "drop_reason": "missing_triplet_field",
                            }
                        )
                        continue

                    decision: dict[str, Any] = {
                        **raw_triplet,
                        "kept": False,
                        "drop_reason": None,
                    }

                    rel_norm = standardize_label(relation)
                    allowed_norm = set(ontology_relation_types)
                    if rel_norm not in allowed_norm:
                        decision["drop_reason"] = "relation_type_not_in_rdf_ontology"
                        triplet_decisions.append(decision)
                        continue

                    decision["kept"] = True
                    triplet_decisions.append(decision)

                    source = standardize_label(source)
                    target = standardize_label(target)
                    relation = standardize_label(relation)
                    source_type = standardize_label(source_type) if source_type else "ENTITY"
                    target_type = standardize_label(target_type) if target_type else "ENTITY"

                    props = {
                        "evidence": evidence,
                        "source_type": source_type,
                        "target_type": target_type,
                    }
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
