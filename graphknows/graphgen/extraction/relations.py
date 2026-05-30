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
import time
from abc import ABC, abstractmethod
from typing import Any

import dspy
from pydantic import BaseModel, Field

from graphknows.graphgen.llm import (
    _get_openrouter_api_key,
    normalize_openrouter_model,
)
from graphknows.graphgen.utils import standardize_label

logger = logging.getLogger(__name__)

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


class EntityRelation(dspy.Signature):
    """Extract knowledge graph triplets (source, relation, target) from text.

    Rules:
    - Extract relations that are clearly stated or strongly implied by the text.
    - The *relation* field MUST be chosen from allowed_relation_types.
      Do NOT invent new relation labels.
    - Use entity_hints as the preferred surface forms for source/target.
    - Use candidate_relations as a starting point; verify them and add new ones.
    - Be generous: include all plausible relations supported by the text.
    - Aim to produce at least one triplet per pair of co-occurring named entities.
    - Set confidence >= 0.5 for relations clearly supported by the text.
    """

    text: str = dspy.InputField(desc="Input text to analyse.")
    ontology_classes: list[str] = dspy.InputField(
        desc="Allowed entity type labels for source/target classification."
    )
    entity_hints: list[str] = dspy.InputField(
        desc="Pre-identified entities (canonical surface forms) to guide extraction."
    )
    allowed_relation_types: list[str] = dspy.InputField(
        desc=(
            "Exhaustive list of permitted relation labels. "
            "The relation field in every output triplet MUST be one of these."
        )
    )
    candidate_relations: list[dict[str, Any]] = dspy.InputField(
        desc=(
            "Candidate (source, relation, target) triplets produced by the fast "
            "GLiNER2 pass.  Verify these and include confirmed ones in the output."
        )
    )
    triplets: list[Triplet] = dspy.OutputField(
        desc="Verified and enriched list of triplets. relation must be from allowed_relation_types."
    )


class GraphExtractorModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.ChainOfThought(EntityRelation)

    def forward(
        self,
        text: str,
        ontology_classes: list[str],
        entity_hints: list[str],
        allowed_relation_types: list[str] | None = None,
        candidate_relations: list[dict[str, Any]] | None = None,
    ):
        return self.extract(
            text=text,
            ontology_classes=ontology_classes,
            entity_hints=entity_hints,
            allowed_relation_types=allowed_relation_types or [],
            candidate_relations=candidate_relations or [],
        )


# ---------------------------------------------------------------------------
# Grounding helpers (used by DSPyExtractor)
# ---------------------------------------------------------------------------


def _normalize_relation_candidate(value: Any) -> str:
    return standardize_label(value) if value else ""


def _candidate_grounded_in_evidence(candidate: str, evidence: str) -> bool:
    candidate_norm = _normalize_relation_candidate(candidate)
    evidence_norm = _normalize_relation_candidate(evidence)
    if not candidate_norm or not evidence_norm:
        return False
    return f"_{candidate_norm}_" in f"_{evidence_norm}_"


def _endpoint_matches_hint(endpoint: str, entity_hints: list[str]) -> bool:
    endpoint_norm = _normalize_relation_candidate(endpoint)
    if not endpoint_norm:
        return False

    endpoint_tokens = {token for token in endpoint_norm.split("_") if len(token) > 2}
    for hint in entity_hints or []:
        hint_norm = _normalize_relation_candidate(hint)
        if not hint_norm:
            continue
        if endpoint_norm == hint_norm:
            return True
        if hint_norm in endpoint_norm or endpoint_norm in hint_norm:
            return True
        hint_tokens = {token for token in hint_norm.split("_") if len(token) > 2}
        overlap = endpoint_tokens & hint_tokens
        if overlap and (len(overlap) >= 2 or len(endpoint_tokens) == 1 or len(hint_tokens) == 1):
            return True

    return False


def _is_grounded_relation_endpoint(
    endpoint: str,
    entity_hints: list[str],
    ontology_classes: list[str],
    evidence: str,
) -> bool:
    endpoint_norm = _normalize_relation_candidate(endpoint)
    if not endpoint_norm:
        return False

    if _endpoint_matches_hint(endpoint, entity_hints):
        return True

    if _candidate_grounded_in_evidence(endpoint, evidence):
        return True

    ontology_set = {
        _normalize_relation_candidate(item) for item in (ontology_classes or []) if item
    }
    if endpoint_norm in ontology_set:
        return False

    return False


def _is_ungrounded_relation_triplet(
    source: str,
    target: str,
    entity_hints: list[str],
    ontology_classes: list[str],
    evidence: str,
) -> bool:
    return not (
        _is_grounded_relation_endpoint(source, entity_hints, ontology_classes, evidence)
        and _is_grounded_relation_endpoint(target, entity_hints, ontology_classes, evidence)
    )


def _relation_endpoints_in_hints(source: str, target: str, entity_hints: list[str]) -> bool:
    return _endpoint_matches_hint(source, entity_hints) and _endpoint_matches_hint(
        target, entity_hints
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
        keywords: list[str] | None = None,
        entities: list[str] | None = None,
        abstract_concepts: list[str] | None = None,
        allowed_relation_types: list[str] | None = None,
        candidate_relations: list[dict[str, Any]] | None = None,
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

        model = (
            llm_config.get("extraction_model")
            or llm_config.get("base_model")
            or llm_config.get("model")
            or "moonshotai/kimi-k2"
        )

        api_key = _get_openrouter_api_key(config)
        try:
            or_model = normalize_openrouter_model(model)
            logger.info(f"Configuring DSPy for OpenRouter with model {or_model}")
            lm = dspy.LM(
                model=or_model,
                api_key=api_key,
                api_base="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_tokens=2048,
                num_retries=0,
            )
            dspy.configure(lm=lm)
        except Exception as e:
            logger.warning(f"Failed to configure DSPy LM: {e}")

        self.module = GraphExtractorModule()
        logger.info(f"Initialized DSPy extractor with model {model}")

    async def extract_relations(
        self,
        text: str,
        keywords: list[str] | None = None,
        entities: list[str] | None = None,
        abstract_concepts: list[str] | None = None,
        allowed_relation_types: list[str] | None = None,
        candidate_relations: list[dict[str, Any]] | None = None,
    ) -> tuple[list[tuple[str, str, str, dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
        """Extract relations using DSPy."""
        ontology_classes = abstract_concepts or []
        entity_hints = entities or []
        _allowed = allowed_relation_types or []
        _candidates = candidate_relations or []

        try:
            def _extract_sync():
                prediction = self.module(
                    text=text,
                    ontology_classes=ontology_classes,
                    entity_hints=entity_hints,
                    allowed_relation_types=_allowed,
                    candidate_relations=_candidates,
                )
                return prediction.triplets

            triplets = await asyncio.to_thread(_extract_sync)

            relations: list[tuple[str, str, str, dict[str, Any]]] = []
            nodes_data: list[dict[str, Any]] = []
            seen_nodes: set[str] = set()
            raw_triplets: list[dict[str, Any]] = []
            triplet_decisions: list[dict[str, Any]] = []

            if triplets:
                for triplet in triplets:
                    if isinstance(triplet, dict):
                        source = triplet.get("source")
                        relation = triplet.get("relation")
                        target = triplet.get("target")
                        source_type = triplet.get("source_type")
                        target_type = triplet.get("target_type")
                        confidence = triplet.get("confidence", 1.0)
                        evidence = triplet.get("evidence", "")
                    else:
                        source = getattr(triplet, "source", None)
                        relation = getattr(triplet, "relation", None)
                        target = getattr(triplet, "target", None)
                        source_type = getattr(triplet, "source_type", None)
                        target_type = getattr(triplet, "target_type", None)
                        confidence = getattr(triplet, "confidence", 1.0)
                        evidence = getattr(triplet, "evidence", "")

                    raw_triplet: dict[str, Any] = {
                        "source": source,
                        "relation": relation,
                        "target": target,
                        "source_type": source_type,
                        "target_type": target_type,
                        "confidence": confidence,
                        "evidence": evidence,
                    }
                    raw_triplets.append(raw_triplet)

                    if not (source and relation and target):
                        triplet_decisions.append(
                            {
                                **raw_triplet,
                                "kept": False,
                                "drop_reason": "missing_triplet_field",
                                "source_matches_hint": _endpoint_matches_hint(source, entity_hints),
                                "target_matches_hint": _endpoint_matches_hint(target, entity_hints),
                                "source_grounded": _is_grounded_relation_endpoint(
                                    source, entity_hints, ontology_classes, evidence
                                ),
                                "target_grounded": _is_grounded_relation_endpoint(
                                    target, entity_hints, ontology_classes, evidence
                                ),
                            }
                        )
                        continue

                    decision: dict[str, Any] = {
                        **raw_triplet,
                        "kept": False,
                        "drop_reason": None,
                        "source_matches_hint": _endpoint_matches_hint(source, entity_hints),
                        "target_matches_hint": _endpoint_matches_hint(target, entity_hints),
                        "source_grounded": _is_grounded_relation_endpoint(
                            source, entity_hints, ontology_classes, evidence
                        ),
                        "target_grounded": _is_grounded_relation_endpoint(
                            target, entity_hints, ontology_classes, evidence
                        ),
                    }

                    if _allowed:
                        rel_norm = standardize_label(relation)
                        allowed_norm = {standardize_label(r) for r in _allowed}
                        if rel_norm not in allowed_norm:
                            decision["drop_reason"] = "relation_type_not_in_taxonomy"
                            triplet_decisions.append(decision)
                            continue

                    if entity_hints and not _relation_endpoints_in_hints(
                        source, target, entity_hints
                    ):
                        decision["drop_reason"] = "endpoint_not_grounded_in_hints"
                        triplet_decisions.append(decision)
                        continue

                    if _is_ungrounded_relation_triplet(
                        source, target, entity_hints, ontology_classes, evidence
                    ):
                        decision["drop_reason"] = "endpoint_not_grounded_after_evidence_check"
                        triplet_decisions.append(decision)
                        continue

                    decision["kept"] = True
                    triplet_decisions.append(decision)

                    source = standardize_label(source)
                    target = standardize_label(target)
                    relation = standardize_label(relation)
                    source_type = standardize_label(source_type) if source_type else "ENTITY"
                    target_type = standardize_label(target_type) if target_type else "ENTITY"

                    props = {"confidence": confidence, "evidence": evidence}
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
                },
            )

        except Exception as e:
            raise RuntimeError(f"DSPy relation extraction failed: {e}") from e


def get_extractor(config: dict[str, Any]) -> BaseExtractor:
    """Return the sole supported extractor: DSPy."""
    return DSPyExtractor(config)
