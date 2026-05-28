"""
Graph Extractor — DSPy + GLiNER2 backend.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import logging
import asyncio
import time

from graphknows.graphgen.config.llm import (
    _extract_secret,
    normalize_openrouter_model,
    _get_openrouter_api_key,
)
from graphknows.graphgen.pipeline.entity_relation.dspy_module import GraphExtractorModule
import dspy
import os

from graphknows.graphgen.utils.utils import standardize_label

logger = logging.getLogger(__name__)


def _normalize_relation_candidate(value: Any) -> str:
    return standardize_label(value) if value else ""


def _candidate_grounded_in_evidence(candidate: str, evidence: str) -> bool:
    candidate_norm = _normalize_relation_candidate(candidate)
    evidence_norm = _normalize_relation_candidate(evidence)
    if not candidate_norm or not evidence_norm:
        return False
    return f"_{candidate_norm}_" in f"_{evidence_norm}_"


def _endpoint_matches_hint(endpoint: str, entity_hints: List[str]) -> bool:
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
    entity_hints: List[str],
    ontology_classes: List[str],
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
    entity_hints: List[str],
    ontology_classes: List[str],
    evidence: str,
) -> bool:
    return not (
        _is_grounded_relation_endpoint(source, entity_hints, ontology_classes, evidence)
        and _is_grounded_relation_endpoint(target, entity_hints, ontology_classes, evidence)
    )


def _relation_endpoints_in_hints(source: str, target: str, entity_hints: List[str]) -> bool:
    return _endpoint_matches_hint(source, entity_hints) and _endpoint_matches_hint(
        target, entity_hints
    )


class BaseExtractor(ABC):
    """Abstract base for graph relation extractors."""

    @abstractmethod
    async def extract_relations(
        self,
        text: str,
        keywords: List[str] = None,
        entities: List[str] = None,
        abstract_concepts: List[str] = None,
        allowed_relation_types: Optional[List[str]] = None,
        candidate_relations: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Tuple[str, str, str, Dict[str, Any]]], List[Dict[str, Any]], Dict[str, Any]]:
        """Extract (source, relation, target, props) triplets from *text*."""
        pass

    async def close(self):
        """Cleanup resources."""
        pass


class DSPyExtractor(BaseExtractor):
    """DSPy-based extractor."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with config."""
        self.config = config

        # Configure DSPy with the LLM from config
        llm_config = config.get("llm", {})
        # Flatten if it's a pydantic model dump
        if hasattr(llm_config, "model_dump"):
            llm_config = llm_config.model_dump()

        # Try to find the best model name from config
        model = (
            llm_config.get("extraction_model")
            or llm_config.get("base_model")
            or llm_config.get("model")
            or "moonshotai/kimi-k2"
        )

        # Configure DSPy for OpenRouter
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
            # Fallback or already configured
            pass

        self.module = GraphExtractorModule()
        logger.info(f"Initialized DSPy extractor with model {model}")

    async def extract_relations(
        self,
        text: str,
        keywords: List[str] = None,
        entities: List[str] = None,
        abstract_concepts: List[str] = None,
        allowed_relation_types: Optional[List[str]] = None,
        candidate_relations: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Tuple[str, str, str, Dict[str, Any]]], List[Dict[str, Any]], Dict[str, Any]]:
        """Extract relations using DSPy."""

        ontology_classes = abstract_concepts or []
        entity_hints = entities or []
        _allowed = allowed_relation_types or []
        _candidates = candidate_relations or []

        try:
            # DSPy calls are synchronous, so we run in a thread
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

            # Convert to expected format
            relations = []
            nodes_data = []
            seen_nodes = set()
            raw_triplets = []
            triplet_decisions = []

            # triplets is expected to be a list of Triplet objects (pydantic models)
            if triplets:
                for triplet in triplets:
                    # Handle both dict and object access just in case
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

                    raw_triplet = {
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

                    decision = {
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

                    # --- Relation-type allowlist gate ---
                    # Only enforce when the caller supplied an explicit allowlist.
                    if _allowed:
                        rel_norm = standardize_label(relation)
                        allowed_norm = {standardize_label(r) for r in _allowed}
                        if rel_norm not in allowed_norm:
                            decision["drop_reason"] = "relation_type_not_in_taxonomy"
                            triplet_decisions.append(decision)
                            logger.debug(
                                "Dropping DSPy triplet with unknown relation type %r (allowed: %s)",
                                relation,
                                _allowed,
                            )
                            continue

                    if entity_hints and not _relation_endpoints_in_hints(
                        source, target, entity_hints
                    ):
                        decision["drop_reason"] = "endpoint_not_grounded_in_hints"
                        triplet_decisions.append(decision)
                        logger.debug(
                            "Dropping DSPy triplet without both endpoints grounded in hints source=%s relation=%s target=%s",
                            source,
                            relation,
                            target,
                        )
                        continue
                    if _is_ungrounded_relation_triplet(
                        source,
                        target,
                        entity_hints,
                        ontology_classes,
                        evidence,
                    ):
                        decision["drop_reason"] = "endpoint_not_grounded_after_evidence_check"
                        triplet_decisions.append(decision)
                        logger.debug(
                            "Dropping ungrounded DSPy triplet source=%s relation=%s target=%s evidence=%s",
                            source,
                            relation,
                            target,
                            evidence,
                        )
                        continue

                    decision["kept"] = True
                    triplet_decisions.append(decision)

                    # Standardize upstream
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


def get_extractor(config: Dict[str, Any]) -> BaseExtractor:
    """Return the sole supported extractor: DSPy."""
    return DSPyExtractor(config)
