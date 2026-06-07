"""Lightweight DSPy ontology-guided relation extractor.

The extractor is intentionally small: GLiNER2 provides entity candidates, the
active taxonomy provides allowed node/relation labels, and DSPy predicts only
typed relation triples.  There are no domain routers, prompt warm-starts, or
empty-list fallbacks on model failure.
"""

from __future__ import annotations

import json
import logging
import os
import re
from ast import literal_eval
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class RelationTriple(BaseModel):
    """Typed relation triple predicted by DSPy."""

    head: str = Field(description="Entity name for the relation head.")
    relation: str = Field(description="Allowed relation label from the taxonomy.")
    tail: str = Field(description="Entity name for the relation tail.")
    confidence: float = Field(default=0.5, description="Confidence score from 0.0 to 1.0.")


def _normalise_label(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalise_entity(entity: dict[str, Any]) -> dict[str, Any] | None:
    name = str(entity.get("name") or entity.get("text") or "").strip()
    if not name:
        return None
    etype = str(entity.get("type") or entity.get("label") or "ENTITY").strip()
    score = float(entity.get("score", entity.get("confidence", 1.0)) or 0.0)
    return {"name": name, "type": etype, "score": score}


def _coerce_relations(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("DSPy returned relations as invalid JSON") from exc
        if not isinstance(parsed, list):
            raise ValueError("DSPy returned relations JSON that is not a list")
        return parsed
    if isinstance(value, list):
        return value
    raise ValueError(f"DSPy returned unsupported relations type: {type(value).__name__}")


def _json_from_text(value: str) -> Any:
    """Decode the first JSON object/array embedded in *value*."""
    decoder = json.JSONDecoder()
    starts = [idx for idx in (value.find('{"relations"'), value.find("[")) if idx >= 0]
    for start in sorted(starts):
        try:
            parsed, _ = decoder.raw_decode(value[start:])
            return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid relations JSON found in LM response")


def _recover_relations_from_exception(exc: Exception) -> list[Any]:
    """Recover relation JSON from provider reasoning_content when DSPy parsing fails."""
    message = str(exc)
    for match in re.finditer(r"LM Response:\s*(\{.*?\})\s*Expected", message, flags=re.DOTALL):
        try:
            response = literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        for field in ("text", "reasoning_content"):
            content = response.get(field)
            if not isinstance(content, str) or not content.strip():
                continue
            try:
                parsed = _json_from_text(content)
            except ValueError:
                continue
            if isinstance(parsed, dict) and "relations" in parsed:
                return _coerce_relations(parsed["relations"])
            return _coerce_relations(parsed)
    parsed = _json_from_text(message)
    if isinstance(parsed, dict) and "relations" in parsed:
        return _coerce_relations(parsed["relations"])
    return _coerce_relations(parsed)


def _make_extractor() -> Any:
    """Build a typed DSPy Predict module."""
    import dspy  # type: ignore

    class OntologyRelationSignature(dspy.Signature):  # type: ignore[misc]
        """Extract relations grounded in the supplied text and entity list.

        Rules:
        - Use only entity names from entities for head and tail.
        - Use only labels from allowed_relation_types for relation.
        - Use allowed_node_types only to understand entity categories.
        - Return an empty list when no grounded relation is present.
        """

        text: str = dspy.InputField(desc="Source text to extract relations from.")
        allowed_node_types: list[str] = dspy.InputField(
            desc="Allowed taxonomy entity type labels."
        )
        allowed_relation_types: list[str] = dspy.InputField(
            desc="Allowed taxonomy relation labels."
        )
        entities: list[dict[str, Any]] = dspy.InputField(
            desc="Pre-extracted GLiNER2 entities with name, type, and score."
        )
        relations: list[RelationTriple] = dspy.OutputField(
            desc="Grounded relation triples using only supplied entities and relation labels."
        )

    return dspy.Predict(OntologyRelationSignature)


class OntologyRelExtractor:
    """Extract taxonomy-guided relations from text and GLiNER2 entities.

    Args:
        lm: Optional pre-configured DSPy LM.  When omitted, an OpenRouter LM is
            built from ``OPENROUTER_API_KEY`` and ``EXTRACTION_MODEL``.
        confidence_threshold: Minimum confidence to keep a returned triple.
        taxonomy_file: Optional taxonomy file or bundled short name.
        relation_schema: Optional relation label -> description mapping.
        entity_labels: Optional allowed entity type labels.

    Raises:
        RuntimeError: If DSPy or OpenRouter configuration is missing.
    """

    def __init__(
        self,
        lm: Any | None = None,
        confidence_threshold: float = 0.5,
        taxonomy_file: str | None = None,
        relation_schema: dict[str, str] | None = None,
        entity_labels: list[str] | None = None,
    ) -> None:
        try:
            import dspy  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OntologyRelExtractor requires DSPy to be installed") from exc

        if relation_schema is None or entity_labels is None:
            from graphknows.ingestion.domain.taxonomy import load_taxonomy_artifacts

            artifacts = load_taxonomy_artifacts(taxonomy_file)
            relation_schema = relation_schema or artifacts.relation_schema
            entity_labels = entity_labels or artifacts.entity_labels

        if not relation_schema:
            raise RuntimeError("OntologyRelExtractor requires at least one allowed relation type")
        if not entity_labels:
            raise RuntimeError("OntologyRelExtractor requires at least one allowed node type")

        self._threshold = confidence_threshold
        self._relation_schema = dict(relation_schema)
        self._entity_labels = list(entity_labels)
        self._allowed_relations = {_normalise_label(r): r for r in self._relation_schema}
        self._allowed_node_types = [_normalise_label(t) for t in self._entity_labels]

        if lm is None:
            api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
            if not api_key:
                raise RuntimeError(
                    "OntologyRelExtractor requires OPENROUTER_API_KEY in llm_assisted mode"
                )
            model_name = os.environ.get("EXTRACTION_MODEL", "moonshotai/kimi-k2")
            self._lm = dspy.LM(
                model=f"openrouter/{model_name}",
                api_key=api_key,
                api_base="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_tokens=1200,
                num_retries=0,
            )
            log.info("OntologyRelExtractor: DSPy ready (model=%s)", model_name)
        else:
            self._lm = lm
            log.info("OntologyRelExtractor: DSPy ready (external LM)")

        self._extractor = _make_extractor()

    @property
    def allowed_relation_types(self) -> list[str]:
        """Return the configured relation labels."""
        return list(self._relation_schema.keys())

    @property
    def allowed_node_types(self) -> list[str]:
        """Return the configured entity type labels."""
        return list(self._entity_labels)

    def _validate_relations(
        self,
        relations: list[Any],
        entities: list[dict[str, Any]],
        threshold: float,
    ) -> list[dict[str, Any]]:
        by_name = {e["name"].casefold(): e["name"] for e in entities}
        validated: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for item in relations:
            if isinstance(item, BaseModel):
                data = item.model_dump()
            elif isinstance(item, dict):
                data = item
            else:
                raise ValueError(f"DSPy returned unsupported relation item: {type(item).__name__}")

            head = str(data.get("head") or data.get("subject") or "").strip()
            rel = str(data.get("relation") or data.get("predicate") or "").strip()
            tail = str(data.get("tail") or data.get("object") or "").strip()
            if not head or not rel or not tail:
                continue
            if head.casefold() == tail.casefold():
                continue

            head_name = by_name.get(head.casefold())
            tail_name = by_name.get(tail.casefold())
            rel_name = self._allowed_relations.get(_normalise_label(rel))
            if head_name is None or tail_name is None or rel_name is None:
                continue

            confidence = float(data.get("confidence", 0.5) or 0.0)
            if confidence < threshold:
                continue

            key = (head_name.casefold(), rel_name, tail_name.casefold())
            if key in seen:
                continue
            seen.add(key)
            validated.append(
                {
                    "subject": head_name,
                    "predicate": rel_name,
                    "object": tail_name,
                    "confidence": confidence,
                }
            )

        return validated

    def extract(
        self,
        text: str,
        entities: list[dict[str, Any]],
        ontology_schema: str | dict[str, str] | None = None,
        ontology_id: str = "auto",
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Extract ontology-typed relations from text.

        ``ontology_schema`` is retained for call-site compatibility and is not
        used; the extractor is configured from taxonomy artifacts at creation.

        Raises:
            RuntimeError: If the DSPy call fails or returns malformed output.
        """
        del ontology_schema
        normalised_entities = [
            item for item in (_normalise_entity(e) for e in entities) if item is not None
        ]
        if len(normalised_entities) < 2:
            return []

        threshold = confidence_threshold if confidence_threshold is not None else self._threshold
        try:
            import dspy  # type: ignore

            with dspy.settings.context(lm=self._lm):
                prediction = self._extractor(
                    text=text,
                    allowed_node_types=self._allowed_node_types,
                    allowed_relation_types=list(self._relation_schema.keys()),
                    entities=normalised_entities,
                )
            raw_relations = _coerce_relations(getattr(prediction, "relations", None))
        except Exception as exc:
            try:
                raw_relations = _recover_relations_from_exception(exc)
            except Exception as recover_exc:
                raise RuntimeError(
                    f"OntologyRelExtractor DSPy extraction failed [{ontology_id}]: {exc}"
                ) from exc
            log.warning(
                "OntologyRelExtractor [%s]: recovered relations from LM reasoning_content "
                "after DSPy adapter parse failure: %s",
                ontology_id,
                type(exc).__name__,
            )

        validated = self._validate_relations(raw_relations, normalised_entities, threshold)

        log.info(
            "OntologyRelExtractor [%s]: %d triples from %d entities",
            ontology_id,
            len(validated),
            len(normalised_entities),
        )
        return validated
