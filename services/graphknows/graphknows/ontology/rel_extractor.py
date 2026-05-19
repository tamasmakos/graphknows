"""DSPy-based ontology-guided relation extractor."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)


class OntologyRelExtractor:
    """Extract typed relations from text guided by an ontology schema.

    Uses DSPy ChainOfThought with Groq as the backend. The ontology's
    JSON-LD summary is injected into the prompt to constrain relation types
    to those defined in the active ontology.

    Args:
        lm: A configured DSPy LM instance. If None, uses the global default.
    """

    def __init__(self, lm: Any | None = None) -> None:
        try:
            import dspy  # type: ignore

            self._dspy = dspy
            self._extractor = dspy.ChainOfThought(
                "text, entities_json, ontology_schema -> relations_json"
            )
            if lm is not None:
                dspy.settings.configure(lm=lm)
            elif os.environ.get("OPENROUTER_API_KEY"):
                import dspy as _dspy

                _lm = _dspy.LM(
                    model="openrouter/moonshotai/kimi-k2",
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    temperature=0.1,
                    max_tokens=1024,
                )
                _dspy.settings.configure(lm=_lm)
            self._available = True
            log.info("OntologyRelExtractor: DSPy ready")
        except ImportError:
            log.warning("DSPy not installed — OntologyRelExtractor disabled")
            self._available = False

    def extract(
        self,
        text: str,
        entities: list[dict],
        ontology_schema: str,
        ontology_id: str = "unknown",
    ) -> list[dict]:
        """Extract ontology-typed relations from text.

        Args:
            text: Source document text (first 1500 chars used).
            entities: List of entity dicts with keys ``name`` and ``label``.
            ontology_schema: JSON-LD summary string for the active ontology.
            ontology_id: Ontology identifier for logging.

        Returns:
            List of relation dicts: ``{subject, predicate, object, confidence}``.
            Returns empty list on failure.
        """
        if not self._available or not entities:
            return []

        try:
            schema_data = json.loads(ontology_schema)
            properties = schema_data.get("properties", [])[:30]
            relations_hint = schema_data.get("typical_relations", [])[:5]
            schema_compact = json.dumps(
                {
                    "relation_types": properties,
                    "examples": relations_hint,
                }
            )
        except Exception:
            schema_compact = ontology_schema[:800]

        entity_names = [e.get("name", "") for e in entities if e.get("name")]
        entity_names_str = json.dumps(entity_names[:20])

        try:
            pred = self._extractor(
                text=text[:1500],
                entities_json=entity_names_str,
                ontology_schema=schema_compact,
            )
            raw = getattr(pred, "relations_json", "") or ""
            # Strip markdown code fences if model wraps output
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            relations = json.loads(raw)
            if not isinstance(relations, list):
                return []
            validated: list[dict] = []
            for r in relations:
                if (
                    isinstance(r, dict)
                    and r.get("subject")
                    and r.get("predicate")
                    and r.get("object")
                ):
                    validated.append(
                        {
                            "subject": str(r["subject"]),
                            "predicate": str(r["predicate"]),
                            "object": str(r["object"]),
                            "confidence": float(r.get("confidence", 0.5)),
                        }
                    )
            log.debug(
                "OntologyRelExtractor [%s]: %d relations from %d entities",
                ontology_id,
                len(validated),
                len(entities),
            )
            return validated
        except Exception as exc:
            log.warning("OntologyRelExtractor extraction failed: %s", exc)
            return []
