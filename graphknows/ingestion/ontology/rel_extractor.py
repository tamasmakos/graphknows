"""Strict RDF-backed DSPy relation extractor.

This module intentionally does not accept taxonomy YAML, GLiNER2 entity hints,
or candidate relations.  It delegates to the graphgen DSPy extractor, which
loads one explicit RDF/OWL ontology file and constrains relation labels from
that file's RDF properties.
"""

from __future__ import annotations

import os
from typing import Any


class OntologyRelExtractor:
    """Compatibility import name for the RDF-only DSPy extractor.

    Args:
        ontology_file: Required RDF/OWL ontology file path.
        lm: Unsupported; DSPyExtractor owns OpenRouter LM construction.
        confidence_threshold: Retained as a validation threshold for returned
            relations after DSPy extraction.

    Raises:
        RuntimeError: If no RDF/OWL ontology file is supplied.
    """

    def __init__(
        self,
        ontology_file: str,
        lm: Any | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        if lm is not None:
            raise RuntimeError("OntologyRelExtractor no longer accepts an injected LM")
        if not ontology_file:
            raise RuntimeError("OntologyRelExtractor requires one RDF/OWL ontology_file")

        from graphknows.graphgen.extraction.relations import DSPyExtractor

        self._threshold = confidence_threshold
        self._extractor = DSPyExtractor(
            config={
                "llm": {"model": os.environ.get("EXTRACTION_MODEL", "moonshotai/kimi-k2")},
                "infra": {"openrouter_api_key": os.environ.get("OPENROUTER_API_KEY", "")},
                "ontology": {"file": ontology_file},
            }
        )

    async def extract_async(self, text: str) -> list[dict[str, Any]]:
        """Extract RDF-constrained relations from text."""
        relations, _nodes, _audit = await self._extractor.extract_relations(text=text)
        return [
            {
                "subject": source,
                "predicate": relation,
                "object": target,
                **metadata,
            }
            for source, relation, target, metadata in relations
            if float(metadata.get("confidence", 1.0) or 0.0) >= self._threshold
        ]

