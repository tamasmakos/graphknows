"""Tests for graphgen relation extraction behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from graphknows.ingestion.extraction.relations.llm_assisted.dspy import DSPyExtractor


@pytest.mark.asyncio
async def test_extract_relations_treats_empty_lm_response_as_no_relations() -> None:
    """An empty provider response should not make LTM promotion fail."""
    extractor = DSPyExtractor.__new__(DSPyExtractor)
    extractor._lm = MagicMock()
    extractor._ontology = None
    extractor._ontology_classes = ["PERSON"]
    extractor._ontology_relation_types = ["SUPPORTS"]
    extractor.module = MagicMock(
        side_effect=ValueError(
            "The LM returned an empty or null response. "
            "Adapter JSONAdapter failed to parse the LM response."
        )
    )

    assert await extractor.extract_relations("Alice supports Bob.") == []
