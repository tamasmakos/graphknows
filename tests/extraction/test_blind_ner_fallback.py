"""Extraction must not give up when spaCy's NER observes nothing.

The entity label set handed to relex is the chunk's OBSERVED spaCy labels
UNIONed with a 20-label base inventory (activity/emotion/occupation/animal/...).
Short-circuiting on an empty observation therefore skips the fallback that
exists precisely for chunks spaCy cannot read — and conversational text is
exactly that case. Measured on LoCoMo conv-30: 170 of 369 turns (46%) observed
no spaCy label and so produced no entities at all, including plain statements
of fact like "I'm starting a dance studio".
"""

from __future__ import annotations

import pytest

from graphknows.ingestion.extraction.entities.extractor import GLiNER2EntityExtractor

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def extractor() -> GLiNER2EntityExtractor:
    return GLiNER2EntityExtractor()


@pytest.mark.parametrize(
    "text",
    [
        "I'm starting a dance studio 'cause I'm passionate about dancing.",
        "I've been into dancing since I was a kid and it's been my passion.",
        "My cat Mochi is a ragdoll.",
    ],
)
def test_chunk_with_no_spacy_ner_still_extracts(
    extractor: GLiNER2EntityExtractor, text: str
) -> None:
    assert not extractor._schema_miner.build(text).entity_spec, (
        "fixture assumes spaCy observes nothing here"
    )
    result = extractor.extract(text)
    assert result.entities, "base label inventory should still yield entities"


def test_empty_text_still_returns_nothing(extractor: GLiNER2EntityExtractor) -> None:
    assert extractor.extract("   ").entities == []
