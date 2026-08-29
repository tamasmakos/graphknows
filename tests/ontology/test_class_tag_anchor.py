"""Chunk-to-class tagging must obey the anchor law too (issue #158).

``_STMIngestHandler._match_classes`` produces the ``CHUNK-[:MATCHES_CLASS]->
ONTOLOGY_CLASS`` edge from a pure whole-vocabulary cosine lookup, with no
lexical anchor. Cosine over CCO's 1437 class definitions is uncalibrated (see
the docstring of ``graphknows/symbolic/ontology/catalog.py``: median 0.409,
p95 0.493 on real conversation), so every accepted match lands in a
near-uniform ~0.02 band — measured here as 0.56-0.58 — and the top-3 ordering
inside that band is noise: a BFO upper class like "Curvilinear Motion" can
outscore "Dance Studio" on a chunk about a dance studio, sharing no vocabulary
with it at all.

The fix already exists one path over: entity typing anchors on
``graphknows.symbolic.ontology.lexical.evoked`` — a class may be tagged only
if the chunk text LEXICALLY evokes it — and cosine may only RANK what lexical
evidence already admitted. Raising the similarity floor cannot fix this: the
whole band is uncalibrated, so any floor either admits the noise or admits
nothing. Only an independent signal (the chunk's actual words) can tell the
two candidates apart.

These tests are written against the POST-FIX seam: ``_match_classes(self,
embedding, text)``, an added second argument. They fail today because that
signature does not exist yet.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import spacy

from graphknows.ingestion.stm.ingest import _STMIngestHandler
from graphknows.settings import GraphKnowsSettings
from graphknows.symbolic.ontology import OntologyTerm
from graphknows.symbolic.ontology.lexical import usable_index
from graphknows.symbolic.ontology.skos import Concept

_CURVILINEAR = OntologyTerm(
    uri="u:CurvilinearMotion", label="Curvilinear Motion", definition="", kind="class"
)
_DANCE_STUDIO = OntologyTerm(uri="u:DanceStudio", label="Dance Studio", definition="", kind="class")


class _StubIndex:
    """Ontology index whose ``match_classes`` always returns both terms, at
    the degenerate 0.557-0.590 sims issue #158 measured — regardless of what
    the chunk actually says. This is the bug: cosine alone cannot distinguish
    them, so anchoring must come from elsewhere.
    """

    classes: ClassVar[list[OntologyTerm]] = [_CURVILINEAR, _DANCE_STUDIO]
    properties: ClassVar[list[OntologyTerm]] = []

    @property
    def property_labels(self) -> list[str]:
        return []

    def match_classes(
        self, embedding: list[float], top_k: int = 3, min_sim: float = 0.0
    ) -> list[tuple[OntologyTerm, float]]:
        return [(_CURVILINEAR, 0.58), (_DANCE_STUDIO, 0.56)]


class _StubExtractor:
    """Exposes only what ``_ensure_scheme``/``_match_classes`` read: a spaCy
    pipeline (so lexical selection isn't disabled) and ``parse``.
    """

    def __init__(self, nlp: Any) -> None:
        self.nlp = nlp

    def parse(self, text: str) -> Any:
        return self.nlp(text)


class _FakeStore:
    async def write_class_matches(self, chunk_id: str, matches: Any) -> None:
        return None


@pytest.fixture(scope="module")
def nlp() -> Any:
    return spacy.load(GraphKnowsSettings().spacy_model)


@pytest.fixture()
def handler(nlp: Any) -> _STMIngestHandler:
    stub_index = _StubIndex()
    h = _STMIngestHandler(
        store=_FakeStore(),
        extractor=_StubExtractor(nlp),
        embed_fn=None,
        ontology_index=stub_index,
        relation_index=stub_index,
    )
    # Inject a tiny scheme instead of loading CCO: fast, deterministic, and
    # `_ensure_scheme` short-circuits to it because `_corpus_seen` is empty.
    scheme = {
        "Curvilinear Motion": Concept(
            uri="u:CurvilinearMotion",
            pref_label="Curvilinear Motion",
            definition="",
            kind="class",
        ),
        "Dance Studio": Concept(
            uri="u:DanceStudio", pref_label="Dance Studio", definition="", kind="class"
        ),
    }
    h._scheme = scheme
    h._lex_index = usable_index(scheme, {}, doc_count=0)
    return h


def test_only_the_lexically_anchored_class_is_tagged(handler: _STMIngestHandler) -> None:
    """ "Dance Studio" is in the text; "Curvilinear Motion" is not — even though
    its cosine (0.58) beats "Dance Studio"'s (0.56). That is the whole bug.
    """
    text = "We spent the afternoon at the dance studio downtown."

    result = handler._match_classes([0.1, 0.2, 0.3], text)

    assert result == [("u:DanceStudio", 0.56)]


def test_a_chunk_that_evokes_no_class_is_tagged_with_nothing(handler: _STMIngestHandler) -> None:
    """No lexical evidence for either class -> no tags, and a counted
    abstention rather than a silent empty list.
    """
    text = "Thanks, Gina!"

    result = handler._match_classes([0.1, 0.2, 0.3], text)

    assert result == []
    assert handler._abstentions["unanchored_class_tag"] == 1


def test_accepted_tags_keep_their_similarity(handler: _STMIngestHandler) -> None:
    """The surviving tuple still carries the index's real sim, not a stand-in
    like 1.0 — the sim spread stays reportable even once lexis gates entry.
    """
    text = "We spent the afternoon at the dance studio downtown."

    result = handler._match_classes([0.1, 0.2, 0.3], text)

    assert result[0][1] == 0.56
