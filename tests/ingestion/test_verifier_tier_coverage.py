"""Verification is a property of an edge, not of the producer that minted it.

``RelationVerifierGate.filter`` is called inside the relex extractor, so only
relex candidates can ever be scored. Every ingest-time REL edge, whatever tier
produced it, is written by ``_STMIngestHandler._write_relations`` — and its
second loop (the DSPy/llm_assisted tier) writes ``source="dspy"`` with no gate
and no ``verifier_score`` at all. A tier that cannot be judged is a tier whose
false positives reach the graph unopposed.

What these tests pin:
  * gated at the funnel — the DSPy tier goes through the same gate as relex, so
    a rejected triple is never written and a survivor carries its score
  * no gate never deletes — with no verifier available the tier still writes,
    unscored; a missing model must degrade to "unjudged", not to "dropped"
  * coverage is countable — ``verifier_coverage_rows`` aggregates edges/scored/
    mean per source, so a future producer that skips the gate shows up as a
    number instead of a surprise. "Never scored" reports ``None``, not 0.0,
    keeping it distinguishable from "scored zero" exactly as ``write_relation``
    already promises.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.ingestion.stm.ingest import _STMIngestHandler

CHUNK_ID = "chunk-1"
TEXT = "Gina works for Acme and spent the evening dancing."

ENTITIES = [
    {"name": "Gina", "type": "PERSON"},
    {"name": "Acme", "type": "ORGANIZATION"},
    {"name": "Dancing", "type": "ACTIVITY"},
]

# (subject, predicate, object, props) as the DSPy relation extractor yields them.
DSPY_TRIPLES = [
    ("Gina", "WORKS_FOR", "Acme", {"evidence": "Gina works for Acme"}),
    ("Gina", "DO_WITH", "Dancing", {}),
]

ALLOWED = {("Gina", "Acme")}


class _FakeStore:
    """Records every write instead of performing it."""

    def __init__(self) -> None:
        self.relations: list[dict[str, Any]] = []

    async def write_relation(self, head, relation, tail, **kwargs: Any) -> None:
        self.relations.append({"head": head, "relation": relation, "tail": tail, **kwargs})

    def by_predicate(self, predicate: str) -> list[dict[str, Any]]:
        return [r for r in self.relations if r["relation"] == predicate]


class _FakeGate:
    """Keeps only the allowed (head, tail) pairs, scoring the survivors."""

    def filter(self, text: str, relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept = [r for r in relations if (r["head"], r["tail"]) in ALLOWED]
        for rel in kept:
            rel["verifier_score"] = 0.9
        return kept


class _StubExtractorWithGate:
    """Stands in for the entity extractor, which owns the loaded gate.

    Exposes the gate under both the public and the private name: the real
    extractor stores it as ``_relation_verifier``, and this test is about the
    edge being scored, not about which attribute the funnel reads.
    """

    def __init__(self) -> None:
        gate = _FakeGate()
        self.relation_verifier = gate
        self._relation_verifier = gate


class _StubExtractorNoGate:
    """The extractor of a build where the verifier is disabled or failed to load."""


class _StubRelExtractor:
    """DSPy relation extractor stub — yields the same triples every time."""

    async def extract_relations(self, text: str) -> list[tuple[str, str, str, dict[str, Any]]]:
        return list(DSPY_TRIPLES)


def _handler(store: _FakeStore, extractor: Any) -> _STMIngestHandler:
    return _STMIngestHandler(
        store=store, extractor=extractor, embed_fn=None, rel_extractor=_StubRelExtractor()
    )


@pytest.mark.asyncio
async def test_dspy_tier_relations_are_verifier_gated_like_relex():
    """A DSPy triple the verifier rejects must not reach the graph; a kept one carries its score."""
    store = _FakeStore()
    handler = _handler(store, _StubExtractorWithGate())

    await handler._write_relations(
        ExtractionResult(entities=ENTITIES, relations=[]), CHUNK_ID, TEXT
    )

    kept = store.by_predicate("WORKS_FOR")
    assert len(kept) == 1, "the verified DSPy relation must still be written"
    assert kept[0].get("verifier_score") == pytest.approx(0.9)
    assert store.by_predicate("DO_WITH") == [], "the rejected DSPy relation must not be written"


@pytest.mark.asyncio
async def test_dspy_tier_writes_unscored_when_no_verifier_is_available():
    """No gate means unjudged, never deleted — a missing model must not lose facts."""
    store = _FakeStore()
    handler = _handler(store, _StubExtractorNoGate())

    await handler._write_relations(
        ExtractionResult(entities=ENTITIES, relations=[]), CHUNK_ID, TEXT
    )

    written = {r["relation"] for r in store.relations}
    assert written == {"WORKS_FOR", "DO_WITH"}
    assert all(r.get("verifier_score") is None for r in store.relations)


def test_verifier_coverage_rows_aggregates_per_tier():
    """Coverage per source: how many edges, how many judged, and how strongly."""
    from graphknows.storage.arcadedb.graph_store import verifier_coverage_rows

    coverage = verifier_coverage_rows(
        [
            {"source": "gliner", "verifier_score": 0.6},
            {"source": "gliner", "verifier_score": 0.8},
            {"source": "gliner", "verifier_score": None},
            {"source": "dspy", "verifier_score": None},
            {"source": "dspy", "verifier_score": None},
            {"source": "", "verifier_score": 0.5},
        ]
    )

    assert coverage["gliner"] == {"edges": 3, "scored": 2, "mean": pytest.approx(0.7)}
    assert coverage["dspy"]["edges"] == 2
    assert coverage["dspy"]["scored"] == 0
    # "never scored" must stay distinguishable from "scored zero".
    assert coverage["dspy"]["mean"] is None
    assert coverage["unknown"] == {"edges": 1, "scored": 1, "mean": pytest.approx(0.5)}
