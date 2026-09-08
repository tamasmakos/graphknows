"""FR-036: the relation verifier claims spans, skips truncated ones, logs.

Three independent behaviours, each a bare bug the old code had:

* ``verifier.py`` anchored every occurrence of a surface to
  ``text.find()``'s first match — two relations mentioning the same text at
  two different points both got the first point's offset. Fixed by claiming
  successive occurrences per surface (see :func:`_claim_span`), while a
  surface mentioned only once still resolves for every relation that
  references it.
* ``_gliner2_verifier.py`` substituted a bogus token span (token 1, or the
  last token) for a head/tail whose characters fall past the encoder's
  512-token truncation window, silently scoring the wrong text. Fixed by
  reporting that span unmapped and skipping the relation.
* ``_lingfeatures.py`` swallowed a missing WordNet corpus (``LookupError``)
  with no trace. Fixed by logging it before returning "".

No network or model download: the DeBERTa checkpoint is stubbed out exactly
like ``tests/ingestion/test_gliner2_verifier_load.py`` and
``tests/memory/test_relation_verifier.py`` already do.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch

from graphknows.ingestion.extraction.relations._gliner2_verifier import (
    RelationVerifier,
    RelationVerifierModel,
    VerifierConfig,
)
from graphknows.ingestion.extraction.relations._lingfeatures import noun_supersense
from graphknows.ingestion.extraction.relations.verifier import RelationVerifierGate

pytestmark = pytest.mark.unit


# --- verifier.py: claiming spans instead of bare text.find() ----------------


def _rel(head: str, relation: str, tail: str) -> dict[str, Any]:
    return {"head": head, "relation": relation, "tail": tail, "source": "relex"}


class _CapturingVerifier:
    """Stands in for the loaded model: records the spans it was asked to score."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def verify(self, text: str, grouped: dict[str, list]) -> dict[str, list]:
        out: dict[str, list] = {}
        for rel_type, instances in grouped.items():
            self.seen.extend(instances)
            out[rel_type] = [{**inst, "verifier_score": 0.9} for inst in instances]
        return out


def _gate_with(stub: _CapturingVerifier) -> RelationVerifierGate:
    gate = RelationVerifierGate("stub", 0.5)
    gate._verifier = stub  # bypass lazy load
    return gate


def test_gate_claims_successive_occurrences_of_a_repeated_surface():
    """Two relations mentioning "Jon" at different points don't both anchor
    to the first "Jon"."""
    text = "Jon met Mel. Later Jon met Gina."
    rels = [_rel("Jon", "MET", "Mel"), _rel("Jon", "MET", "Gina")]
    stub = _CapturingVerifier()

    _gate_with(stub).filter(text, rels)

    starts = [inst["head"]["start"] for inst in stub.seen]
    assert starts == [0, 19]  # the text's two distinct "Jon" mentions


def test_gate_reuses_the_only_occurrence_shared_by_two_relations():
    """A surface mentioned once, referenced as one relation's tail and
    another's head, resolves to that mention for both instead of one going
    unlocatable once the other has "claimed" it."""
    text = "Jon works on dance routines but did not tackle business goals."
    rels = [
        _rel("Jon", "WORK_ON", "dance routines"),
        _rel("dance routines", "TACKLE", "business goals"),
    ]
    stub = _CapturingVerifier()

    _gate_with(stub).filter(text, rels)

    assert len(stub.seen) == 2  # neither relation went unlocatable
    tail0_start = stub.seen[0]["tail"]["start"]
    head1_start = stub.seen[1]["head"]["start"]
    assert tail0_start == head1_start


# --- _gliner2_verifier.py: skip rather than substitute past truncation ------


class _StubEncoder:
    """Stands in for the deberta encoder: needs only .to()/.eval()."""

    def to(self, _device: str) -> _StubEncoder:
        return self

    def eval(self) -> _StubEncoder:
        return self


def _build_verifier(threshold: float = 0.0) -> RelationVerifier:
    config = VerifierConfig(threshold=threshold)
    model = RelationVerifierModel(config)
    return RelationVerifier(model, _StubEncoder(), _StubEncoder(), config, device="cpu")


# offset_mapping for 4 tokens: [CLS](0,0) "abc"(0,3) "de"(4,5) [SEP](0,0) —
# i.e. an encoding truncated after char 5.
_TRUNCATED_OFFSETS = torch.tensor([[0, 0], [0, 3], [4, 5], [0, 0]])


def test_char_to_token_idx_returns_none_past_the_truncation_window():
    verifier = _build_verifier()

    assert verifier._char_to_token_idx(0, 3, _TRUNCATED_OFFSETS) == (1, 2)
    assert verifier._char_to_token_idx(50, 53, _TRUNCATED_OFFSETS) is None


def test_build_features_returns_none_when_a_span_is_past_truncation():
    verifier = _build_verifier()
    embeddings = torch.zeros(4, 768)

    features = verifier._build_features(
        "text",
        {"start": 0, "end": 3},
        {"start": 50, "end": 53},  # past the truncated encoding
        "REL",
        embeddings,
        _TRUNCATED_OFFSETS,
        rel_emb=torch.zeros(768),
    )

    assert features is None


def test_verify_skips_a_relation_past_the_truncation_window(monkeypatch):
    verifier = _build_verifier(threshold=0.0)  # any real score clears the bar
    embeddings = torch.zeros(4, 768)
    monkeypatch.setattr(
        verifier, "_get_token_embeddings", lambda text: (embeddings, _TRUNCATED_OFFSETS)
    )
    monkeypatch.setattr(verifier, "_get_relation_embedding", lambda relation: torch.zeros(768))

    relations = {
        "REL": [
            {"head": {"start": 0, "end": 3}, "tail": {"start": 4, "end": 5}},  # in range
            {"head": {"start": 50, "end": 53}, "tail": {"start": 54, "end": 55}},  # truncated
        ]
    }

    verified = verifier.verify("irrelevant, offsets are stubbed", relations)

    assert len(verified["REL"]) == 1
    assert verified["REL"][0]["head"]["start"] == 0


# --- _lingfeatures.py: log a missing WordNet corpus, don't swallow it -------


def test_missing_wordnet_corpus_is_logged_not_swallowed(monkeypatch, caplog):
    from nltk.corpus import wordnet as wn

    def _raise_lookup_error(*_args: Any, **_kwargs: Any) -> None:
        raise LookupError("Resource wordnet not found.")

    monkeypatch.setattr(wn, "synsets", _raise_lookup_error)
    noun_supersense.cache_clear()

    with caplog.at_level(logging.WARNING):
        result = noun_supersense("a phrase no other test looks up")

    assert result == ""
    assert "wordnet" in caplog.text.lower()
