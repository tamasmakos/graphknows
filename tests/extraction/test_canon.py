"""Unit tests for WordNet verb canonicalization (open-vocab relation labels)."""

from __future__ import annotations

from graphknows.ingestion.extraction.relations.canon import canonical_verb


def test_synset_canonicalization_purchase_to_buy():
    assert canonical_verb("purchase", "VERB") == "buy"


def test_nominalization_adoption_to_adopt():
    assert canonical_verb("adoption", "NOUN") == "adopt"


def test_gerund_noun_camping_to_camp():
    assert canonical_verb("camping", "NOUN") == "camp"


def test_unknown_lemma_passes_through():
    assert canonical_verb("frobnicate", "VERB") == "frobnicate"


def test_multiword_canonical_rejected_for_single_word_input():
    # First-synset first-lemma of 'start' is 'get_down' — keep the input instead.
    assert canonical_verb("start", "VERB") == "start"


def test_stable_verb_unchanged():
    assert canonical_verb("love", "VERB") == "love"


def test_empty_and_garbage_input():
    assert canonical_verb("", "VERB") == ""
    assert canonical_verb("!!!", "VERB") == ""
