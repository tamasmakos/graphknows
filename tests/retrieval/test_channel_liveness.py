"""Liveness: the BM25 channel must actually be BM25, not a term-frequency count.

The defect class this repo keeps hitting is not a crash. It is code that runs,
passes and does nothing: a cross-encoder that was a silent no-op because its
dependency was absent, a `verifier_score` computed and discarded, and — for
months — a "BM25" channel that scored with a term-frequency fallback because
`rank-bm25` was never declared as a dependency, so it was never installed, so
the `except ImportError` branch took over and said nothing.

Every existing BM25 test passes under that fallback. "An exact match outranks a
non-match" and "two query terms beat one" are true of any scorer that counts
terms, so a green suite told us nothing about which scorer was running.

These tests are chosen so the term-frequency answer is the OPPOSITE of the BM25
answer. Each one fails the moment the channel degrades to counting:

  * IDF        a rare term beats a common one, whatever the counts
  * length     at equal counts, the shorter document wins
  * negative   a term in most of the corpus scores worse the more it repeats

Measured against the live BM25Okapi rather than assumed, so the margins are
real: in the IDF case the rare-term document scores 2.13 against 0.04.
"""

from __future__ import annotations

import pytest

from graphknows.retrieval.retriever import _bm25_rank

pytestmark = pytest.mark.unit


def _ranked_ids(query: list[str], chunks: list[dict[str, str]]) -> list[str]:
    return [c["chunk_id"] for _, c in _bm25_rank(query, chunks)]


def test_rare_term_beats_repeated_common_term() -> None:
    """IDF is live: one hit on a rare term outranks five on a common one.

    A term-frequency scorer ranks `common_many` (5 occurrences) first. BM25
    weights by rarity, so the single rare hit wins by two orders of magnitude.
    """
    corpus = [
        {"chunk_id": f"filler{i}", "text": "meeting notes about the meeting agenda"}
        for i in range(8)
    ]
    chunks = [
        {"chunk_id": "rare_once", "text": "she brought her viola"},
        {"chunk_id": "common_many", "text": "meeting meeting meeting meeting meeting"},
        *corpus,
    ]

    ids = _ranked_ids(["meeting", "viola"], chunks)

    assert ids[0] == "rare_once", (
        "the document matching the RARE term must rank first; ranking the "
        "repeated common term first means the channel is counting, not scoring"
    )


def test_shorter_document_wins_at_equal_term_count() -> None:
    """Length normalisation is live: identical counts, shorter document first.

    A term-frequency scorer ties these — both contain the term exactly once.
    """
    chunks = [
        {"chunk_id": "short", "text": "viola practice"},
        {"chunk_id": "long", "text": "viola " + " ".join(["padding"] * 60)},
        {"chunk_id": "other", "text": "unrelated text here"},
    ]

    ranked = _bm25_rank(["viola"], chunks)
    scores = {c["chunk_id"]: s for s, c in ranked}

    assert scores["short"] > scores["long"], (
        "equal term counts must be separated by document length; a tie means "
        "no length normalisation, i.e. not BM25"
    )


def test_repeating_a_corpus_wide_term_scores_worse() -> None:
    """Okapi IDF goes negative for a term in most of the corpus.

    Once a term appears in more than about half the documents, `log((N - df +
    0.5)/(df + 0.5))` is negative, so *more* occurrences means a *lower* score.
    A term-frequency scorer ranks these exactly the other way round. This is
    also the shape that made a `score > 0` filter empty the channel entirely.
    """
    chunks = [
        {"chunk_id": "one", "text": "viola " + " ".join(["x"] * 20)},
        {"chunk_id": "ten", "text": " ".join(["viola"] * 10) + " " + " ".join(["x"] * 11)},
        {"chunk_id": "none", "text": " ".join(["y"] * 21)},
    ]

    ranked = _bm25_rank(["viola"], chunks)
    ids = [c["chunk_id"] for _, c in ranked]
    scores = {c["chunk_id"]: s for s, c in ranked}

    assert ids, "documents containing the term must not be filtered out"
    assert scores["one"] > scores["ten"], (
        "with a corpus-wide term, repetition must lower the score; the "
        "opposite ordering is what a counting fallback produces"
    )
