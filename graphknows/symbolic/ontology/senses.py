"""Sense-anchored WordNet expansion (issue #141).

``skos.lexicalize`` used to stamp a concept's altLabels with the lemmas of its
label's FIRST WordNet synset, at scheme-BUILD time — before any mention
existed to disambiguate against. A one-token concept then fired on that
synonym in ANY sense ("let" minting the permission property ``permits`` from
its LEASING sense, 22% of conv-30's REL edges). This module replaces that with
expansion at MATCH time, where a mention — and therefore a context to
disambiguate with — actually exists: the trigger-starred sentence containing
the word is embedded against the candidate synsets' definitions, and a synset
is accepted only when it is both a confident match and a clear one. Both
failure directions matter here, so abstaining (returning an empty set) is
always the safe default — a missed expansion costs recall, a wrong one mints a
relation nobody said.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# How similar the mention's context must be to a candidate sense's definition
# before that sense is trusted at all. Trades recall (a real but weakly-worded
# sense gets skipped) against precision (a low-similarity "best guess" is not
# actually evidence of that sense). Tune against tests/ontology/test_sense_anchored_expansion.py.
MIN_SENSE_SIM = 0.30

# How much clearer the top sense must be than the runner-up. A thin margin
# means the mention doesn't discriminate between two candidate senses, so
# picking the top one anyway would be a coin flip dressed up as a decision.
# Tune against tests/ontology/test_sense_anchored_expansion.py.
MIN_SENSE_MARGIN = 0.02


@lru_cache(maxsize=1)
def _wordnet() -> Any:
    """The WordNet reader, or ``None`` if the corpus is not provisioned.

    ``nltk.corpus.wordnet`` is a ``LazyCorpusLoader``: importing it never touches
    disk, so the corpus is only resolved on first *use* (e.g. ``wn.NOUN``). The
    guard has to sit around that use, not the import above it.

    A missing corpus is an ENVIRONMENT defect, so this neither downloads it nor
    passes over it quietly. It does not download because this is library code: a
    package installed from PyPI must not reach the network mid-ingest, and
    ``nltk.download()`` without an explicit ``download_dir`` ignores ``NLTK_DATA``
    anyway (see ``scripts/bake_models.py``), so the fetch would land somewhere
    nothing reads. It warns rather than raising because sense-anchored expansion
    is an enrichment: losing it degrades matching instead of breaking it.
    Warning once per process — the cache sees to that — is what keeps the
    degradation visible.
    """
    from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]  # nltk ships no stubs

    try:
        wn.synsets("dog", pos=wn.NOUN)  # probe: forces the lazy corpus load
    except LookupError:
        log.warning(
            "WordNet corpus not found, so sense-anchored expansion is disabled and "
            "relation/entity matching will be measurably worse. Provision it rather "
            "than working around it: `python scripts/bake_models.py` writes it to "
            "NLTK_DATA (%s).",
            os.environ.get("NLTK_DATA") or "unset — nltk's default search path",
        )
        return None
    return wn


_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str) -> list[tuple[int, str]]:
    """``(offset, sentence)`` pairs, offsets into *text*."""
    out: list[tuple[int, str]] = []
    pos = 0
    for part in _SPLIT.split(text.strip()):
        if not part:
            continue
        start = text.index(part, pos)
        out.append((start, part))
        pos = start + len(part)
    return out


def _marked_line(text: str, offset: int, word: str) -> str:
    """The sentence containing *offset*, with the trigger at *offset* starred.

    Sentence-scoped, not line-scoped (issue #161): a line can state several
    events ("Tom got fired... Caroline got hired..."), so a line-wide context
    let one event pull in the other's entities.

    The star still separates two triggers of one word inside one sentence ("I
    gave a book and gave a pen"): naming the trigger cannot, since both are
    'gave', but marking it in the TEXT gives each mention its own input.
    """
    sent_start, sent = next(s for s in _sentences(text) if s[0] <= offset < s[0] + len(s[1]))
    local = offset - sent_start
    return f"{sent[:local]}*{sent[local : local + len(word)]}*{sent[local + len(word) :]}"


def sense_lemmas(word: str, pos: str, text: str, offset: int, surface: str) -> frozenset[str]:
    """The lemma names of the synset this mention of *word* actually uses.

    *word* is the LEMMA (used to look up WordNet synsets); *surface* is the
    actual inflected token at *offset* ("rushed", not "rush") and is what gets
    starred in the mention line — ``_marked_line`` slices by character length,
    so starring with the lemma corrupts every inflected mention ("rush*ed*"
    instead of "*rushed*").

    Empty when the corpus is missing, *word* has no synsets of *pos*, or the
    embedding choice is not confident enough — abstention is always the safe
    direction (see the module docstring). The mention's own sentence, with the
    trigger starred (:func:`_marked_line`), is
    embedded and ranked against each candidate synset's gloss (definition plus
    examples plus lemma names, since a bare definition is often too short to
    embed well); the winner is accepted only above both a similarity floor and
    a margin over the runner-up.
    """
    wn = _wordnet()
    if wn is None:
        return frozenset()
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        return frozenset()

    from graphknows.storage.embedder import embed_one
    from graphknows.symbolic.index import match

    try:
        mention = _marked_line(text, offset, surface)
    except StopIteration:
        log.debug("sense_lemmas: offset %d for %r falls outside every sentence span", offset, word)
        return frozenset()

    ids = [synset.name() for synset in synsets]
    glosses = [
        " ".join(
            [
                synset.definition(),
                *synset.examples(),
                *(lemma.name().replace("_", " ") for lemma in synset.lemmas()),
            ]
        )
        for synset in synsets
    ]
    mat = np.array([embed_one(gloss) for gloss in glosses], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    mat = mat / norms

    ranked = match(embed_one(mention), ids, mat, top_k=2, min_sim=-1.0)
    if not ranked:
        return frozenset()
    top1_name, top1_sim = ranked[0]
    if top1_sim < MIN_SENSE_SIM:
        return frozenset()
    # The margin gate only means something with a runner-up to be clearer than.
    # A single candidate synset has nothing to discriminate against, so it is
    # judged on MIN_SENSE_SIM alone rather than a fake margin against 0.0.
    if len(ranked) > 1 and top1_sim - ranked[1][1] < MIN_SENSE_MARGIN:
        return frozenset()

    synset = next(s for s in synsets if s.name() == top1_name)
    log.info("sense-anchored expansion: %r -> %s", word, synset.name())
    return frozenset(lemma.name().replace("_", " ").lower() for lemma in synset.lemmas())
