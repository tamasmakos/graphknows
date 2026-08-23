"""Lexical matching of text to ontology concepts — no embeddings.

The first cut matched altLabels as substrings with a suffix wildcard, and real
conversation destroyed it: "sounds awesome" evoked ``Sound Pressure``,
``Sound Wavelength`` and ``Measurement Unit of Sound Level``; "'cause" evoked
``caused by``; "starting a business" evoked ``process starts`` and
``interval started by``; "sorry about your job" evoked ``is about``.

Every one of those is the same failure in two parts, so this module fixes both.

**Part 1 — a word is not a string.** Matching is on spaCy LEMMAS with the
part of speech gated: a class concept may only be evoked by a NOUN, a property
concept only by a VERB. "sounds" as a verb can then never reach the acoustics
concept, and the suffix wildcard that let "start" match "starting" is replaced by
lemmatisation, which is what it was badly approximating.

**Part 2 — a word that means everything discriminates nothing.** ``about``,
``cause``, ``start``, ``change``, ``act`` are ordinary discourse vocabulary, and
an ontology that claims them will fire on every chunk. Rather than hand-listing
them, altLabels are scored by DOCUMENT FREQUENCY over the corpus itself and the
ubiquitous ones are dropped — the same principle as this repo's
``_discriminative_entities`` (drop entities appearing in >50% of chunks), applied
one layer earlier. It is data-driven, needs no curation, and adapts to whatever
corpus is being ingested.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from graphknows.symbolic.ontology.skos import Concept

# A lemma occurring in more than this fraction of chunks carries no information
# about any individual chunk. 0.15 is deliberately aggressive: the failure being
# fixed is confident nonsense, and a missed tag costs less than a wrong one.
MAX_DOC_FREQ = 0.15


def lemma_view(doc: object) -> tuple[set[str], set[str], set[str]]:
    """``(noun lemmas, verb lemmas, noun compounds)`` for one parsed chunk.

    Compounds are adjacent noun pairs ("dance studio", "clothing store"). They
    are what make a match specific enough to trust, so they are collected
    separately and weighted far above single tokens by the caller.
    """
    nouns: set[str] = set()
    verbs: set[str] = set()
    compounds: set[str] = set()
    tokens = [t for t in doc]  # type: ignore[attr-defined]
    for i, token in enumerate(tokens):
        if token.is_stop or not token.lemma_.isalpha() or len(token.lemma_) < 3:
            continue
        lemma = token.lemma_.lower()
        if token.pos_ in {"NOUN", "PROPN"}:
            nouns.add(lemma)
            if i + 1 < len(tokens) and tokens[i + 1].pos_ in {"NOUN", "PROPN"}:
                compounds.add(f"{lemma} {tokens[i + 1].lemma_.lower()}")
        elif token.pos_ == "VERB":
            verbs.add(lemma)
    return nouns, verbs, compounds


def document_frequency(docs: list[object]) -> dict[str, float]:
    """Fraction of chunks each lemma appears in — the discriminativeness signal."""
    counts: Counter[str] = Counter()
    for doc in docs:
        nouns, verbs, compounds = lemma_view(doc)
        counts.update(nouns | verbs | compounds)
    total = max(len(docs), 1)
    return {lemma: n / total for lemma, n in counts.items()}


def usable_index(
    scheme: dict[str, Concept],
    doc_freq: dict[str, float],
    max_df: float = MAX_DOC_FREQ,
    doc_count: int = 0,
    min_docs: int = 3,
) -> dict[str, set[str]]:
    """AltLabel -> prefLabels, minus the altLabels too common to mean anything.

    An altLabel unseen in the corpus is KEPT: absence of evidence is not
    ubiquity, and it simply never fires.

    ``min_docs`` is what makes this safe on a short corpus. A frequency alone is
    meaningless when there are five documents — one occurrence is already 20%,
    above the threshold, so every content word would be discarded and the
    vocabulary would empty. Requiring an absolute count as well means the filter
    only starts removing words once there is enough text to judge them by.
    """
    index: dict[str, set[str]] = defaultdict(set)
    for concept in scheme.values():
        for alt in concept.alt_labels:
            freq = doc_freq.get(alt, 0.0)
            ubiquitous = freq > max_df and (not doc_count or freq * doc_count >= min_docs)
            if not ubiquitous:
                index[alt].add(concept.pref_label)
    return index


def evoked(
    doc: object,
    scheme: dict[str, Concept],
    index: dict[str, set[str]],
    kind: str,
) -> dict[str, float]:
    """Concepts of ``kind`` this chunk lexically evokes, scored by evidence.

    Weighting encodes what makes a lexical match trustworthy: a two-word
    compound is worth far more than a single noun (it is much less likely to be
    coincidence), and a rarer lemma is worth more than a common one — plain IDF,
    reusing the document frequencies already computed.
    """
    from graphknows.symbolic.ontology.skos import content_tokens

    nouns, verbs, compounds = lemma_view(doc)
    # A property matches on NOUNS as well as verbs. CCO names its relations as
    # relational nouns — "has brother", "is spouse of", "has mother" — so the
    # word that evokes them in speech ("my brother Daniel") is a noun, and a
    # verb-only gate can never fire on the exact cases this corpus is full of.
    present = (nouns | compounds) if kind == "class" else (nouns | verbs | compounds)

    # float-valued, so a compound match can outweigh a single token; Counter is
    # int-valued and would silently truncate the weighting to nothing.
    scores: dict[str, float] = defaultdict(float)
    for label, concept in scheme.items():
        if concept.kind != kind:
            continue
        # An overlay concept carries AUTHORED altLabels: a human said these
        # words evoke it, which is stronger evidence than any derivation, so any
        # one of them firing is enough.
        if concept.module == "overlay":
            hits = sum(1 for alt in concept.alt_labels if alt in present)
            if hits:
                scores[label] += 3.0 * hits
            continue
        tokens = content_tokens(label)
        if not tokens:
            continue
        # STRONG: every content word of the label is in the chunk. This is the
        # rule that killed the last round of nonsense — ``Round Shot`` fired on
        # "take a shot at", ``Calendar Month`` on "this month", because a single
        # token of a multi-word label was enough. Requiring the whole label
        # makes a match mean the ontology's term, not one of its words. IDF
        # cannot catch these: "shot" is genuinely rare in this corpus.
        if all(t in present for t in tokens):
            scores[label] += 3.0 * len(tokens)
            continue
        # WEAK: a one-word concept reached through a dominant-sense synonym
        # (``Hospital`` via "infirmary"). Never applies to multi-word labels,
        # where partial evidence is exactly the failure above.
        if len(tokens) == 1:
            synonyms = concept.alt_labels - set(tokens)
            if any(alt in present and alt in index for alt in synonyms):
                scores[label] += 1.0
    return dict(scores)


def idf(doc_freq: dict[str, float], lemma: str) -> float:
    """Inverse document frequency, floored so an unseen lemma is merely strong."""
    return math.log(1.0 / max(doc_freq.get(lemma, 0.01), 0.01))
