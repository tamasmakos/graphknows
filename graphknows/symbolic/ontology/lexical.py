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

from graphknows.symbolic.ontology import senses
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


def _candidate_mentions(doc: object) -> dict[str, tuple[str, int, str]]:
    """``lemma -> (WordNet pos, char offset, surface text)`` of its FIRST token.

    A sibling of :func:`lemma_view`, not a widening of it: ``document_frequency``
    shares that signature and has no use for offsets, so folding this in would
    cost every caller a field it never reads. Only the first occurrence is kept
    — sense-anchoring is a per-lemma decision, not a per-token one. The surface
    text is kept alongside the offset because :func:`senses.sense_lemmas` stars
    it in the mention line, and an inflected surface form ("rushed") is rarely
    the same length as its lemma ("rush").
    """
    out: dict[str, tuple[str, int, str]] = {}
    for token in doc:  # type: ignore[attr-defined]
        if token.is_stop or not token.lemma_.isalpha() or len(token.lemma_) < 3:
            continue
        lemma = token.lemma_.lower()
        if lemma in out:
            continue
        if token.pos_ in {"NOUN", "PROPN"}:
            out[lemma] = ("n", token.idx, token.text)
        elif token.pos_ == "VERB":
            out[lemma] = ("v", token.idx, token.text)
    return out


def _sense_anchored_matches(
    doc: object,
    scheme: dict[str, Concept],
    index: dict[str, set[str]],
    kind: str,
) -> dict[str, float]:
    """WEAK evidence: a mention licensed, by its IN-CONTEXT WordNet sense, to reach a single-token concept whose label it did not literally use.

    Candidates still come only from the lexical index — the embedding chooses
    among a word's senses, it never creates a candidate that was not already a
    key of ``index`` (the anchor law from spec #135). The cheap pure-WordNet
    prefilter below exists so ``senses.sense_lemmas`` — an embedding call — only
    runs on a mention that could possibly matter, never on every noun and verb
    in the chunk.

    ponytail: one mention-line embedding per surviving lemma per chunk; upgrade
    to a batched embed if this shows up in ingest timings.
    """
    wn = senses._wordnet()
    if wn is None:
        return {}
    pos_gate = {"n"} if kind == "class" else {"n", "v"}
    scores: dict[str, float] = defaultdict(float)
    for lemma, (pos, offset, surface) in _candidate_mentions(doc).items():
        if pos not in pos_gate:
            continue
        synset_lemmas = {
            name.replace("_", " ").lower()
            for synset in wn.synsets(lemma, pos=pos)
            for name in synset.lemma_names()
        }
        if not (synset_lemmas & index.keys()):
            continue
        for name in senses.sense_lemmas(lemma, pos, doc.text, offset, surface):  # type: ignore[attr-defined]
            for label in index.get(name, ()):
                concept = scheme.get(label)
                # Upper-ontology plumbing is NOT excluded here: that guard is
                # structural-half-only (see labels.relation_labels' docstring)
                # — a mention whose in-context WordNet sense anchors it to
                # "requires"/"permits" IS the spoken word asserting the
                # property, same as the STRONG branch above.
                if concept is None or concept.kind != kind or len(concept.tokens) != 1:
                    continue
                scores[label] += 1.0
    return scores


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

    The WEAK branch (a one-word concept reached through a mention that isn't
    its literal label) is sense-anchored: see ``_sense_anchored_matches``.
    """
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
        tokens = concept.tokens
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

    for label, weak_score in _sense_anchored_matches(doc, scheme, index, kind).items():
        scores[label] += weak_score
    return dict(scores)


def idf(doc_freq: dict[str, float], lemma: str) -> float:
    """Inverse document frequency, floored so an unseen lemma is merely strong."""
    return math.log(1.0 / max(doc_freq.get(lemma, 0.01), 0.01))
