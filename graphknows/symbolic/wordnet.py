"""WordNet domain logic for the schema plane (issue #142).

Pure domain: id minting, most-frequent-sense (MFS) baseline WSD, the shortest
hypernym ancestor chain, and the synset/sense relation split. No graph writes
and no I/O beyond the corpus reader — the writers that turn these dicts into
vertices/edges live in ``GraphStore``; the write seams that call them live in
the ingest pipeline. Keeping this module I/O-free is what makes it testable
without a store or a mocked client.

Reuses :func:`graphknows.symbolic.ontology.senses._wordnet` for the corpus
probe rather than duplicating it — that function already owns the "warn once,
never download" contract for an unprovisioned corpus.
"""

from __future__ import annotations

from graphknows.symbolic.ontology.senses import _wordnet

# WordNet synset OFFSETS are not stable across releases (they can shift when
# the corpus is rebuilt), so every id this module mints is release-prefixed —
# an id from 3.0 must never be silently treated as identical to one from a
# future 3.1.
ID_PREFIX = "wn30:"

# The WordNet release this module is built against. Carried on groundings and
# the SOURCE row so a later release migration has something to key off.
WN_VERSION = "3.0"

# Safety cap on a hypernym ancestor chain. WordNet's deepest noun chain is
# ~19 hops; 24 gives headroom without allowing a pathological or corrupted
# corpus to mint an unbounded number of ancestor nodes. Truncation is
# reported back to the caller (never silent) via ancestors()'s bool return.
MAX_HYPERNYM_DEPTH = 24

# Relations that hold between SYNSETS (concepts), never between individual
# word senses.
SYNSET_RELATIONS: frozenset[str] = frozenset(
    {
        "hypernym",
        "instance_hypernym",
        "hyponym",
        "instance_hyponym",
        "member_meronym",
        "part_meronym",
        "substance_meronym",
        "member_holonym",
        "part_holonym",
        "substance_holonym",
        "similar_to",
        "also_see",
        "attribute",
        "entailment",
        "cause",
        "verb_group",
    }
)

# The only relation a closure query may transitively close over.
# instance_hypernym is deliberately excluded: an instance is not a subclass
# (Mount Everest is A mountain, not a KIND of mountain), so closing over it
# would let a traversal treat an instance as if it were a taxonomic node.
TRANSITIVE_RELATIONS: frozenset[str] = frozenset({"hypernym"})

# Meronymy/holonymy exists only over nouns in WordNet; a writer must never
# emit one of these for a verb/adjective/adverb synset.
NOUN_ONLY_RELATIONS: frozenset[str] = frozenset(
    {
        "member_meronym",
        "part_meronym",
        "substance_meronym",
        "member_holonym",
        "part_holonym",
        "substance_holonym",
    }
)


def synset_id(synset: object) -> str:
    """Release-prefixed id for a synset, built from its stable offset+pos."""
    return f"{ID_PREFIX}{synset.offset():08d}-{synset.pos()}"  # type: ignore[attr-defined]


def sense_id(lemma: object) -> str:
    """Release-prefixed id for a lemma, built from its WordNet sense key.

    The sense key (not an nltk-internal ordinal) is the cross-release bridge:
    it stays stable across corpus versions where synset offsets do not.
    """
    return f"{ID_PREFIX}{lemma.key()}"  # type: ignore[attr-defined]


def most_frequent_sense(word: str) -> dict | None:
    """The MFS baseline: WordNet's first-listed noun sense for *word*.

    WordNet orders a word's synsets by corpus frequency, so the first noun
    synset returned by ``synsets()`` already IS the most-frequent sense — no
    ranking is computed here. This is a deliberate BASELINE: supervised WSD
    and personalized PageRank re-ranking are stubbed upstream (issue #135)
    until the sense layer this ticket adds proves it is actually read.

    Returns ``None`` when the corpus is unprovisioned or *word* has no noun
    synset.
    """
    wn = _wordnet()
    if wn is None:
        return None
    lookup = word.lower().replace(" ", "_")
    synsets = wn.synsets(lookup, pos=wn.NOUN)
    if not synsets:
        return None
    synset = synsets[0]
    lemma = next(
        (lm for lm in synset.lemmas() if lm.name().lower() == lookup),
        synset.lemmas()[0],
    )
    return {
        "synset_id": synset_id(synset),
        "synset_name": synset.name(),
        "pos": synset.pos(),
        "definition": synset.definition(),
        "sense_id": sense_id(lemma),
        "sense_key": lemma.key(),
        "lemma": lemma.name().replace("_", " "),
    }


def ancestors(synset_name: str) -> tuple[list[dict], bool]:
    """The shortest hypernym path from *synset_name* to a root, as edge rows.

    Picks ``min(synset.hypernym_paths(), key=len)`` rather than every path:
    materialisation is lazy, so the point is to mint the fewest ancestor
    nodes that still connect a grounding to the taxonomy, not the whole
    hypernym lattice.

    Each row is ``{"child", "parent", "parent_name", "parent_definition",
    "rel", "transitive"}``. A parent that is one of ``synset.instance_hypernyms()``
    is recorded as ``rel="instance_hypernym"`` with ``transitive=False`` (see
    TRANSITIVE_RELATIONS); every other hop is ``rel="hypernym"`` with
    ``transitive=True``.

    Truncates at MAX_HYPERNYM_DEPTH edges and reports that via the bool
    return — silently truncating would hide a pathological or corrupted
    corpus from the caller. Returns ``([], False)`` when the corpus is
    unprovisioned or *synset_name* is unknown.
    """
    wn = _wordnet()
    if wn is None:
        return [], False
    from nltk.corpus.reader.wordnet import WordNetError  # type: ignore[import-untyped]

    try:
        synset = wn.synset(synset_name)
    except WordNetError:
        return [], False

    path = min(synset.hypernym_paths(), key=len)
    # hypernym_paths() includes the synset itself as the last element, root
    # first — the edge chain walks it back to front, synset -> root.
    chain = list(reversed(path[:-1]))

    rows: list[dict] = []
    child = synset
    truncated = False
    for parent in chain:
        if len(rows) >= MAX_HYPERNYM_DEPTH:
            truncated = True
            break
        is_instance = parent in child.instance_hypernyms()
        rows.append(
            {
                # Keyed by the same offset-based synset_id() write_senses()
                # uses for the SYNSET vertex it mints — a grounding writes
                # its synset under synset_id(), so this chain must key its
                # `child`/`parent` endpoints identically or write_synset_hypernyms'
                # MATCH on `child` silently finds nothing and the chain never
                # attaches to the grounded synset.
                "child": synset_id(child),
                "parent": synset_id(parent),
                "parent_name": parent.name(),
                "parent_definition": parent.definition(),
                "rel": "instance_hypernym" if is_instance else "hypernym",
                "transitive": not is_instance,
            }
        )
        child = parent
    return rows, truncated
