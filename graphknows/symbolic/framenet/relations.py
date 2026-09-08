"""The FrameNet frame-to-frame relation mapping — one table, one place.

FrameNet declares nine frame-to-frame relation types. **Exactly one of them,
``Inheritance``, is subsumption.** Persisting any of the others as a taxonomy
edge produces false generalisations, and this module exists so that judgement
lives in one table rather than being re-derived at each call site.

Why the other two candidates are not taxonomy:

``Using``
    The child *presupposes* the parent as background structure and binds only
    some of its frame elements. ``Shopping`` Uses ``Commerce_buy`` — and you can
    shop without buying anything, so shopping is not a kind of buying. Closing
    over it generalises a frame to background it merely refers to. 556 of the
    corpus's relations are Using, the single largest non-taxonomic type.

``Perspective_on``
    A neutral frame viewed from one side. ``Commerce_buy`` and ``Commerce_sell``
    both perspectivise ``Commerce_goods-transfer``. Treating a perspective as a
    parent makes a buy generalise through a transfer to whatever the transfer
    inherits, and silently discards the directionality that *is* the
    perspective — the buy/sell distinction is the whole content of the relation.

This is the same error class as treating ``skos:broader`` as transitive, which
this codebase gets right; :func:`broader_pairs` is the half that was wrong.

**Expansion is a different question from taxonomy.** Widening a query to
``Using`` and ``Perspective_on`` neighbours is legitimate and is what
``framenet.ontology_up`` does. Only the persisted ``BROADER`` hierarchy is
restricted here, because a stored edge is read as a claim about the world.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any

# The only FrameNet relation whose closure is licensed. Every Parent frame
# element binds to a corresponding Child one, which is what makes it is-a.
TAXONOMIC: frozenset[str] = frozenset({"Inheritance"})

# The remaining eight, persisted as SYM_REL and never closed over. Listed
# explicitly rather than derived as "everything else" so a new relation type in
# a future FrameNet release shows up as an unknown rather than being silently
# swept into the non-taxonomic bucket.
NON_TAXONOMIC: frozenset[str] = frozenset(
    {
        "Using",
        "Perspective_on",
        "Subframe",
        "Precedes",
        "Causative_of",
        "Inchoative_of",
        "See_also",
        "ReFraming_Mapping",
        # Not in the canonical nine-relation lists: FrameNet 1.7 ships 217
        # ReFraming_Mapping and 20 Metaphor relations. Metaphor relates a source
        # frame to a target one in metaphorical use and is emphatically not
        # subsumption. Found by unknown_relation_types() on its first run, which
        # is the whole reason that guard exists.
        "Metaphor",
    }
)

KNOWN: frozenset[str] = TAXONOMIC | NON_TAXONOMIC


def _relations(fn: Any) -> list[tuple[str, str, str]]:
    """``(sub, relation_type, super)`` for every frame relation in the corpus.

    ``sub``/``super`` follow FrameNet's own ``subFrameName``/``superFrameName``
    fields. Their meaning varies by relation type — for ``Inheritance`` the
    super is the parent, for ``Precedes`` it is the earlier frame — so callers
    must interpret them per type rather than assuming a hierarchy.
    """
    out: list[tuple[str, str, str]] = []
    for r in fn.frame_relations():
        general, specific = r.get("superFrameName"), r.get("subFrameName")
        if general and specific:
            out.append((str(specific), str(r.type.name), str(general)))
    return out


@lru_cache(maxsize=1)
def _split(fn: Any) -> tuple[dict[str, set[str]], tuple[tuple[str, str, str], ...], set[str]]:
    """Partition the corpus relations into (taxonomic map, non-taxonomic, unknown types)."""
    up: dict[str, set[str]] = defaultdict(set)
    other: list[tuple[str, str, str]] = []
    unknown: set[str] = set()
    for sub, rel, sup in _relations(fn):
        if rel in TAXONOMIC:
            up[sub].add(sup)
        elif rel in NON_TAXONOMIC:
            other.append((sub, rel, sup))
        else:
            unknown.add(rel)
    return dict(up), tuple(other), unknown


def broader_pairs(fn: Any) -> dict[str, set[str]]:
    """Frame -> strictly more general frames. ``Inheritance`` only.

    This is what gets persisted as ``BROADER`` and closed over at query time.
    """
    return _split(fn)[0]


def sym_rel_triples(fn: Any) -> tuple[tuple[str, str, str], ...]:
    """``(sub, relation_type, super)`` for every non-taxonomic frame relation.

    Persisted as ``SYM_REL {rel}`` — navigable, never closed over.
    """
    return _split(fn)[1]


def unknown_relation_types(fn: Any) -> set[str]:
    """Relation types the shipped corpus declares that this table does not name.

    Non-empty means FrameNet added a relation type: decide whether it is
    subsumption and add it to one of the two sets above. Reported at install
    rather than swallowed, because the silent default would be to drop it.
    """
    return _split(fn)[2]
