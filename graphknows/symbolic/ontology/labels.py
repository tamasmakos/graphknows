"""Pick the entity and relation labels for one chunk, LEXICALLY.

The selection half of the SKOS scheme: exact lemma work over ``altLabels``,
POS-gated and IDF-filtered (see ``match.py``). Nothing is embedded — embedding
selection has failed four separate ways in this repo, while exact matching is the
only thing that has ever been right.

BUDGET IS A HARD CONSTRAINT, NOT A PREFERENCE
---------------------------------------------
Measured (see docs/ontology.md): relex's score is RELATIVE to the
offered set. The same correct triple scores 0.98 with 4 labels offered and 0.67
with 39. So a selector that returns everything it matched destroys the signal it
meant to feed. The budgets below are small on purpose, and the ranking that
fills them is the real work.

The coarse types are always included: they are the labels measured to work
(``'Gina'=person 1.00``, ``'Door Dash'=organization 0.99``), so they cost
nothing ontologically while guaranteeing the offer is never empty. No reject
label is added here — measured, ``other`` absorbed 10 of 16 spans in a chunk,
out-competing the real types it was meant to protect. Rejection belongs at the
score threshold instead.
"""

from __future__ import annotations

from graphknows.symbolic.ontology.lexical import evoked
from graphknows.symbolic.ontology.skos import Concept, ancestors

# The measured-good coarse types. Three of these are NOT CCO class labels
# verbatim (event, occupation, product) — they reach CCO through the overlay's
# curated `coarseAliases` block (conversational.json), read by
# `ingest._lexical_candidates`, not through their own bare spelling.
COARSE = ["person", "organization", "facility", "occupation", "event", "animal", "product"]


def entity_labels(
    doc: object,
    scheme: dict[str, Concept],
    index: dict[str, set[str]],
    budget: int = 14,
) -> list[str]:
    """The ENTITY type vocabulary to offer the extractor for this chunk."""
    scores = evoked(doc, scheme, index, "class")
    # Drop a concept whose own descendant is also evoked: the specific one wins,
    # and the general one stays reachable through `broader` at query time.
    redundant = {a for label in scores for a in ancestors(scheme, label)}
    ranked = sorted(
        (lbl for lbl in scores if lbl not in redundant and lbl.lower() not in COARSE),
        key=lambda lbl: -scores[lbl],
    )
    return [*COARSE, *ranked[: max(budget - len(COARSE), 0)]]


def relation_labels(
    doc: object,
    entity_types: set[str],
    scheme: dict[str, Concept],
    index: dict[str, set[str]],
    budget: int = 10,
) -> list[str]:
    """The RELATION vocabulary: lexically evoked properties, then domain/range fit.

    Two sources in priority order. A property whose own name is spoken
    ("married", "works", "teaches") is the strongest evidence available and goes
    first. The remaining budget goes to properties whose declared domain and
    range are satisfied by the types actually present — the structural mechanism
    that already works, now filling a much smaller gap.

    UPPER-ONTOLOGY PLUMBING IS EXCLUDED FROM THE STRUCTURAL HALF ONLY. A
    BFO-typed property (``continuant part of``, ``is predecessor of``) is
    satisfiable by literally anything, because the ancestor walk above climbs
    every entity into ``material entity``/``continuant``. It would therefore win
    budget slots on every chunk while asserting a relation between categories of
    being that no speaker uttered — the ``permits``/``requires`` failure that
    produced 64 of 286 REL edges on conv-30. The lexical half keeps them: text
    that actually says "requires" IS asserting the property.
    """
    scored = evoked(doc, scheme, index, "property")
    lexical = sorted(scored, key=lambda label: -scored[label])

    satisfied = {t.casefold() for t in entity_types}
    for etype in list(entity_types):
        satisfied.update(a.casefold() for a in ancestors(scheme, etype))

    # Keyed on the (domain, range) signature, which is what the round-robin
    # below balances over.
    by_sig: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = {}
    for concept in scheme.values():
        if (
            concept.kind != "property"
            or concept.pref_label in lexical
            or concept.upper_ontology
            or not (concept.domain and concept.range)
            or not ({d.casefold() for d in concept.domain} & satisfied)
            or not ({r.casefold() for r in concept.range} & satisfied)
        ):
            continue
        by_sig.setdefault((concept.domain, concept.range), []).append(concept.pref_label)

    # Round-robin over (domain, range) signatures so one large family — CCO has
    # 89 Person->Person properties — cannot take every remaining slot.
    interleaved: list[str] = []
    for rank in range(max((len(v) for v in by_sig.values()), default=0)):
        for members in by_sig.values():
            if rank < len(members):
                interleaved.append(members[rank])

    return [*lexical, *interleaved][:budget]
