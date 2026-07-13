"""Deterministic cleaning of GLiNER2 relation objects (llm_free, no LLM).

The schema-guided GLiNER2 relation model emits the conversational-taxonomy
predicates (attended / happened_on / owns / likes / dislikes / plans_to /
located_in) with clean subjects, but its *object* span is often a verb-headed
clause, a question echo, or a wrong-type role rather than a canonical content
noun. Left raw, those objects produce facts that outrank real gold in retrieval
(e.g. ``plans_to "Any plans coming up"``, ``owns husband``).

:func:`clean_relation_object` reduces each object to a canonical content noun
phrase using the spaCy dependency parse, and drops the non-content classes.
Rules (structural, domain-agnostic):

* **question echo** — object led by an interrogative / open quantifier
  ("Any plans coming up", "what movie") → drop.
* **verb-headed clause** — a multi-token span rooted on a verb ("expressing my
  trans experience", "make a difference") → trim to its head noun phrase, or
  drop if it has no nominal head.
* **non-content** — object with no noun / proper noun / gerund head → drop.
* **relationship role** — a bare kinship / social role ("husband", "best
  friend") is never a valid object of these predicates → drop.

Clean noun objects ("camping trip", "LGBTQ activist group") pass through
unchanged.
"""

from __future__ import annotations

import re
from typing import Any

# Interrogatives / open quantifiers that only ever lead a question echo, never a
# content object.
_INTERROGATIVE = frozenset(
    {"what", "which", "whose", "whom", "who", "how", "where", "when", "why", "any"}
)
_DET_RE = re.compile(r"^(a|an|the|my|your|his|her|their|our|some|that|this)\s+", re.I)


def _strip_leading_determiner(span: str) -> str:
    return _DET_RE.sub("", span.strip()).strip(" .,!?;:\"'")


def clean_relation_object(
    tail: str,
    nlp: Any,
    relationship_terms: frozenset[str] = frozenset(),
) -> str | None:
    """Return a canonical content-noun object, or ``None`` to drop the relation.

    Args:
        tail: The raw object span emitted by GLiNER2.
        nlp: A loaded spaCy pipeline (dependency parse + POS).
        relationship_terms: Lower-cased kinship / social-role words that must not
            stand as objects of these predicates (from the taxonomy).
    """
    raw = (tail or "").strip()
    if not raw:
        return None
    doc = nlp(raw)
    if len(doc) == 0:
        return None

    # Question echo — a leading interrogative / open quantifier.
    if doc[0].lower_ in _INTERROGATIVE:
        return None

    root = next((t for t in doc if t.head is t), doc[0])
    if root.pos_ in ("VERB", "AUX") and root.is_alpha and len(doc) >= 2:
        # Verb-headed clause: keep only its head noun phrase.
        chunks = list(doc.noun_chunks)
        if not chunks:
            return None
        span = chunks[-1]
    else:
        span = doc[:]

    # Re-check the (possibly trimmed) span's lead — a noun chunk can still open
    # with an open quantifier ("any more art").
    if span[0].lower_ in _INTERROGATIVE:
        return None

    def _is_content(s: Any) -> bool:
        # A real noun/proper-noun head, or a single gerund used as an activity
        # noun ("camping", "hiking").
        if any(t.pos_ in ("NOUN", "PROPN") for t in s):
            return True
        return len(s) == 1 and s[0].tag_ == "VBG"

    if not _is_content(span):
        return None

    text = _strip_leading_determiner(span.text)
    if len(text) < 3:
        return None
    if text.lower() in relationship_terms:
        return None
    return text
