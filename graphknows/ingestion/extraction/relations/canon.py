"""WordNet verb canonicalization for open-vocabulary relation labels.

Maps mined relation triggers to a canonical verb lemma so lexical variants
collapse onto one edge type: nominalizations/gerunds go to their derivationally
related verb (adoption -> adopt), then the verb is canonicalized to the first
lemma of its first WordNet verb synset (purchase -> buy).

Conservative by design: when WordNet has no verb synset — or the canonical
lemma would be a multiword form the input is not (start -> get_down) — the
cleaned input lemma is returned unchanged. Never returns empty for valid input.
"""

from __future__ import annotations

import re
from functools import lru_cache

_CLEAN = re.compile(r"[^a-z0-9_ -]+")


def _stem_match(noun: str, verb: str) -> bool:
    """True when *verb* plausibly derives from *noun* (shared >=4-char prefix).

    Guards against WordNet's looser derivational links ('dog' -> 'chase')
    while keeping morphological ones (adoption -> adopt, residence -> reside).
    """
    need = min(len(noun), len(verb), 4)
    return noun[:need] == verb[:need]


def _derived_verb(noun: str) -> str:
    """First derivationally-related verb form of *noun* ('adoption' -> 'adopt')."""
    from nltk.corpus import wordnet as wn

    for synset in wn.synsets(noun, pos=wn.NOUN):
        for lemma in synset.lemmas():
            if lemma.name().casefold() != noun:
                continue
            for related in lemma.derivationally_related_forms():
                if related.synset().pos() == "v" and _stem_match(noun, related.name().casefold()):
                    return related.name().casefold()
    return ""


@lru_cache(maxsize=4096)
def canonical_verb(lemma: str, pos: str = "VERB") -> str:
    """Canonical verb lemma for a mined relation trigger.

    Args:
        lemma: spaCy lemma of the trigger token.
        pos: spaCy coarse POS of the trigger ("VERB" or "NOUN").

    Returns:
        Canonical lowercase lemma; the cleaned input when WordNet cannot
        improve on it. Empty only for empty/garbage input.
    """
    cleaned = _CLEAN.sub("", (lemma or "").strip().casefold()).strip()
    if not cleaned:
        return ""
    try:
        from nltk.corpus import wordnet as wn

        word = cleaned
        if pos != "VERB":
            word = _derived_verb(cleaned)
            if not word:
                # No morphological verb derivation: return the noun unchanged
                # rather than canonicalizing a homograph verb sense
                # (cabin/winter/dog are arguments, not predicates).
                return cleaned
        synsets = wn.synsets(word, pos=wn.VERB)
        if not synsets:
            return word
        canonical = synsets[0].lemmas()[0].name().casefold()
        # Reject multiword canonicals for single-word inputs (start -> get_down).
        if "_" in canonical and "_" not in word:
            return word
        return canonical
    except LookupError:  # WordNet corpus unavailable — pass through
        return cleaned
