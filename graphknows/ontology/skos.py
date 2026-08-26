"""CCO as a LEXICALIZED SKOS concept scheme, for per-chunk label selection.

THE PROBLEM THIS SOLVES
-----------------------
CCO has semantics and no lexical surface. It ships zero ``skos:altLabel``, so the
only way anyone has tried to reach a class from text is embedding similarity —
which is measurably noise here (11% of chunk tags share a word with their chunk;
``Gina`` -> ``Iris``; "Lost my job as a banker" -> ``Grocery Store``). Meanwhile
exact label match is the ONE mapping that has ever been correct in this repo, and
FrameNet — which is nothing but lexical surface — triggers correctly on 100% of
the same turns.

So the missing piece is not a better matcher. It is the surface itself. This
module builds it: every CCO class and object property becomes a
:class:`Concept` carrying a ``prefLabel``, a set of ``altLabels`` harvested from
the label's own content words and their WordNet synonyms, its ``broader`` chain,
and (for properties) domain/range. Selection then becomes LEXICAL LOOKUP —
exact, cheap, explainable, and with no embedding in the loop.

WHY SKOS AND NOT JUST A DICT
----------------------------
``broader``/``narrower`` is what multi-hop reasoning traverses. A question about
"the hospital" should reach a chunk typed ``Medical Facility`` and one typed
``Facility``; that is a hierarchy walk, not a similarity score. Keeping the
scheme SKOS-shaped means the same structure that selects labels at ingest is the
structure that generalises at query time.

WHAT IS DELIBERATELY NOT DONE
-----------------------------
Synonyms are taken from the FIRST WordNet synset of a content word only. Deeper
expansion adds polysemy fast ("second" -> every temporal and ordinal sense), and
this repo has already been burned once by a lexicalized set ranked on sense count
turning into ``Strength``/``Pitch``/``Power``. Precision over recall here: an
altLabel that fires on the wrong chunk is worse than one that never fires.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Tokens that are structural in CCO labels rather than descriptive. "Act of
# Employment" and "Act of Travel" share "act"; letting it become an altLabel
# makes every Act class fire on the word "act" and on nothing useful.
_STRUCTURAL = frozenset(
    [
        "act",
        "acts",
        "of",
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "in",
        "by",
        "with",
        "for",
        "from",
        "at",
        "on",
        "entity",
        "entities",
        "role",
        "roles",
        "process",
        "processes",
        "profile",
        "profiles",
        "quality",
        "qualities",
        "artifact",
        "artifacts",
        "information",
        "content",
        "ontology",
        "generically",
        "dependent",
        "continuant",
        "occurrent",
    ]
)

# Labels too generic to be worth matching on even as a whole.
_TOO_GENERIC = frozenset({"entity", "continuant", "occurrent", "object", "quality", "process"})


@dataclass
class Concept:
    """One SKOS concept: a CCO class or object property with a lexical surface."""

    uri: str
    pref_label: str
    definition: str
    kind: str  # "class" | "property"
    alt_labels: set[str] = field(default_factory=set)
    broader: tuple[str, ...] = ()
    domain: tuple[str, ...] = ()
    range: tuple[str, ...] = ()
    module: str = ""


def content_tokens(label: str) -> list[str]:
    """The descriptive words of a label, minus CCO's structural scaffolding."""
    words = re.findall(r"[A-Za-z]+", label.lower())
    return [w for w in words if w not in _STRUCTURAL and len(w) > 2]


def _wordnet_synonyms(word: str, kind: str) -> set[str]:
    """Lemmas of the word's FIRST synset — its dominant sense only."""
    try:
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]  # nltk ships no stubs
    except Exception:
        return set()
    pos = wn.NOUN if kind == "class" else wn.VERB
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        return set()
    return {lemma.name().replace("_", " ").lower() for lemma in synsets[0].lemmas()}


def lexicalize(concept: Concept, synonyms: bool = True) -> Concept:
    """Fill ``alt_labels`` from the label's own words and their dominant synonyms."""
    label = concept.pref_label.lower()
    alts: set[str] = {label}
    tokens = content_tokens(concept.pref_label)
    if tokens:
        # The trailing content word is the semantic head for CCO's "X of Y"
        # naming ("Act of Employment" -> employment, "Commercial Organization"
        # -> organization); the leading ones stay as separate handles.
        alts.update(tokens)
        alts.add(" ".join(tokens))
        if synonyms:
            for token in tokens:
                alts |= _wordnet_synonyms(token, concept.kind)
    concept.alt_labels = {a for a in alts if a not in _TOO_GENERIC and len(a) > 2}
    return concept


def build_scheme(digest: Path, synonyms: bool = True) -> dict[str, Concept]:
    """The whole CCO digest as a lexicalized SKOS scheme, keyed by prefLabel."""
    data = json.loads(digest.read_text(encoding="utf-8"))
    scheme: dict[str, Concept] = {}
    for kind, key in (("class", "classes"), ("property", "properties")):
        for entry in data.get(key) or []:
            concept = Concept(
                uri=str(entry["uri"]),
                pref_label=str(entry["label"]),
                definition=str(entry.get("definition") or ""),
                kind=kind,
                broader=tuple(entry.get("parents") or ()),
                domain=tuple(entry.get("domain") or ()),
                range=tuple(entry.get("range") or ()),
                module=str(entry.get("module") or ""),
            )
            scheme[concept.pref_label] = lexicalize(concept, synonyms)
    return scheme


_NLP = None


def _lemmatise(labels: set[str]) -> set[str]:
    """Both the surface and its lemma, so either spelling matches.

    Keeps the original too: a multi-word altLabel ("used to work") is matched
    against noun compounds where lemmatising the whole phrase would not help.
    """
    global _NLP
    if _NLP is None:
        import spacy

        from graphknows.settings import GraphKnowsSettings

        _NLP = spacy.load(GraphKnowsSettings().spacy_model)
    out = set(labels)
    for label in labels:
        out.add(" ".join(t.lemma_.lower() for t in _NLP(label)))
    return out


def extend_scheme(scheme: dict[str, Concept], overlay: Path) -> dict[str, Concept]:
    """Merge an INPUT SKOS scheme over the CCO base.

    This is the seam that makes the whole design ontology-DRIVEN rather than
    hardcoded: the overlay is data, supplying the everyday concepts CCO does not
    carry, with explicit ``altLabels`` (the lexical surface) and ``broader``
    links into CCO's own hierarchy so multi-hop generalisation still works.
    Authored altLabels are taken verbatim and NOT WordNet-expanded — whoever
    wrote them knows the domain better than a dominant-sense lookup does.
    """
    data = json.loads(overlay.read_text(encoding="utf-8"))
    for kind, key in (("class", "classes"), ("property", "properties")):
        for entry in data.get(key) or []:
            concept = Concept(
                uri=str(entry["uri"]),
                pref_label=str(entry["label"]),
                definition=str(entry.get("definition") or ""),
                kind=kind,
                broader=tuple(entry.get("parents") or ()),
                domain=tuple(entry.get("domain") or ()),
                range=tuple(entry.get("range") or ()),
                module="overlay",
            )
            authored = {a.lower() for a in entry.get("altLabels") or ()}
            # Authored altLabels MUST be lemmatised, because matching happens on
            # spaCy lemmas. Written as "lost"/"bought"/"went" they can never
            # fire: the text lemmatises to lose/buy/go and the surface form is
            # not what is compared. This cost a real concept — "formerly worked
            # for" was authored with altLabel "lost", matched nothing, and looked
            # like a model failure rather than a normalisation bug.
            concept.alt_labels = _lemmatise(authored) | {concept.pref_label.lower()}
            scheme[concept.pref_label] = concept
    return scheme


def lexical_index(scheme: dict[str, Concept]) -> dict[str, set[str]]:
    """AltLabel -> the prefLabels that claim it. The whole selection mechanism."""
    index: dict[str, set[str]] = defaultdict(set)
    for concept in scheme.values():
        for alt in concept.alt_labels:
            index[alt].add(concept.pref_label)
    return index


def ancestors(scheme: dict[str, Concept], label: str, depth: int = 12) -> list[str]:
    """Every transitive ``broader`` ancestor, nearest level first.

    Not one chain: a concept can have two parents — a Store is both a Facility
    and an Organization — and walking only the first silently drops the other
    branch from ``relation_labels``' satisfied domain/range set and from
    ``entity_labels``' redundancy set. ``depth`` limits the levels walked.
    """
    out: list[str] = []
    seen = {label}
    frontier = [label]
    for _ in range(depth):
        parents = list(
            dict.fromkeys(
                parent
                for child in frontier
                if child in scheme
                for parent in scheme[child].broader
                if parent not in seen
            )
        )
        if not parents:
            break
        seen.update(parents)
        out.extend(parents)
        frontier = parents
    return out
