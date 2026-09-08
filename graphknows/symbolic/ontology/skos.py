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
``alt_labels`` here carries only the label itself and its own content words —
no WordNet expansion. A content word has many senses and this module has no
mention to disambiguate one against at build time, only the bare label; baking
a dominant-sense synonym in here is exactly the context-free expansion issue
#141 removed (it minted "Gina permits fashion" from "let" meaning "inform").
Expansion is instead a MATCH-TIME decision, made where a mention — and
therefore a context — exists: see ``senses.sense_lemmas`` and
``lexical.evoked``'s WEAK branch. Precision over recall here: an altLabel that
fires on the wrong chunk is worse than one that never fires.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# English closed-class function words that survive the `len(w) > 2` filter in
# `content_tokens`. These are a fact about English grammar, not about any
# ontology's vocabulary — unlike the old `_STRUCTURAL` list, they cannot be
# derived from a digest's structure, so they stay a literal list. `of`/`a`/
# `an`/`to`/`in`/`by`/`at`/`on`/`or` are already 2 letters or shorter and need
# no entry here. `has` is a spaCy stopword `lexical.lemma_view` never emits,
# so keeping it as a content token would make any `has<Noun>`-shaped label
# (`hasOccupation`, and the same pattern in other vocabularies) permanently
# unreachable by `evoked`'s STRONG all-tokens-present rule, regardless of how
# frequent "has" is in the loaded vocabulary's own labels.
_FUNCTION_WORDS = frozenset({"the", "and", "with", "for", "from", "has"})

# A class within this many hops of a parentless root sits in the scaffolding
# of the hierarchy rather than naming a concrete kind.
_UPPER_CLASS_MAX_DEPTH = 3

# A class subsuming at least this fraction of the vocabulary's classes is
# scaffolding, not a concrete kind. On a small vocabulary (e.g. the 9-class
# personal profile) 5% is under one class, so every branch node would qualify
# — the absolute floor below is what keeps a small vocabulary from calling
# its every branch node upper-ontology. On CCO, 0.05 (72 classes) also pulls
# in `Act` (154 descendants) and `Information Content Entity` (153), stamping
# the overlay's `teaches`/`likes`/`plans to`/`attended` upper-ontology since
# they range over `Act`; 0.14 (~201 classes) sits between those two and CCO's
# next-largest depth<=3 class, `material entity` (546) — high enough to
# exclude Act/ICE, low enough to keep `material entity` (needed by the
# `has member part`/`member part of` regression) and `realizable entity`
# (215, needed to type `realizes`).
_UPPER_CLASS_MIN_DESCENDANT_FRACTION = 0.14
_UPPER_CLASS_MIN_DESCENDANTS = 8

# A token appearing in at least this fraction of labels — counted separately
# for classes and for properties, then unioned — is scaffolding rather than a
# discriminating word. Per-kind is load-bearing: "has" is 28% of CCO's
# property labels but only ~4% of all labels pooled. 0.09 (not 0.10) is
# deliberate: "act" is 9.5% of CCO's class labels ("Act of Employment",
# "Act of Travel", ...) — with `Act` itself no longer an upper class (see
# `_UPPER_CLASS_MIN_DESCENDANT_FRACTION`), this frequency-based route is the
# only thing that still keeps "act" out of `Act of X` altLabels. On a small
# vocabulary (e.g. the 9-class personal profile) even 9% is under one label, so any token
# seen even once would qualify — the absolute floor below is what keeps a
# small vocabulary's every class-label word from becoming "structural".
_STRUCTURAL_TOKEN_MIN_DOC_FREQ = 0.09
_STRUCTURAL_TOKEN_MIN_DOC_COUNT = 3


@dataclass(frozen=True)
class VocabularyProfile:
    """What the loaded vocabulary's own shape says is scaffolding.

    ``upper_classes`` are casefolded class labels that sit near the root of
    the subsumption DAG and subsume a large share of the vocabulary —
    CCO's ``entity``/``continuant``/``process`` and friends, the personal
    profile's ``Thing``, a schema.org-style digest's ``Thing`` alike.
    ``structural_tokens`` are the label words that come along with that:
    every token of an upper class's own label, plus whatever token is
    ubiquitous across class labels or across property labels.
    """

    upper_classes: frozenset[str]
    structural_tokens: frozenset[str]


def _class_depths(
    parents_of: dict[str, list[str]], children_of: dict[str, list[str]]
) -> dict[str, int]:
    """Hops from each class to the nearest parentless root, multi-source BFS down."""
    depth: dict[str, int] = {label: 0 for label, parents in parents_of.items() if not parents}
    frontier = list(depth)
    while frontier:
        nxt: list[str] = []
        for label in frontier:
            for child in children_of.get(label, ()):
                if child not in depth:
                    depth[child] = depth[label] + 1
                    nxt.append(child)
        frontier = nxt
    return depth


def _descendant_closure(
    label: str, children_of: dict[str, list[str]], memo: dict[str, set[str]], visiting: set[str]
) -> set[str]:
    """Every class transitively reachable from ``label`` via ``children_of``."""
    if label in memo:
        return memo[label]
    if label in visiting:
        return set()  # cycle guard
    visiting.add(label)
    out: set[str] = set()
    for child in children_of.get(label, ()):
        out.add(child)
        out |= _descendant_closure(child, children_of, memo, visiting)
    visiting.discard(label)
    memo[label] = out
    return out


def _class_depths_and_descendant_counts(
    classes: list[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Per-class ``(hops to the nearest root, size of the descendant closure)``."""
    parents_of = {str(c["label"]): [str(p) for p in (c.get("parents") or ())] for c in classes}
    children_of: dict[str, list[str]] = defaultdict(list)
    for label, parents in parents_of.items():
        for parent in parents:
            children_of[parent].append(label)

    depth = _class_depths(parents_of, children_of)
    memo: dict[str, set[str]] = {}
    descendant_count = {
        label: len(_descendant_closure(label, children_of, memo, set())) for label in parents_of
    }
    return depth, descendant_count


def _upper_class_labels(classes: list[Mapping[str, Any]]) -> frozenset[str]:
    if not classes:
        return frozenset()
    depth, descendant_count = _class_depths_and_descendant_counts(classes)
    floor = max(
        _UPPER_CLASS_MIN_DESCENDANTS,
        _UPPER_CLASS_MIN_DESCENDANT_FRACTION * len(classes),
    )
    return frozenset(
        str(c["label"]).casefold()
        for c in classes
        if depth.get(str(c["label"]), -1) <= _UPPER_CLASS_MAX_DEPTH
        and descendant_count.get(str(c["label"]), 0) >= floor
    )


def _frequent_tokens(labels: list[str], min_doc_freq: float) -> frozenset[str]:
    """Tokens whose document frequency across ``labels`` is at least ``min_doc_freq``.

    A raw fraction has no floor: on a 9-label vocabulary, 10% is under one
    label, so any token seen even once would qualify. The absolute count
    floor keeps a small vocabulary from calling its every label word
    scaffolding; on CCO's hundreds of labels it never binds.
    """
    token_sets = [set(content_tokens(label)) for label in labels]
    total = len(token_sets)
    if not total:
        return frozenset()
    floor = max(_STRUCTURAL_TOKEN_MIN_DOC_COUNT, min_doc_freq * total)
    counts: Counter[str] = Counter()
    for tokens in token_sets:
        counts.update(tokens)
    return frozenset(tok for tok, n in counts.items() if n >= floor)


def vocabulary_profile(digest: Mapping[str, Any]) -> VocabularyProfile:
    """Derive a :class:`VocabularyProfile` from a parsed digest's own structure.

    Reads only ``label`` and ``parents`` off ``classes``, and ``label`` off
    ``properties`` — no I/O, no spaCy, no WordNet. Works equally on the raw
    digest dict (``build_scheme``) and on a scheme reshaped the same way
    (``extend_scheme``, over the CCO+overlay merge).
    """
    classes = list(digest.get("classes") or ())
    properties = list(digest.get("properties") or ())
    upper_classes = _upper_class_labels(classes)
    upper_label_tokens: set[str] = set()
    for c in classes:
        if str(c["label"]).casefold() in upper_classes:
            upper_label_tokens.update(content_tokens(str(c["label"])))
    structural_tokens = (
        upper_label_tokens
        | _frequent_tokens([str(c["label"]) for c in classes], _STRUCTURAL_TOKEN_MIN_DOC_FREQ)
        | _frequent_tokens([str(p["label"]) for p in properties], _STRUCTURAL_TOKEN_MIN_DOC_FREQ)
    )
    return VocabularyProfile(
        upper_classes=upper_classes, structural_tokens=frozenset(structural_tokens)
    )


def _scheme_as_digest(scheme: dict[str, Concept]) -> dict[str, list[dict[str, Any]]]:
    """Reshape a merged scheme back into digest-dict form for :func:`vocabulary_profile`."""
    classes = [
        {"label": c.pref_label, "parents": list(c.broader)}
        for c in scheme.values()
        if c.kind == "class"
    ]
    properties = [{"label": c.pref_label} for c in scheme.values() if c.kind == "property"]
    return {"classes": classes, "properties": properties}


def _is_upper_ontology_property(concept: Concept, profile: VocabularyProfile) -> bool:
    """Whether this property relates vacuous, near-root categories rather than everyday things.

    A property whose domain or range is a class the ancestor walk in
    ``labels.relation_labels`` reaches from almost any entity (see
    :data:`_UPPER_CLASS_MAX_DEPTH`/:data:`_UPPER_CLASS_MIN_DESCENDANTS`) is
    satisfiable by nearly anything, so it wins a budget slot on every chunk
    while asserting a relation between categories of being that no speaker
    uttered. Measured on conv-30 (CCO): properties typed this way —
    ``permits``, ``requires``, ``realizes`` and their siblings — produced 64 of
    286 REL edges (22%), e.g. `Gina permits fashion` from the word "let",
    `Jon realizes dreams` from "realize" — none of which anyone asserted.

    The prefLabel and the label's own content words are KEPT: text that
    actually says "permits" is asserting the property, and should still match.

    The production caller (``labels.relation_labels``) never calls this
    directly — it reads the ``upper_ontology`` field ``lexicalize``/
    ``build_scheme``/``extend_scheme`` already stamped from a profile onto
    every ``Concept`` in the scheme. This function is the stamping logic
    itself, called at build time and by tests that supply their own profile.
    """
    if concept.kind != "property":
        return False
    return any(t.casefold() in profile.upper_classes for t in (*concept.domain, *concept.range))


@dataclass
class Concept:
    """One SKOS concept: a CCO class or object property with a lexical surface."""

    uri: str
    pref_label: str
    definition: str
    kind: str  # "class" | "property"
    alt_labels: set[str] = field(default_factory=set)
    broader: tuple[str, ...] = ()
    # The subset of ``broader`` that may NOT be composed — parents the digest
    # stamped as coming from ``skos:broader``/``skos:narrower`` rather than from
    # ``rdfs:subClassOf`` or ``skos:broaderTransitive``. Empty for every OWL
    # source, which is why an older digest reads correctly with no field at all.
    broader_nontransitive: frozenset[str] = frozenset()
    domain: tuple[str, ...] = ()
    range: tuple[str, ...] = ()
    module: str = ""
    # Stamped by `lexicalize`/`build_scheme`/`extend_scheme` from a
    # `VocabularyProfile`. `tokens` is the label's own content words (what
    # `lexical.evoked`'s STRONG match and `_sense_anchored_matches` gate on);
    # `upper_ontology` is `_is_upper_ontology_property`'s answer for this concept.
    # A `Concept` built directly (as the hand-built test doubles do) keeps the
    # defaults below.
    tokens: tuple[str, ...] = ()
    upper_ontology: bool = False


def content_tokens(label: str, structural: frozenset[str] = frozenset()) -> list[str]:
    """The descriptive words of a label, minus English grammar and ``structural``.

    Tokenizes BEFORE lowercasing: case is the only word boundary a camelCase
    label ("worksFor") has, and lowercasing first would collapse it into one
    unsplittable string.

    ``structural`` is the vocabulary-specific scaffolding — a
    ``VocabularyProfile.structural_tokens`` — layered on top of the fixed
    English function-word filter that always applies. Omitting it (the
    default) strips only the function words, which is what a caller with no
    scheme in hand — ``ingest._lexical_candidates`` tokenizing an extracted
    entity type — gets.
    """
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", label)
    return [
        w
        for w in (w.lower() for w in words)
        if w not in _FUNCTION_WORDS and w not in structural and len(w) > 2
    ]


def lexicalize(concept: Concept, profile: VocabularyProfile | None = None) -> Concept:
    """Fill ``alt_labels`` from the label itself and its own content words, and stamp it.

    Properties are also lemmatised ("requires" -> "require"): CCO names them as
    inflected present-tense verbs, but matching runs on spaCy lemmas throughout
    (``lexical.lemma_view``) and WordNet's own lemma names are always base
    forms, so an inflected-only altLabel is an index key neither the STRONG
    lexical match nor sense-anchored expansion (``lexical._sense_anchored_matches``)
    could ever reach. Classes are left alone: CCO's class labels are already
    base-form common nouns, and lemmatising 1400+ of them at build time for no
    reachability gain is not a cost worth paying.

    ``profile`` supplies the vocabulary-specific structural tokens and the
    upper-ontology class labels a whole altLabel is dropped for (the old
    ``_TOO_GENERIC``). Omitted, a concept lexicalizes against English grammar
    alone — the behaviour a hand-built test double gets.
    """
    structural = profile.structural_tokens if profile else frozenset()
    upper_classes = profile.upper_classes if profile else frozenset()
    label = concept.pref_label.lower()
    alts: set[str] = {label}
    tokens = content_tokens(concept.pref_label, structural)
    concept.tokens = tuple(tokens)
    if tokens:
        # The trailing content word is the semantic head for CCO's "X of Y"
        # naming ("Act of Employment" -> employment, "Commercial Organization"
        # -> organization); the leading ones stay as separate handles.
        alts.update(tokens)
        alts.add(" ".join(tokens))
    if concept.kind == "property":
        alts = _lemmatise(alts)
    concept.alt_labels = {a for a in alts if a not in upper_classes and len(a) > 2}
    concept.upper_ontology = _is_upper_ontology_property(concept, profile) if profile else False
    return concept


def build_scheme(digest: Path) -> dict[str, Concept]:
    """The whole CCO digest as a lexicalized SKOS scheme, keyed by prefLabel.

    Two passes: every ``Concept`` is constructed first, then the vocabulary
    profile is derived from the digest's own structure, then each concept is
    lexicalized and stamped against it — the profile needs the whole digest
    (class parents, all labels) before any one concept can be lexicalized.
    """
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
                broader_nontransitive=frozenset(entry.get("parents_nontransitive") or ()),
                domain=tuple(entry.get("domain") or ()),
                range=tuple(entry.get("range") or ()),
                module=str(entry.get("module") or ""),
            )
            scheme[concept.pref_label] = concept
    profile = vocabulary_profile(data)
    for concept in scheme.values():
        lexicalize(concept, profile)
    return scheme


def _lemmatise(labels: set[str]) -> set[str]:
    """Both the surface and its lemma, so either spelling matches.

    Keeps the original too: a multi-word altLabel ("used to work") is matched
    against noun compounds where lemmatising the whole phrase would not help.
    """
    from graphknows.nlp import load_spacy_model
    from graphknows.settings import get_settings

    nlp = load_spacy_model(get_settings().spacy_model)
    out = set(labels)
    for label in labels:
        out.add(" ".join(t.lemma_.lower() for t in nlp(label)))
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
                broader_nontransitive=frozenset(entry.get("parents_nontransitive") or ()),
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
    # Re-derive the profile over the MERGED scheme, not just the CCO base: an
    # overlay class can change a CCO class's descendant count, and an overlay
    # concept otherwise never gets `tokens`/`upper_ontology` stamped at all
    # (its alt_labels are authored, so `lexicalize` is not called for it).
    profile = vocabulary_profile(_scheme_as_digest(scheme))
    for concept in scheme.values():
        concept.tokens = tuple(content_tokens(concept.pref_label, profile.structural_tokens))
        concept.upper_ontology = _is_upper_ontology_property(concept, profile)
    return scheme


def coarse_aliases(overlay: Path) -> dict[str, str]:
    """The overlay's curated ``coarseAliases`` block: coarse type -> CCO class.

    The extractor's guaranteed coarse vocabulary (``labels.COARSE``) is offered
    on every chunk, and a human already curated what each one MEANS in CCO terms
    (``event`` -> ``Act``, ``product`` -> ``Artifact``) — this is that mapping,
    read verbatim so it can outrank the accidental altLabel-fragment anchor. An
    empty string is a deliberate answer ("this coarse type is untypeable"), not
    a missing one, so it is kept rather than filtered out.
    """
    data = json.loads(overlay.read_text(encoding="utf-8"))
    block = data.get("coarseAliases") or {}
    return {k.casefold(): v for k, v in block.items() if k != "_comment"}


def lexical_index(scheme: dict[str, Concept]) -> dict[str, set[str]]:
    """AltLabel -> the prefLabels that claim it. The whole selection mechanism."""
    index: dict[str, set[str]] = defaultdict(set)
    for concept in scheme.values():
        for alt in concept.alt_labels:
            index[alt].add(concept.pref_label)
    return index


# scheme id -> (the scheme it was built from, casefolded label -> actual key).
# Keyed by identity because the scheme is a process-wide singleton
# (``scheme.load_scheme`` builds and caches it once, never in place afterwards)
# and ``ancestors`` is called with it on nearly every chunk. The stashed scheme
# reference keeps that id from being recycled onto an unrelated dict; it is a
# handful of long-lived schemes in production and, at worst, one entry per
# scheme a test builds.
_casefold_index_cache: dict[int, tuple[dict[str, Concept], dict[str, str]]] = {}


def _casefold_index(scheme: dict[str, Concept]) -> dict[str, str]:
    """Casefolded label -> the scheme's real key, built once per scheme and cached.

    ``ancestors`` used to fall back to an O(n) walk over the whole scheme (~1,660
    CCO concepts) on every case-mismatched lookup -- which was nearly every
    lookup, since the extractor's coarse types are lowercase and the scheme is
    keyed by capitalised prefLabels.
    """
    cached = _casefold_index_cache.get(id(scheme))
    if cached is not None and cached[0] is scheme:
        return cached[1]
    index = {key.casefold(): key for key in scheme}
    _casefold_index_cache[id(scheme)] = (scheme, index)
    return index


def ancestors(
    scheme: dict[str, Concept], label: str, depth: int = 12, *, transitive_only: bool = True
) -> list[str]:
    """The ``broader`` closure — what multi-hop generalisation walks.

    ``skos:broader`` IS NOT TRANSITIVE. SKOS keeps ``skos:broaderTransitive`` as
    its separate transitive super-property for exactly this reason: "Java broader
    Island" and "Island broader Landform" do not license "Java broader Landform".
    A parent reached over a non-transitive link is therefore reported — it is a
    parent — but never expanded, which makes a three-level ``skos:broader`` chain
    yield one ancestor rather than two. Pass ``transitive_only=False`` to get the
    old loose walk, knowing it concludes things the scheme does not assert.

    This walk was unconditionally transitive until now, and safe only by
    accident: the bundled CCO digest derives every ``broader`` from
    ``rdfs:subClassOf``, which IS transitive. It became wrong the moment a real
    SKOS scheme was imported, which is what ``broader_nontransitive`` fixes.

    Breadth-first over EVERY declared parent, not just the first. CCO uses
    multiple inheritance freely (``Dance Studio`` is both a ``Facility`` and a
    ``Commercial Organization``), and this closure gates two live decisions:
    which evoked classes ``entity_labels`` prunes as redundant, and which
    properties ``relation_labels`` considers domain/range satisfiable. Following
    a single parent silently drops whole branches from both, so a property
    declared over the skipped branch is never offered to the extractor.

    The entry label is resolved case-insensitively. The scheme is keyed by
    prefLabel (``Person``) while callers hand over extracted entity types
    (``person``, the coarse vocabulary ``entity_labels`` emits), so an exact
    lookup missed every coarse type and the widening this function exists for
    contributed nothing at all. Only the entry needs it — everything reached
    from ``broader`` is already a prefLabel.

    ``depth`` bounds the number of LEVELS walked; ``seen`` makes a cyclic
    hierarchy terminate.
    """
    if label not in scheme:
        label = _casefold_index(scheme).get(label.casefold(), label)
    out: list[str] = []
    seen: set[str] = {label}
    frontier = [label]
    while frontier and depth > 0:
        nxt: list[str] = []
        for current in frontier:
            concept = scheme.get(current)
            if concept is None:
                continue
            for parent in concept.broader:
                if parent in seen:
                    continue
                seen.add(parent)
                out.append(parent)
                if transitive_only and parent in concept.broader_nontransitive:
                    continue  # a parent, but not a step the closure may take
                nxt.append(parent)
        frontier = nxt
        depth -= 1
    return out
