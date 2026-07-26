"""Subject-centric SVO relation extraction with speaker resolution.

The dominant dialogue fact pattern is a first-person clause ("I lost my job",
"I work as a nurse") whose subject is the *speaker*, not a named entity in the
text. Entity-pair mining misses these because it needs two named arguments with
a dependency path between them. This module iterates clause verbs instead: the
subject is ``nsubj``/``nsubjpass`` (``conj``/``xcomp`` verbs inherit the
governing subject), first/second-person pronouns resolve to the utterance
speaker, and dangling anaphora ("it", "they", "that") are dropped because a fact
cannot be anchored to them.

The predicate is the raw clause verb lemma plus any phrasal particle and
preposition (WORK_AS, LOSE_AT, REHEARSE_WITH), lowercased and joined by "_".
No WordNet synonym canonicalization (it distorts: launch -> establish), no
FrameNet frame names, no predefined edge list — the label is what the text says.
Phatic/mental verbs (WordNet verb.communication/cognition/perception) and
date/number objects (temporal metadata) are dropped so the graph holds facts,
not conversational meta-acts.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

_FIRST_PERSON = {"i", "me", "my", "myself", "we", "us", "our"}
_SECOND_PERSON = {"you", "your", "yourself"}
# WordNet verb supersenses that carry conversational/mental content, not facts.
_PHATIC_LEXNAMES = {"verb.communication", "verb.cognition", "verb.perception"}
# WordNet noun supersenses that denote CONCRETE, referable things worth a HAS
# edge; abstractions (feeling/cognition/attribute/state/act/communication) are
# dropped so "HAS passion / memory / pick / go-to" don't pollute the graph.
_CONCRETE_NOUN = {
    "noun.person",
    "noun.animal",
    "noun.artifact",
    "noun.group",
    "noun.possession",
    "noun.location",
    "noun.body",
    "noun.food",
    "noun.plant",
    "noun.object",
}
_LIGHT = {"be", "have", "do", "get", "go", "come", "let", "seem", "become"}
_SUBJ_DEPS = {"nsubj", "nsubjpass", "expl"}
_OBJ_DEPS = {"dobj", "obj", "dative", "attr", "oprd"}
# Objects of these NER types are metadata (time/number), not relation tails —
# time is carried by the TEMPORAL / MENTIONS_DATE layer.
_META_ENT = {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"}

_SPEAKER_LINE = re.compile(r"^(?:.*?\|\s*)?([A-Za-z][\w' ]*?):\s*(.+)$")


@lru_cache(maxsize=8192)
def _phatic(verb_lemma: str) -> bool:
    """True when the verb's first WordNet sense is a communication/mental act."""
    try:
        from nltk.corpus import wordnet as wn

        ss = wn.synsets(verb_lemma, pos=wn.VERB)
        return bool(ss) and ss[0].lexname() in _PHATIC_LEXNAMES
    except LookupError:
        return False


def _resolve(tok: Any, speaker: str, speakers: list[str]) -> str:
    """Resolve a 1st/2nd-person pronoun to the speaker (or the other of two)."""
    low = tok.text.lower()
    if low in _FIRST_PERSON:
        return speaker
    if low in _SECOND_PERSON and len(speakers) == 2 and speaker:
        return next((s for s in speakers if s != speaker), tok.text)
    return tok.text


def _clean_np(tok: Any, speaker: str, speakers: list[str]) -> str:
    """Head-noun phrase for an argument token.

    Strips det/poss, prefers a named apposition ('a cat named Luna' -> 'Luna'),
    keeps compounds ('dance studio').
    """
    if tok.text.lower() in _FIRST_PERSON | _SECOND_PERSON:
        return _resolve(tok, speaker, speakers)
    for c in tok.children:
        if c.dep_ == "appos" and c.pos_ == "PROPN":
            return c.text
    comp = [c.text for c in tok.children if c.dep_ == "compound"]
    return " ".join([*comp, tok.text]).strip()


def _named(tok: Any) -> bool:
    """True when the argument token is a proper noun / named entity or names one."""
    return (
        tok.pos_ == "PROPN"
        or bool(tok.ent_type_)
        or any(c.dep_ == "appos" and c.pos_ == "PROPN" for c in tok.children)
    )


def _subject_tok(verb: Any) -> Any:
    """The clause subject of *verb*; conj/xcomp/advcl/ccomp inherit the head's."""
    for c in verb.children:
        if c.dep_ in _SUBJ_DEPS:
            return c
    if verb.dep_ in ("conj", "xcomp", "advcl", "ccomp") and verb.head is not verb:
        return _subject_tok(verb.head)
    return None


def _entity_subject(tok: Any, speaker: str, speakers: list[str]) -> tuple[str, bool] | None:
    """Resolve a subject token to (name, is_named), or None if dangling anaphora.

    A knowledge-triplet subject must be nameable: the resolved speaker (1st/2nd
    person), a proper noun, a named entity, or a concrete common noun. Unresolved
    pronouns and demonstratives ("it", "they", "that") are dropped — no anchor.
    ``is_named`` marks a speaker name or spaCy-typed entity (used to gate an
    un-corroborated SVO triple as high-confidence structural).
    """
    if tok is None:
        return None
    low = tok.text.lower()
    if low in _FIRST_PERSON or (low in _SECOND_PERSON and len(speakers) == 2 and speaker):
        return _resolve(tok, speaker, speakers), True
    if tok.pos_ == "PRON":
        return None  # dangling anaphora — can't anchor a fact
    if tok.pos_ in ("PROPN", "NOUN"):
        return _clean_np(tok, speaker, speakers), _named(tok)
    return None


def _has_meta(tok: Any) -> bool:
    """True if the argument's phrase is a date/number span (temporal metadata)."""
    return any(t.ent_type_ in _META_ENT for t in tok.subtree)


def _predicate(verb: Any, prep: str = "") -> str:
    """Open-vocabulary label: raw verb lemma + phrasal particle + preposition."""
    base = (verb.lemma_ or verb.text).lower().strip()
    prt = next((c.text.lower() for c in verb.children if c.dep_ == "prt"), "")
    parts = [p for p in (base, prt, prep) if p]
    return "_".join(parts)


def extract_svo(sent: Any, speaker: str, speakers: list[str]) -> list[dict[str, Any]]:
    """Subject-centric SVO facts from one parsed sentence.

    Returns dicts ``{head, relation, tail, verb, prep, head_named, tail_named}``
    where ``relation`` is the UPPERCASE open-vocab predicate and the ``*_named``
    flags say whether each argument is a speaker name / named entity.
    """
    out: list[dict[str, Any]] = []
    for verb in sent:
        if verb.pos_ not in ("VERB", "AUX"):
            continue
        lemma = (verb.lemma_ or verb.text).lower()
        subj = _entity_subject(_subject_tok(verb), speaker, speakers)
        if not subj or len(subj[0]) < 2:
            continue
        subj_name, subj_named = subj
        args: list[tuple[Any, str]] = []
        for c in verb.children:
            if c.dep_ in _OBJ_DEPS:
                args.append((c, ""))
            elif c.dep_ == "prep":
                for p in c.children:
                    if p.dep_ == "pobj":
                        args.append((p, c.text.lower()))
        for obj_tok, prep in args:
            if (
                obj_tok.pos_ == "PRON"
                and obj_tok.text.lower() not in _FIRST_PERSON | _SECOND_PERSON
            ):
                continue
            if _has_meta(obj_tok):
                continue  # dates/numbers are temporal metadata, not relation tails
            obj = _clean_np(obj_tok, speaker, speakers)
            if not obj or obj.lower() == subj_name.lower() or len(obj) < 2:
                continue
            # A light verb with no prep carries no content ("I have it").
            if lemma in _LIGHT and not prep:
                continue
            if _phatic(lemma):
                continue
            pred = _predicate(verb, prep)
            if not pred:
                continue
            out.append(
                {
                    "head": subj_name,
                    "relation": pred.upper(),
                    "tail": obj,
                    "verb": lemma,
                    "prep": prep,
                    "head_named": subj_named,
                    "tail_named": _named(obj_tok),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Rich dependency-grammar relations (beyond verb-SVO)
#
# The clause-verb pass above misses durable identity/possession facts that a
# dialogue states copularly ("I'm a nurse"), possessively ("Caroline's dog"),
# or appositively ("Melanie, a nurse"). These four patterns recover them, reuse
# the SVO helpers/speaker-resolution, and tag each triple with a ``pattern`` so
# the extractor can gate them (copular/appositive/possessive with a named head
# are high-precision structural facts kept without GLiNER2 corroboration).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8192)
def noun_supersense(phrase: str) -> str:
    """WordNet supersense of a noun phrase's head ("noun.artifact"), or "" if OOV.

    The lexicographer file of the head noun's first sense — the same signal
    :func:`_known_concrete_noun` already reduces to a bool.
    spaCy NER types only NAMED entities, but dialogue relation arguments
    are overwhelmingly common nouns ("a nurse", "cat", "job"), so without this
    they reach the graph as untyped ENTITY vertices. Head = rightmost token,
    matching the head-noun convention _clean_np produces.
    """
    head = phrase.strip().split()[-1].lower() if phrase.strip() else ""
    if not head:
        return ""
    try:
        from nltk.corpus import wordnet as wn
        from nltk.stem import WordNetLemmatizer

        ss = wn.synsets(WordNetLemmatizer().lemmatize(head, "n"), pos=wn.NOUN)
    except LookupError:
        return ""
    return ss[0].lexname() if ss else ""


@lru_cache(maxsize=8192)
def _known_concrete_noun(lemma: str) -> bool:
    """True if *lemma* has a WordNet noun sense whose first supersense is concrete.

    Out-of-vocabulary words are REJECTED. An unnamed IS_A descriptor or possessive
    tail must be a KNOWN dictionary role/thing noun, so the out-of-vocabulary
    capitalized interjections that flood turn-structured dialogue ("Thanks",
    "Woah", "Yep", "Congrats" — spaCy tags them NOUN/PROPN) and conversational
    slang ("fam", "fave", "edu") are dropped rather than admitted as facts. Named
    arguments bypass this via ``_named(tok)``, which is where the proper-name
    exemption belongs.
    """
    try:
        from nltk.corpus import wordnet as wn

        ss = wn.synsets(lemma, pos=wn.NOUN)
    except LookupError:
        return False
    return bool(ss) and ss[0].lexname() in _CONCRETE_NOUN


def _valid_is_a_tail(tok: Any) -> bool:
    """True if *tok* is a usable IS_A descriptor: a KNOWN concrete COMMON noun.

    An identity/role descriptor is by definition a concrete common noun
    (person/role/artifact/group): "nurse", "mentor", "transgender woman", "kid",
    "dance studio". A named descriptor ("Jon IS_A Gina", "Jon IS_A Thanks") is
    dropped (identity is not another proper noun), and an abstract or OOV common
    noun ("passion", "dreams", "pick", "Woah") is dropped by the WordNet gate —
    this is what keeps the copular/appositive IS_A layer precise on dialogue.
    """
    return not _named(tok) and _known_concrete_noun((tok.lemma_ or tok.text).lower())


def _copular(sent: Any, speaker: str, speakers: list[str]) -> list[dict[str, Any]]:
    """X is a nurse / X is a transgender woman -> IS_A (predicate NOMINAL only).

    Predicate adjectives (acomp: 'is excited/hectic') are dropped — transient
    states / opinions, not durable facts. Identity phrases survive as nominals
    ('transgender woman' is a NOUN head carrying an amod); abstract or named
    predicates are dropped by ``_valid_is_a_tail``.
    """
    out: list[dict[str, Any]] = []
    for verb in sent:
        if (verb.lemma_ or "").lower() != "be":
            continue
        subj = _entity_subject(_subject_tok(verb), speaker, speakers)
        if not subj or len(subj[0]) < 2:
            continue
        subj_name, subj_named = subj
        for c in verb.children:
            if c.dep_ not in ("attr", "oprd") or c.pos_ not in ("NOUN", "PROPN"):
                continue  # nominal predicate only, no adjectives/pronouns
            if _has_meta(c) or not _valid_is_a_tail(c):
                continue
            tail = _clean_np(c, speaker, speakers)
            if not tail or tail.lower() == subj_name.lower() or len(tail) < 2:
                continue
            out.append(
                {
                    "head": subj_name,
                    "relation": "IS_A",
                    "tail": tail,
                    "pattern": "copular",
                    "head_named": subj_named,
                    "tail_named": False,
                }
            )
    return out


def _possessive(sent: Any, speaker: str, speakers: list[str]) -> list[dict[str, Any]]:
    """Caroline's dog / my grandmother -> (possessor, HAS, thing).

    The possessor must be a person (proper noun / named entity) or a resolvable
    1st/2nd-person pronoun; generic possessors ('its/their') are skipped. Abstract
    possessed nouns are dropped via WordNet supersense; named things always pass.
    """
    out: list[dict[str, Any]] = []
    for tok in sent:
        if tok.pos_ not in ("NOUN", "PROPN"):
            continue
        for c in tok.children:
            if c.dep_ != "poss":
                continue
            low = c.text.lower()
            if low in _FIRST_PERSON or (low in _SECOND_PERSON and len(speakers) == 2 and speaker):
                possessor, pnamed = _resolve(c, speaker, speakers), True
            elif c.pos_ == "PROPN" or c.ent_type_:
                possessor, pnamed = _clean_np(c, speaker, speakers), True
            else:
                continue  # generic possessor (its/their) -> skip
            if not possessor or len(possessor) < 2:
                continue  # unresolved 1st/2nd person (no speaker) -> no anchor
            thing = _clean_np(tok, speaker, speakers)
            if not thing or thing.lower() == possessor.lower() or _has_meta(tok):
                continue
            # Strict (KNOWN-concrete) gate, not the lenient _is_concrete: the
            # lenient one passes out-of-vocabulary words so proper names survive,
            # but conversational slang is OOV too, so "HAS fam / fave / edu"
            # sailed through as possession facts. A tail that is not a named
            # entity must be a known dictionary thing-noun. Named tails still
            # pass via _named(tok) — the exemption is preserved where it belongs.
            if not _named(tok) and not _known_concrete_noun((tok.lemma_ or tok.text).lower()):
                continue
            out.append(
                {
                    "head": possessor,
                    "relation": "HAS",
                    "tail": thing,
                    "pattern": "possessive",
                    "head_named": pnamed,
                    "tail_named": _named(tok),
                }
            )
    return out


def _appositive(sent: Any, speaker: str, speakers: list[str]) -> list[dict[str, Any]]:
    """Melanie, a nurse / my brother David -> (named, IS_A, descriptor).

    Oriented so the named/proper side is the head; at least one side must be named.
    """
    out: list[dict[str, Any]] = []
    for tok in sent:
        for c in tok.children:
            if c.dep_ != "appos":
                continue
            a, b = tok, c
            a_named, b_named = _named(a), _named(b)
            if b_named and not a_named:
                a, b = b, a
                a_named, b_named = b_named, a_named
            if not a_named or not _valid_is_a_tail(b):
                continue  # need a named anchor and a concrete common-noun descriptor
            head = _clean_np(a, speaker, speakers)
            tail = _clean_np(b, speaker, speakers)
            if not head or not tail or head.lower() == tail.lower():
                continue
            out.append(
                {
                    "head": head,
                    "relation": "IS_A",
                    "tail": tail,
                    "pattern": "appositive",
                    "head_named": a_named,
                    "tail_named": False,
                }
            )
    return out


def _noun_prep(sent: Any, speaker: str, speakers: list[str]) -> list[dict[str, Any]]:
    """Necklace from grandmother / trip to Sweden -> (named noun, PREP, pobj).

    Conservative (low volume): the head noun must be a named entity, and the
    pobj must not be a date/number or a pronoun. The relation label is the
    uppercased preposition (FROM/TO/WITH...).
    """
    out: list[dict[str, Any]] = []
    for tok in sent:
        if not _named(tok):
            continue
        for c in tok.children:
            if c.dep_ != "prep":
                continue
            for p in c.children:
                if p.dep_ != "pobj" or _has_meta(p) or p.pos_ == "PRON":
                    continue
                head = _clean_np(tok, speaker, speakers)
                tail = _clean_np(p, speaker, speakers)
                if not head or not tail or head.lower() == tail.lower():
                    continue
                out.append(
                    {
                        "head": head,
                        "relation": c.text.upper(),
                        "tail": tail,
                        "pattern": "noun_prep",
                        "head_named": True,
                        "tail_named": _named(p),
                    }
                )
    return out


def extract_relations(
    sent: Any, speaker: str, speakers: list[str], include_noun_prep: bool = True
) -> list[dict[str, Any]]:
    """Union of subject-centric SVO facts + rich dependency-grammar relations.

    Returns dicts carrying ``head, relation, tail, head_named, tail_named`` and a
    ``pattern`` tag ('svo' | 'copular' | 'possessive' | 'appositive' |
    'noun_prep'); SVO triples additionally carry ``verb``/``prep``. Deduped by
    (head, relation, tail); SVO wins ties (it is added first).
    """
    base = [{**r, "pattern": "svo"} for r in extract_svo(sent, speaker, speakers)]
    rich = _copular(sent, speaker, speakers)
    rich += _possessive(sent, speaker, speakers)
    rich += _appositive(sent, speaker, speakers)
    if include_noun_prep:
        rich += _noun_prep(sent, speaker, speakers)
    seen: set[tuple[str, str, str]] = set()
    merged: list[dict[str, Any]] = []
    for r in base + rich:
        k = (r["head"].lower(), r["relation"], r["tail"].lower())
        if k not in seen:
            seen.add(k)
            merged.append(r)
    return merged


def parse_speaker(line: str) -> str:
    """Speaker of a LoCoMo 'ts | Speaker: text' line, or '' when there is none."""
    m = _SPEAKER_LINE.match(line.strip())
    return m.group(1).strip() if m else ""
