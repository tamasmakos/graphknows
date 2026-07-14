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
    """Head-noun phrase for an argument token: strip det/poss, prefer a named
    apposition ('a cat named Luna' -> 'Luna'), keep compounds ('dance studio')."""
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
            if obj_tok.pos_ == "PRON" and obj_tok.text.lower() not in _FIRST_PERSON | _SECOND_PERSON:
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


def parse_speaker(line: str) -> str:
    """Speaker of a LoCoMo 'ts | Speaker: text' line, or '' when there is none."""
    m = _SPEAKER_LINE.match(line.strip())
    return m.group(1).strip() if m else ""
