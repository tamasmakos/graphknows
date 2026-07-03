"""General, domain-agnostic SVO relation extractor.

The grammatical subject–predicate–object skeleton is universal across registers
(chat, articles, transcripts, docs), unlike the narrow speech-act verb register in
:mod:`graphknows.extraction.relations.spacy`. This module reads the spaCy
dependency parse, resolves conversational first/second person to the
speaker/addressee, maps predicates to a compact universal relation taxonomy, and
emits clean (subject, relation, object) triples.

Output triples match the shape consumed by the STM ingest writer:
``{source, source_label, relation, target, target_label, confidence, evidence,
method, negated}``.
"""

from __future__ import annotations

import re
from typing import Any

from graphknows.extraction.relations.spacy import _has_negation, get_spacy_nlp

# ---------------------------------------------------------------------------
# Universal relation taxonomy: predicate lemma -> canonical relation.
# Coarse enough to generalise across domains, specific enough to retrieve on.
# This map IS the relation schema (and the curation allowlist).
# ---------------------------------------------------------------------------
VERB_RELATION: dict[str, str] = {
    # movement / attendance
    "go": "WENT_TO",
    "visit": "WENT_TO",
    "travel": "WENT_TO",
    "come": "WENT_TO",
    "attend": "ATTENDED",
    "move": "MOVED_TO",
    "join": "JOINED",
    # activities / doing / creating
    "do": "DID",
    "run": "DID",
    "play": "DID",
    "paint": "DID",
    "practice": "DID",
    "host": "DID",
    "organize": "DID",
    "read": "DID",
    "watch": "DID",
    "carve": "DID",
    "make": "MADE",
    "build": "MADE",
    "create": "MADE",
    "write": "MADE",
    "cook": "MADE",
    "start": "STARTED",
    "continue": "PURSUES",
    "pursue": "PURSUES",
    "study": "STUDIES",
    "prioritize": "PRIORITIZES",
    # preferences
    "like": "LIKES",
    "love": "LIKES",
    "enjoy": "LIKES",
    "adore": "LIKES",
    "prefer": "PREFERS",
    # desires / plans
    "want": "WANTS",
    "hope": "WANTS",
    "wish": "WANTS",
    "plan": "PLANS_TO",
    "intend": "PLANS_TO",
    "aim": "PLANS_TO",
    "decide": "DECIDED",
    "choose": "DECIDED",
    # possession
    "have": "HAS",
    "own": "HAS",
    "keep": "HAS",
    "get": "GOT",
    "buy": "GOT",
    "adopt": "ADOPTED",
    # emotion
    "feel": "FEELS",
    "miss": "FEELS",
    "cherish": "FEELS",
    "fear": "FEELS",
    "worry": "FEELS",
    # work / knowledge
    "work": "WORKS_ON",
    "know": "KNOWS",
    "meet": "MET",
}

# Copular "be" + noun phrase -> identity/role.
_COPULA = frozenset({"be"})

# Turn delimiter: a capitalised 1–2 word speaker name followed by ": ". Works on
# both newline-separated and space-flattened transcripts (the chunker joins
# sentences with spaces, dropping line breaks), so we cannot rely on ^...$ lines.
_TURN_RE = re.compile(r"(?:^|[\s|])([A-Z][A-Za-z'’.-]+(?:\s[A-Z][A-Za-z'’.-]+)?):\s")
# Trailing timestamp that belongs to the *next* turn, left dangling after slicing.
_TS_TAIL = re.compile(r"\d{1,2}:\d{2}\s*[ap]?m?\b.*$", re.I | re.S)
_DET_RE = re.compile(r"^(a|an|the|my|your|his|her|their|our|some|any|that|this)\s+", re.I)

_FIRST_PERSON = frozenset({"i", "me", "my", "myself", "mine", "i'm", "im"})
_SECOND_PERSON = frozenset({"you", "your", "yours", "u"})
_DISCOURSE = frozenset(
    {
        "it",
        "that",
        "this",
        "these",
        "those",
        "which",
        "there",
        "here",
        "what",
        "who",
        "they",
        "we",
        "one",
        "everyone",
        "something",
        "anyone",
        "nothing",
    }
)
_FILLER_OBJ = frozenset(
    {
        "you",
        "me",
        "them",
        "us",
        "him",
        "her",
        "best",
        "thing",
        "lot",
        "stuff",
        "way",
        "kind",
        "sort",
        "part",
        "bit",
    }
)
_NOISE_LABELS = frozenset({"TIME", "DATE", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"})
_BARE_NUM = re.compile(r"^[\W\d]+$")


def _speaker_units(text: str) -> tuple[list[tuple[str | None, str]], set[str]]:
    """Split a transcript into (speaker, content) turns; ``(None, text)`` for prose.

    Robust to space-flattened text: turns are delimited by the universal
    ``Name:`` marker.  A name is accepted as a real speaker if it appears at
    the very start of a line/text (threshold = 1 occurrence) or recurs >=2
    times anywhere, to avoid misclassifying mid-sentence colons in prose.
    """
    _LINE_START_RE = re.compile(r"(?:^|\n)([A-Z][A-Za-z''.-]+(?:\s[A-Z][A-Za-z''.-]+)?):\s")
    line_start_names: set[str] = {m.group(1).strip() for m in _LINE_START_RE.finditer(text)}
    matches = list(_TURN_RE.finditer(text))
    freq: dict[str, int] = {}
    for m in matches:
        freq[m.group(1).strip()] = freq.get(m.group(1).strip(), 0) + 1
    speakers = {name for name, n in freq.items() if n >= 2 or name in line_start_names}
    if not speakers:
        return [(None, text)], set()

    units: list[tuple[str | None, str]] = []
    turns = [m for m in matches if m.group(1).strip() in speakers]
    for i, m in enumerate(turns):
        sp = m.group(1).strip()
        start = m.end()
        end = turns[i + 1].start() if i + 1 < len(turns) else len(text)
        content = _TS_TAIL.sub("", text[start:end]).strip(" |\n\t")
        if content:
            units.append((sp, content))
    return units, speakers


def _clean_obj(span: str) -> str:
    return _DET_RE.sub("", span.strip()).strip(" .,!?;:")


def extract_svo_relations(
    text: str,
    entities: list[dict[str, Any]],
    spacy_model: str = "en_core_web_lg",
) -> list[dict[str, Any]]:
    """Extract universal SVO relations, resolving conversational participants.

    Args:
        text: Raw chunk text (may contain ``Speaker:`` prefixed dialogue lines).
        entities: Clean entity dicts (``name``/``type``) used to validate subjects.
        spacy_model: spaCy model with a dependency parser + NER.

    Returns:
        List of relation-triple dicts (see module docstring).
    """
    nlp = get_spacy_nlp(spacy_model)
    units, speakers = _speaker_units(text)
    entity_names = [e.get("name", "").lower() for e in entities if e.get("name")]

    def is_entity(name: str) -> bool:
        t = name.strip().lower()
        if not t:
            return False
        if t in speakers or any(t == s.lower() for s in speakers):
            return True
        return any(
            t == n or (len(t) > 3 and t in n) or (len(n) > 3 and n in t) for n in entity_names
        )

    def resolve(token_text: str, speaker: str | None, others: set[str]) -> str:
        t = token_text.lower().strip()
        if speaker and t in _FIRST_PERSON:
            return speaker
        if speaker and t in _SECOND_PERSON:
            return next(iter(sorted(others)), token_text)
        return token_text

    # Canonicalization vocab. Subject aliases collapse against the *speaker* set
    # only (participants), so real distinct entities are never merged; objects
    # reuse an existing entity node's spelling by exact match.
    _entity_orig = [e.get("name", "") for e in entities if e.get("name")]
    _speaker_vocab = sorted(speakers, key=len, reverse=True)

    def canon_subject(name: str) -> str:
        """Collapse a person alias to its longest prefix-compatible participant name.

        Generic (no hard-coded names): a short form is unified with a longer one
        when either is a case-insensitive prefix of the other (e.g. a nickname of
        a speaker). Longest match wins so fragments converge on one node.
        """
        nl = name.lower().strip()
        if not nl:
            return name
        for cand in _speaker_vocab:
            cl = cand.lower()
            if cl != nl and (cl.startswith(nl) or nl.startswith(cl)):
                return cand
        return name

    def canon_object(name: str) -> str:
        """Reuse an existing entity node's canonical spelling for an object, when
        the object matches one case-insensitively (exact match only, no fuzzy
        merge, so distinct concepts are never conflated)."""
        nl = name.lower().strip()
        for cand in _entity_orig:
            if cand.lower() == nl and cand != name:
                return cand
        return name

    triples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for speaker, content in units:
        if not content.strip() or content.strip().endswith("?"):
            continue  # questions are not facts
        others = speakers - {speaker} if speaker else set()
        doc = nlp(content)
        for tok in doc:
            if tok.pos_ not in ("VERB", "AUX"):
                continue
            lemma = tok.lemma_.lower()
            relation = VERB_RELATION.get(lemma) or ("IS_A" if lemma in _COPULA else None)
            if relation is None:
                continue
            subs = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")]
            if not subs:
                continue
            obj_toks = []
            for c in tok.children:
                if c.dep_ in ("dobj", "attr", "dative", "oprd"):
                    obj_toks.append(c)
                elif c.dep_ == "prep":
                    obj_toks.extend(p for p in c.children if p.dep_ == "pobj")
                elif c.dep_ == "acomp":
                    for gc in c.children:
                        if gc.dep_ == "prep":
                            obj_toks.extend(p for p in gc.children if p.dep_ == "pobj")
            negated = _has_negation(tok)
            for sub in subs:
                src = resolve(sub.text, speaker, others)
                if src.lower() in _DISCOURSE or not is_entity(src):
                    continue
                for o in obj_toks:
                    ospan = doc[o.left_edge.i : o.right_edge.i + 1]
                    if not any(t.pos_ in ("NOUN", "PROPN") for t in ospan):
                        continue  # object must be a noun phrase, not adj/pronoun
                    obj = _clean_obj(ospan.text)
                    ol = obj.lower()
                    if (
                        len(obj) < 3
                        or ol in _DISCOURSE
                        or ol in _FILLER_OBJ
                        or _BARE_NUM.match(obj)
                    ):
                        continue
                    key = (src.lower(), relation, ol)
                    if key in seen:
                        continue
                    seen.add(key)
                    triples.append(
                        {
                            "source": canon_subject(src),
                            "source_label": "ENTITY",
                            "relation": relation,
                            "target": canon_object(obj),
                            "target_label": "CONCEPT",
                            "confidence": 0.7,
                            "evidence": content.strip(),
                            "method": f"svo:{lemma}",
                            "negated": negated,
                        }
                    )
    return triples
