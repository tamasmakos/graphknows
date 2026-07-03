"""Enhanced deterministic fact extraction (llm_free-legal, no LLM at ingest).

Layers on top of the bare SVO extractor to break its coverage ceiling — the SVO
path only fires when the subject is an explicit name and drops facts buried in
prepositions, conjunctions and subordinate clauses. All levers are deterministic
(spaCy dependency parse), toggleable, and emit the same triple dicts as
``extract_svo_relations`` so the provenance-carrying edge writer is unchanged.

Levers
------
* ``use_patterns`` — beyond bare dobj SVO: an extended verb→relation map,
  prepositional objects ("attended / joined / lives in X"), copular attributes,
  and CONJUNCTION EXPANSION (split "camping and hiking" into two facts).

Quality guards (always on): negation drop, and head-noun trimming of objects via
noun chunks so clause-objects ("all the support I've got …") collapse to a head
noun ("support") instead of leaking whole clauses.
"""

from __future__ import annotations

import logging
from typing import Any

from graphknows.extraction.relations.svo import (
    _COPULA,
    _DISCOURSE,
    _FILLER_OBJ,
    _FIRST_PERSON,
    _SECOND_PERSON,
    VERB_RELATION,
    _clean_obj,
    _has_negation,
    _speaker_units,
    get_spacy_nlp,
)

log = logging.getLogger(__name__)

# Extra verb lemmas the bare map omits (deterministic, domain-agnostic).
_EXTRA_VERBS: dict[str, str] = {
    "attend": "WENT_TO",
    "join": "JOINED",
    "enroll": "JOINED",
    "live": "WENT_TO",
    "move": "WENT_TO",
    "return": "WENT_TO",
    "explore": "WENT_TO",
    "see": "DID",
    "try": "DID",
    "start": "DID",
    "finish": "DID",
    "celebrate": "DID",
    "study": "DID",
    "learn": "DID",
    "teach": "DID",
    "volunteer": "DID",
    "participate": "DID",
    "buy": "GOT",
    "adopt": "GOT",
    "receive": "GOT",
    "pass": "DID",
    "share": "DID",
    "take": "DID",
    "spend": "DID",
    "win": "GOT",
    "complete": "DID",
    "pursue": "DID",
    "talk": "DID",
    "plan": "DID",
    "decide": "DID",
    "help": "DID",
    "meet": "MET",
    "remember": "HAS",
    "own": "HAS",
    "love": "LIKES",
    "enjoy": "LIKES",
    "prefer": "LIKES",
}

# First-person plural resolves to the speaker too ("we celebrated" said by X → X).
_FIRST_PERSON_PLURAL = frozenset({"we", "us", "our", "ours", "ourselves"})
_VERB_REL = {**VERB_RELATION, **_EXTRA_VERBS}

def _obj_texts(tok: Any, doc: Any, use_patterns: bool) -> list[str]:
    """Return cleaned object strings for a verb token (head-noun trimmed, conj-expanded)."""
    # Map each token index to its enclosing noun chunk (head-noun phrase, no relcl).
    chunk_of: dict[int, Any] = {}
    for nc in doc.noun_chunks:
        for t in nc:
            chunk_of[t.i] = nc

    heads: list[Any] = []
    activities: list[str] = []  # xcomp verbs used as activity objects ("went hiking")
    for c in tok.children:
        if c.dep_ in ("dobj", "attr", "dative", "oprd"):
            heads.append(c)
        elif c.dep_ == "prep":
            heads.extend(p for p in c.children if p.dep_ == "pobj")
        elif c.dep_ == "acomp":
            for gc in c.children:
                if gc.dep_ == "prep":
                    heads.extend(p for p in gc.children if p.dep_ == "pobj")
        elif use_patterns and c.dep_ == "xcomp":
            # "went hiking", "started painting" — the xcomp verb IS the activity;
            # if it governs its own object ("want to pursue counseling") use that.
            inner = [g for g in c.children if g.dep_ in ("dobj", "pobj")]
            if inner:
                heads.extend(inner)
            elif c.pos_ in ("VERB", "NOUN"):
                activities.append(c.lemma_.lower() if c.pos_ == "VERB" else c.text)
    # Conjunction expansion: each object head's conj siblings become extra objects.
    expanded: list[Any] = []
    for h in heads:
        expanded.append(h)
        if use_patterns:
            expanded.extend(c for c in h.children if c.dep_ == "conj")

    out: list[str] = []
    for o in expanded:
        nc = chunk_of.get(o.i)
        span = nc.text if nc is not None else doc[o.left_edge.i : o.right_edge.i + 1].text
        if not any(t.pos_ in ("NOUN", "PROPN") for t in (nc if nc is not None else [o])):
            continue
        out.append(_clean_obj(span))
    out.extend(activities)
    return out


def extract_facts(
    text: str,
    entities: list[dict[str, Any]],
    *,
    use_patterns: bool = False,
    spacy_model: str = "en_core_web_lg",
) -> list[dict[str, Any]]:
    """Extract deterministic SVO+ facts with richer dependency patterns.

    Returns the same triple-dict shape as ``extract_svo_relations`` (source,
    relation, target, evidence, …) so the provenance edge writer is unchanged.
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
        if speaker and (t in _FIRST_PERSON or t in _FIRST_PERSON_PLURAL):
            return speaker
        if speaker and t in _SECOND_PERSON:
            return next(iter(sorted(others)), token_text)
        return token_text

    verb_map = _VERB_REL if use_patterns else VERB_RELATION
    triples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for speaker, content in units:
        if not content.strip() or content.strip().endswith("?"):
            continue
        others = speakers - {speaker} if speaker else set()
        doc = nlp(content)
        for tok in doc:
            if tok.pos_ not in ("VERB", "AUX"):
                continue
            lemma = tok.lemma_.lower()
            relation = verb_map.get(lemma) or ("IS_A" if lemma in _COPULA else None)
            if relation is None:
                continue
            subs = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")]
            if not subs:
                continue
            negated = _has_negation(tok)
            if negated:
                continue  # quality guard: never emit a negated fact as positive
            objs = _obj_texts(tok, doc, use_patterns)
            for sub in subs:
                src = resolve(sub.text, speaker, others)
                if src.lower() in _DISCOURSE or not is_entity(src):
                    continue
                for obj in objs:
                    ol = obj.lower()
                    if len(obj) < 3 or ol in _DISCOURSE or ol in _FILLER_OBJ:
                        continue
                    key = (src.lower(), relation, ol)
                    if key in seen:
                        continue
                    seen.add(key)
                    triples.append(
                        {
                            "source": src,
                            "source_label": "ENTITY",
                            "relation": relation,
                            "target": obj,
                            "target_label": "CONCEPT",
                            "confidence": 0.7,
                            "evidence": content.strip(),
                            "method": f"facts:{lemma}",
                            "negated": False,
                        }
                    )
    return triples
