"""SpaCy dependency-pattern relation extractor for intentionality-typed triples.

Architectural role
------------------
GLiNER2 (taxonomy-guided) extracts ENTITY TYPE SPANS — what things are.
This module extracts STRUCTURAL RELATIONS — how those things relate, by reading
SpaCy's dependency parse rather than asking GLiNER2's relation head.

The two layers are complementary:
  - GLiNER2 finds: Brentano → AGENT, intentionality → CONCEPT
  - SpaCy finds:   Brentano ASSERTS "intentionality is the mark of the mental"

Why dep parsing and not GLiNER2 for relations
----------------------------------------------
GLiNER2's relation head extracts spans using semantic similarity against a
description, not syntactic structure. For proposition-level targets (full
subordinate clauses) and semantically precise relation types (BELIEVES vs
ASSERTS vs INTENDS), dependency arcs are deterministic and require no LLM.

Patterns implemented
--------------------
A  Cognitive verb + complement:
     nsubj → VERB(cognitive) → ccomp | xcomp | dobj
     e.g. "Brentano CLAIMS [that intentionality is the mark of the mental]"
          "Sarah INTENDS [to migrate the Go services]"

B  Copular definition:
     nsubj → is/becomes(ROOT) → attr
     e.g. "Intentionality IS [the mark of the mental]"

C  Passive definition:
     nsubjpass → defined/characterized → prep(as) → pobj
     e.g. "Intentionality IS DEFINED AS [the property of being directed]"

D  Appositive:
     token(appos) → head
     e.g. "Belief, [a paradigmatic epistemic state], exhibits intentionality"
     → Belief EXEMPLIFIES epistemic state

E  Referential verb + prepositional object:
     nsubj → refer/denote/express → prep(to/for) → pobj | dobj
     e.g. "Sinn REFERS TO [the sense of an expression]"

F  Distinction verb:
     nsubj → distinguish/contrast → dobj
                                   → prep(from) → pobj  (if present)
     e.g. "Husserl DISTINGUISHES [noetic act] FROM [noematic content]"

G  Deontic modal obligation (Stratum IV — directive illocution):
     nsubj → VERB[AUX:must/shall/should/ought] → VP
     e.g. "The system must process requests"
          → system DIRECTS "process requests"
          "Sarah should call John" → Sarah DIRECTS "call John"

H  Causal subordinate clause (Stratum V — abductive reasoning):
     nsubj(main) → ROOT(main) ←advcl← ROOT(causal)[mark:because/since]
     e.g. "The file was deleted because storage was full"
          → "storage was full" EXPLAINS file

I  Discourse connective entailment (Stratum V — deductive reasoning):
     ROOT(ant) ←parataxis/conj← ROOT(con)[advmod:therefore/thus/hence]
     e.g. "The experiment failed; therefore the study was invalid"
          → "The experiment failed" ENTAILS "the study was invalid"

Entity grounding
----------------
For each argument span, overlap-match against GLiNER2 entities by text.
If matched → use GLiNER2 entity label as the arg type.
If not matched → use SpaCy NER type mapped to intentionality label,
               or "ENTITY" as fallback.
Clause-level targets (ccomp, xcomp) → assumed label "PROPOSITION".

Output format
-------------
Each triple: {
    "source":        str,   # surface text of subject/head argument
    "source_label":  str,   # intentionality label (AGENT, CONCEPT, …)
    "relation":      str,   # from relation_types vocabulary
    "target":        str,   # surface text of object/complement argument
    "target_label":  str,
    "evidence":      str,   # original sentence text
    "confidence":    float, # 0.9 exact structural, 0.75 appositive/attr
    "method":        str,   # "spacy_dep:A" etc.
    "negated":       bool,
}
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cognitive verb → relation type mapping
# ---------------------------------------------------------------------------

_ASSERTIVE = "ASSERTS"
_EPISTEMIC = "BELIEVES"
_CONATIVE = "INTENDS"
_AFFECTIVE = "FEELS"
_PERCEPTUAL = "PERCEIVES"
_DOUBTING = "DOUBTS"
_ILLOCUTIONARY_DEC = "DECLARES"
_ILLOCUTIONARY_PRO = "PROMISES"
_ILLOCUTIONARY_COM = "COMMANDS"
_ILLOCUTIONARY_EXP = "EXPRESSES"  # Searle expressives: thank, apologize, congratulate
_REFERENTIAL = "REFERS_TO"
_DENOTATIONAL = "DENOTES"
_EXPRESSIVE = "EXPRESSES"
_PRESUPPOSITIONAL = "PRESUPPOSES"
_DEFINITIONAL = "DEFINED_AS"
_DISTINCTION = "DISTINGUISHES"
_EXEMPLIFICATION = "EXEMPLIFIES"
_ENTAILMENT = "ENTAILS"
_SUPPORTIVE = "SUPPORTS"
_EXPLANATORY = "EXPLAINS"
_CONTRADICTION = "CONTRADICTS"
_GROUNDING = "GROUNDS"
_CRITICAL = "CRITIQUES"
_DEFENSIVE = "DEFENDS"

COGNITIVE_VERB_MAP: dict[str, str] = {
    # ── Searle: ASSERTIVES — speaker commits to truth of p ────────────────────
    # Verba dicendi: the complete English semantic class of saying/reporting verbs
    "say": _ASSERTIVE,
    "tell": _ASSERTIVE,
    "state": _ASSERTIVE,
    "claim": _ASSERTIVE,
    "assert": _ASSERTIVE,
    "report": _ASSERTIVE,
    "announce": _ASSERTIVE,
    "mention": _ASSERTIVE,
    "note": _ASSERTIVE,
    "remark": _ASSERTIVE,
    "describe": _ASSERTIVE,
    "specify": _ASSERTIVE,
    "narrate": _ASSERTIVE,
    "allege": _ASSERTIVE,
    "submit": _ASSERTIVE,
    "affirm": _ASSERTIVE,
    "aver": _ASSERTIVE,
    "attest": _ASSERTIVE,
    "inform": _ASSERTIVE,
    "reply": _ASSERTIVE,
    "answer": _ASSERTIVE,
    "reveal": _ASSERTIVE,
    "disclose": _ASSERTIVE,
    "divulge": _ASSERTIVE,
    "publish": _ASSERTIVE,
    # Epistemic commitment expressed as public speech act
    "acknowledge": _ASSERTIVE,
    "admit": _ASSERTIVE,
    "concede": _ASSERTIVE,
    "grant": _ASSERTIVE,
    "confess": _ASSERTIVE,
    # Argumentative / evidential assertives
    "argue": _ASSERTIVE,
    "maintain": _ASSERTIVE,
    "contend": _ASSERTIVE,
    "insist": _ASSERTIVE,
    "hold": _ASSERTIVE,
    "posit": _ASSERTIVE,
    "hypothesize": _ASSERTIVE,
    "theorize": _ASSERTIVE,
    "conjecture": _ASSERTIVE,
    "propose": _ASSERTIVE,
    "suggest": _ASSERTIVE,
    "show": _ASSERTIVE,
    "demonstrate": _ASSERTIVE,
    "prove": _ASSERTIVE,
    "establish": _ASSERTIVE,
    "conclude": _ASSERTIVE,
    "find": _ASSERTIVE,
    "assume": _ASSERTIVE,
    "presume": _ASSERTIVE,
    "suppose": _ASSERTIVE,
    "predict": _ASSERTIVE,
    "forecast": _ASSERTIVE,
    "anticipate": _ASSERTIVE,
    # ── Brentano: DOXASTIC — propositional attitude of belief / knowledge ─────
    "believe": _EPISTEMIC,
    "know": _EPISTEMIC,
    "think": _EPISTEMIC,
    "recognize": _EPISTEMIC,
    "understand": _EPISTEMIC,
    "suspect": _EPISTEMIC,
    # ── Brentano: CONATIVE — propositional attitude of desire / intention ─────
    "intend": _CONATIVE,
    "plan": _CONATIVE,
    "aim": _CONATIVE,
    "seek": _CONATIVE,
    "want": _CONATIVE,
    "try": _CONATIVE,
    "attempt": _CONATIVE,
    "mean": _CONATIVE,
    # ── Brentano: AFFECTIVE — propositional attitude of emotion ──────────────
    "fear": _AFFECTIVE,
    "resent": _AFFECTIVE,
    "admire": _AFFECTIVE,
    "love": _AFFECTIVE,
    "hate": _AFFECTIVE,
    "dread": _AFFECTIVE,
    "hope": _AFFECTIVE,
    "worry": _AFFECTIVE,
    "regret": _AFFECTIVE,
    # ── Brentano: PERCEPTUAL ─────────────────────────────────────────────────
    "perceive": _PERCEPTUAL,
    "observe": _PERCEPTUAL,
    "see": _PERCEPTUAL,
    "notice": _PERCEPTUAL,
    "detect": _PERCEPTUAL,
    # ── Epistemic DOUBT (negated assertive commitment) ───────────────────────
    "doubt": _DOUBTING,
    "question": _DOUBTING,
    "deny": _DOUBTING,
    # ── Searle: DIRECTIVES — speaker directs hearer to bring about p ─────────
    "order": _ILLOCUTIONARY_COM,
    "command": _ILLOCUTIONARY_COM,
    "demand": _ILLOCUTIONARY_COM,
    "require": _ILLOCUTIONARY_COM,
    "request": _ILLOCUTIONARY_COM,
    "beg": _ILLOCUTIONARY_COM,
    "urge": _ILLOCUTIONARY_COM,
    "instruct": _ILLOCUTIONARY_COM,
    "direct": _ILLOCUTIONARY_COM,
    "invite": _ILLOCUTIONARY_COM,
    "implore": _ILLOCUTIONARY_COM,
    "entreat": _ILLOCUTIONARY_COM,
    "petition": _ILLOCUTIONARY_COM,
    "advise": _ILLOCUTIONARY_COM,
    "recommend": _ILLOCUTIONARY_COM,
    "warn": _ILLOCUTIONARY_COM,
    "encourage": _ILLOCUTIONARY_COM,
    "discourage": _ILLOCUTIONARY_COM,
    "forbid": _ILLOCUTIONARY_COM,
    "prohibit": _ILLOCUTIONARY_COM,
    "permit": _ILLOCUTIONARY_COM,
    "allow": _ILLOCUTIONARY_COM,
    # ── Searle: COMMISSIVES — speaker commits to future action ───────────────
    "promise": _ILLOCUTIONARY_PRO,
    "pledge": _ILLOCUTIONARY_PRO,
    "vow": _ILLOCUTIONARY_PRO,
    "swear": _ILLOCUTIONARY_PRO,
    "guarantee": _ILLOCUTIONARY_PRO,
    "warrant": _ILLOCUTIONARY_PRO,
    "undertake": _ILLOCUTIONARY_PRO,
    "commit": _ILLOCUTIONARY_PRO,
    "offer": _ILLOCUTIONARY_PRO,
    "volunteer": _ILLOCUTIONARY_PRO,
    "threaten": _ILLOCUTIONARY_PRO,
    # ── Searle: EXPRESSIVES — speaker expresses psychological state about p ──
    "thank": _ILLOCUTIONARY_EXP,
    "congratulate": _ILLOCUTIONARY_EXP,
    "praise": _ILLOCUTIONARY_EXP,
    "compliment": _ILLOCUTIONARY_EXP,
    "welcome": _ILLOCUTIONARY_EXP,
    "condole": _ILLOCUTIONARY_EXP,
    "sympathize": _ILLOCUTIONARY_EXP,
    "lament": _ILLOCUTIONARY_EXP,
    "deplore": _ILLOCUTIONARY_EXP,
    "apologize": _ILLOCUTIONARY_EXP,
    # ── Searle: DECLARATIVES — speaker changes institutional reality ──────────
    "declare": _ILLOCUTIONARY_DEC,
    "proclaim": _ILLOCUTIONARY_DEC,
    "rule": _ILLOCUTIONARY_DEC,
    "pronounce": _ILLOCUTIONARY_DEC,
    "adjourn": _ILLOCUTIONARY_DEC,
    "appoint": _ILLOCUTIONARY_DEC,
    "nominate": _ILLOCUTIONARY_DEC,
    "designate": _ILLOCUTIONARY_DEC,
    "elect": _ILLOCUTIONARY_DEC,
    "hire": _ILLOCUTIONARY_DEC,
    "dismiss": _ILLOCUTIONARY_DEC,
    "fire": _ILLOCUTIONARY_DEC,
    "sentence": _ILLOCUTIONARY_DEC,
    "acquit": _ILLOCUTIONARY_DEC,
    "convict": _ILLOCUTIONARY_DEC,
    "enact": _ILLOCUTIONARY_DEC,
    "ratify": _ILLOCUTIONARY_DEC,
    "veto": _ILLOCUTIONARY_DEC,
    "resign": _ILLOCUTIONARY_DEC,
    "inaugurate": _ILLOCUTIONARY_DEC,
    # ── Semiotic: how signs relate to their objects (Peirce semiosis) ─────────
    "refer": _REFERENTIAL,
    "point": _REFERENTIAL,
    "denote": _DENOTATIONAL,
    "name": _DENOTATIONAL,
    "signify": _DENOTATIONAL,
    "represent": _EXPRESSIVE,
    "convey": _EXPRESSIVE,
    "encode": _EXPRESSIVE,
    "express": _EXPRESSIVE,
    "presuppose": _PRESUPPOSITIONAL,
    # ── Definitional / taxonomic ──────────────────────────────────────────────
    "define": _DEFINITIONAL,
    "characterize": _DEFINITIONAL,
    "constitute": _DEFINITIONAL,
    # ── Distinction ──────────────────────────────────────────────────────────
    "distinguish": _DISTINCTION,
    "differentiate": _DISTINCTION,
    "contrast": _DISTINCTION,
    "separate": _DISTINCTION,
    "compare": _DISTINCTION,
    # ── Exemplification ──────────────────────────────────────────────────────
    "exhibit": _EXEMPLIFICATION,
    "exemplify": _EXEMPLIFICATION,
    "illustrate": _EXEMPLIFICATION,
    "instantiate": _EXEMPLIFICATION,
    "display": _EXEMPLIFICATION,
    # ── Peirce: DEDUCTION — what necessarily follows ──────────────────────────
    "entail": _ENTAILMENT,
    "imply": _ENTAILMENT,
    "follow": _ENTAILMENT,
    # ── Peirce: INDUCTION — what the evidence supports ───────────────────────
    "support": _SUPPORTIVE,
    "confirm": _SUPPORTIVE,
    "corroborate": _SUPPORTIVE,
    "indicate": _SUPPORTIVE,
    # ── Peirce: ABDUCTION — what best explains ───────────────────────────────
    "explain": _EXPLANATORY,
    "account": _EXPLANATORY,
    "cause": _EXPLANATORY,
    # ── Meta-reasoning ───────────────────────────────────────────────────────
    "contradict": _CONTRADICTION,
    "conflict": _CONTRADICTION,
    "ground": _GROUNDING,
    # ── Critical / defensive ─────────────────────────────────────────────────
    "critique": _CRITICAL,
    "criticize": _CRITICAL,
    "reject": _CRITICAL,
    "refute": _CRITICAL,
    "challenge": _CRITICAL,
    "oppose": _CRITICAL,
    "defend": _DEFENSIVE,
    "advocate": _DEFENSIVE,
}

_COPULA_LEMMAS: frozenset[str] = frozenset({"be", "become", "remain"})
_DEFINITIONAL_LEMMAS: frozenset[str] = frozenset(
    {"define", "characterize", "constitute", "mean", "represent"}
)

# Stratum IV — directive illocution: deontic modals
_DEONTIC_MODALS: frozenset[str] = frozenset({"must", "shall", "should", "ought"})

# Stratum V — Peirce abductive connectives (causal "mark" dep → EXPLAINS)
_CAUSAL_MARKS: frozenset[str] = frozenset({"because", "since"})

# Stratum V — Peirce deductive connectives (discourse marker → ENTAILS)
# Inductive ("for example", "such as") is covered verbally via COGNITIVE_VERB_MAP
# (support/confirm/corroborate/indicate) but lacks a discourse-marker pattern.
_ENTAILMENT_CONNECTIVES: frozenset[str] = frozenset(
    {"therefore", "thus", "hence", "consequently", "accordingly", "ergo"}
)

# SpaCy NER type → intentionality label fallback
_SPACY_NER_TO_LABEL: dict[str, str] = {
    "PERSON": "AGENT",
    "ORG": "AGENT",
    "NORP": "AGENT",
    "FAC": "LOCATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "PRODUCT": "ARTIFACT",
    "WORK_OF_ART": "ARTIFACT",
    "LAW": "ARTIFACT",
    "EVENT": "EVENT",
    "DATE": "TIME",
    "TIME": "TIME",
    "MONEY": "CONCEPT",
    "PERCENT": "CONCEPT",
    "QUANTITY": "CONCEPT",
}


# ---------------------------------------------------------------------------
# Singleton spaCy model loader
# ---------------------------------------------------------------------------

_NLP_INSTANCE: Any = None


def get_spacy_nlp(model: str = "en_core_web_lg") -> Any:
    """Return shared spaCy pipeline (lazy load, singleton)."""
    global _NLP_INSTANCE
    if _NLP_INSTANCE is None:
        import spacy

        try:
            _NLP_INSTANCE = spacy.load(model)
            log.info("spaCy model '%s' loaded.", model)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model}' not found. Run: python -m spacy download {model}"
            ) from exc
    return _NLP_INSTANCE


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _span_text(doc: Any, token: Any) -> str:
    """Full subtree text for a token, normalised."""
    return _norm(doc[token.left_edge.i : token.right_edge.i + 1].text)


def _strip_leading_complementizer(text: str) -> str:
    """Remove leading 'that', 'whether', 'if' from a ccomp clause text."""
    for comp in ("that ", "whether ", "if "):
        if text.lower().startswith(comp):
            return text[len(comp) :]
    return text


def _is_pronoun(token: Any) -> bool:
    return token.pos_ == "PRON" and token.lemma_.lower() in {
        "it",
        "he",
        "she",
        "they",
        "we",
        "i",
        "you",
        "this",
        "that",
        "these",
        "those",
        "one",
        "someone",
        "anyone",
        "nobody",
        "everybody",
        "something",
        "anything",
    }


def _has_negation(verb: Any) -> bool:
    """True if the verb has a negation modifier."""
    return any(c.dep_ == "neg" for c in verb.children)


# ---------------------------------------------------------------------------
# Entity grounding
# ---------------------------------------------------------------------------


def _overlaps(a: str, b: str) -> bool:
    na, nb = a.lower().strip(), b.lower().strip()
    return bool(na and nb and (na in nb or nb in na))


def _ground_span(
    span_text: str,
    gliner_entities: list[dict[str, Any]],
    spacy_ent_map: dict[tuple[int, int], tuple[str, str]],
    default_label: str = "ENTITY",
) -> tuple[str, str]:
    """Align a dep-extracted span with a GLiNER2 entity. Return (text, label)."""
    # Prefer GLiNER2 match (richer label set).
    # Accepts both graphgen format (text/label/ontology_label) and
    # memory-extractor format (name/type) so this module can be called from
    # either pipeline without a conversion step.
    best: dict | None = None
    best_len = 0
    for ent in gliner_entities:
        ent_text = (ent.get("text") or ent.get("name") or "").strip()
        if _overlaps(span_text, ent_text) and len(ent_text) > best_len:
            best = ent
            best_len = len(ent_text)
    if best is not None:
        canonical = (best.get("text") or best.get("name") or span_text).strip()
        label = (
            best.get("label") or best.get("ontology_label") or best.get("type") or default_label
        ).upper()
        return canonical, label

    # Fallback: SpaCy NER type
    for ner_type, ent_surface in spacy_ent_map.values():
        if _overlaps(span_text, ent_surface):
            mapped = _SPACY_NER_TO_LABEL.get(ner_type, default_label)
            return span_text.strip(), mapped

    return span_text.strip(), default_label


def _build_spacy_ent_map(doc: Any) -> dict[tuple[int, int], tuple[str, str]]:
    return {(ent.start, ent.end): (ent.label_, ent.text) for ent in doc.ents}


# ---------------------------------------------------------------------------
# Triple builder
# ---------------------------------------------------------------------------


def _triple(
    source: str,
    source_label: str,
    relation: str,
    target: str,
    target_label: str,
    evidence: str,
    confidence: float,
    pattern_id: str,
    negated: bool = False,
) -> dict[str, Any]:
    return {
        "source": source,
        "source_label": source_label,
        "relation": relation,
        "target": target,
        "target_label": target_label,
        "evidence": _norm(evidence),
        "confidence": round(confidence, 3),
        "method": f"spacy_dep:{pattern_id}",
        "negated": negated,
    }


# ---------------------------------------------------------------------------
# Pattern extractors (one per pattern type)
# ---------------------------------------------------------------------------


def _pattern_A(
    sent: Any,
    doc: Any,
    gliner_entities: list[dict],
    spacy_ent_map: dict,
) -> list[dict[str, Any]]:
    """Cognitive verb + nsubj + (ccomp | xcomp | dobj)."""
    results: list[dict] = []
    for token in sent:
        if token.pos_ not in ("VERB", "AUX") or token.dep_ not in (
            "ROOT",
            "relcl",
            "advcl",
            "ccomp",
            "xcomp",
            "acl",
        ):
            continue
        lemma = (token.lemma_ or token.text).lower()
        relation = COGNITIVE_VERB_MAP.get(lemma)
        if not relation:
            continue

        # Find subject(s)
        subjects = [
            c for c in token.children if c.dep_ in ("nsubj", "nsubjpass") and not _is_pronoun(c)
        ]
        if not subjects:
            continue

        negated = _has_negation(token)
        sent_text = sent.text.strip()

        for subj in subjects:
            src_text = _span_text(doc, subj)
            src_text, src_label = _ground_span(src_text, gliner_entities, spacy_ent_map, "AGENT")

            # Priority: ccomp > xcomp > dobj
            ccomp = next((c for c in token.children if c.dep_ == "ccomp"), None)
            xcomp = next((c for c in token.children if c.dep_ == "xcomp"), None)
            dobj = next((c for c in token.children if c.dep_ == "dobj"), None)

            if ccomp is not None:
                raw = _span_text(doc, ccomp)
                tgt_text = _strip_leading_complementizer(raw)
                # Never ground clause-level targets — entity overlap would replace
                # "Python is the best language for data pipelines" with "data pipelines".
                results.append(
                    _triple(
                        src_text,
                        src_label,
                        relation,
                        tgt_text,
                        "PROPOSITION",
                        sent_text,
                        0.90,
                        "A-ccomp",
                        negated,
                    )
                )

            elif xcomp is not None:
                raw = _span_text(doc, xcomp)
                tgt_text = re.sub(r"^to\s+", "", raw, flags=re.IGNORECASE)
                # Same: keep full infinitival clause, don't ground to a substring entity.
                results.append(
                    _triple(
                        src_text,
                        src_label,
                        relation,
                        tgt_text,
                        "PROPOSITION",
                        sent_text,
                        0.85,
                        "A-xcomp",
                        negated,
                    )
                )

            elif dobj is not None:
                tgt_text = _span_text(doc, dobj)
                # Direct NP objects are safe to ground against GLiNER2 entities.
                tgt_text, tgt_label = _ground_span(
                    tgt_text, gliner_entities, spacy_ent_map, "ENTITY"
                )
                results.append(
                    _triple(
                        src_text,
                        src_label,
                        relation,
                        tgt_text,
                        tgt_label,
                        sent_text,
                        0.80,
                        "A-dobj",
                        negated,
                    )
                )

    return results


def _pattern_B(
    sent: Any,
    doc: Any,
    gliner_entities: list[dict],
    spacy_ent_map: dict,
) -> list[dict[str, Any]]:
    """Copular definition: nsubj + is/becomes(ROOT) + attr."""
    results: list[dict] = []
    root = next((t for t in sent if t.dep_ == "ROOT"), None)
    if root is None or root.lemma_.lower() not in _COPULA_LEMMAS:
        return results

    subjects = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass") and not _is_pronoun(c)]
    attrs = [c for c in root.children if c.dep_ == "attr"]
    if not subjects or not attrs:
        return results

    sent_text = sent.text.strip()
    for subj in subjects:
        src_text = _span_text(doc, subj)
        src_text, src_label = _ground_span(src_text, gliner_entities, spacy_ent_map, "CONCEPT")
        for attr in attrs:
            tgt_text = _span_text(doc, attr)
            tgt_text, tgt_label = _ground_span(tgt_text, gliner_entities, spacy_ent_map, "CONCEPT")
            results.append(
                _triple(
                    src_text,
                    src_label,
                    "DEFINED_AS",
                    tgt_text,
                    tgt_label,
                    sent_text,
                    0.85,
                    "B-copula",
                )
            )
    return results


def _pattern_C(
    sent: Any,
    doc: Any,
    gliner_entities: list[dict],
    spacy_ent_map: dict,
) -> list[dict[str, Any]]:
    """Passive definition: nsubjpass + defined/characterized + prep(as) + pobj."""
    results: list[dict] = []
    root = next((t for t in sent if t.dep_ == "ROOT"), None)
    if root is None:
        # also check passivised verbs deeper in the tree
        root = next((t for t in sent if t.lemma_.lower() in _DEFINITIONAL_LEMMAS), None)
    if root is None:
        return results

    lemma = (root.lemma_ or root.text).lower()
    if lemma not in _DEFINITIONAL_LEMMAS and lemma not in _COPULA_LEMMAS:
        return results

    subjects = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass") and not _is_pronoun(c)]
    as_preps = [c for c in root.children if c.dep_ == "prep" and c.text.lower() in ("as", "to be")]
    if not subjects or not as_preps:
        return results

    sent_text = sent.text.strip()
    for subj in subjects:
        src_text = _span_text(doc, subj)
        src_text, src_label = _ground_span(src_text, gliner_entities, spacy_ent_map, "CONCEPT")
        for prep in as_preps:
            pobjs = [c for c in prep.children if c.dep_ == "pobj"]
            for pobj in pobjs:
                tgt_text = _span_text(doc, pobj)
                tgt_text, tgt_label = _ground_span(
                    tgt_text, gliner_entities, spacy_ent_map, "CONCEPT"
                )
                results.append(
                    _triple(
                        src_text,
                        src_label,
                        "DEFINED_AS",
                        tgt_text,
                        tgt_label,
                        sent_text,
                        0.90,
                        "C-passive-def",
                    )
                )
    return results


def _pattern_D(
    sent: Any,
    doc: Any,
    gliner_entities: list[dict],
    spacy_ent_map: dict,
) -> list[dict[str, Any]]:
    """Appositive: NP head ←appos← appositive NP → EXEMPLIFIES."""
    results: list[dict] = []
    sent_text = sent.text.strip()
    for token in sent:
        if token.dep_ != "appos":
            continue
        head = token.head
        if _is_pronoun(head):
            continue
        head_text = _span_text(doc, head)
        # Exclude the appos span from head's subtree to get cleaner head text
        # Use just the NP root tokens up to the appos
        head_text = head.text.strip()  # just the head token, cleaner
        appos_text = _span_text(doc, token)

        head_text, head_label = _ground_span(head_text, gliner_entities, spacy_ent_map, "ENTITY")
        appos_text, appos_label = _ground_span(appos_text, gliner_entities, spacy_ent_map, "ENTITY")
        results.append(
            _triple(
                head_text,
                head_label,
                "EXEMPLIFIES",
                appos_text,
                appos_label,
                sent_text,
                0.75,
                "D-appos",
            )
        )
    return results


def _pattern_E(
    sent: Any,
    doc: Any,
    gliner_entities: list[dict],
    spacy_ent_map: dict,
) -> list[dict[str, Any]]:
    """Referential: nsubj + refer/denote/express + prep(to/for) + pobj | dobj."""
    results: list[dict] = []
    for token in sent:
        if token.pos_ not in ("VERB",):
            continue
        lemma = (token.lemma_ or token.text).lower()
        if lemma not in ("refer", "denote", "express", "represent", "convey", "presuppose"):
            continue
        relation = COGNITIVE_VERB_MAP.get(lemma, "REFERS_TO")

        subjects = [
            c for c in token.children if c.dep_ in ("nsubj", "nsubjpass") and not _is_pronoun(c)
        ]
        if not subjects:
            continue

        sent_text = sent.text.strip()
        # Try prep(to/for/for) → pobj first, then dobj
        to_preps = [
            c for c in token.children if c.dep_ == "prep" and c.text.lower() in ("to", "for")
        ]
        targets: list[Any] = []
        for prep in to_preps:
            targets.extend(c for c in prep.children if c.dep_ == "pobj")
        if not targets:
            targets = [c for c in token.children if c.dep_ == "dobj"]

        for subj in subjects:
            src_text = _span_text(doc, subj)
            src_text, src_label = _ground_span(src_text, gliner_entities, spacy_ent_map, "CONCEPT")
            for tgt_tok in targets:
                tgt_text = _span_text(doc, tgt_tok)
                tgt_text, tgt_label = _ground_span(
                    tgt_text, gliner_entities, spacy_ent_map, "CONCEPT"
                )
                results.append(
                    _triple(
                        src_text,
                        src_label,
                        relation,
                        tgt_text,
                        tgt_label,
                        sent_text,
                        0.85,
                        "E-referential",
                    )
                )
    return results


def _pattern_F(
    sent: Any,
    doc: Any,
    gliner_entities: list[dict],
    spacy_ent_map: dict,
) -> list[dict[str, Any]]:
    """Distinction: nsubj + distinguish/contrast + dobj + [prep(from) + pobj]."""
    results: list[dict] = []
    for token in sent:
        if token.pos_ not in ("VERB",):
            continue
        lemma = (token.lemma_ or token.text).lower()
        if lemma not in ("distinguish", "contrast", "differentiate", "separate", "compare"):
            continue

        subjects = [
            c for c in token.children if c.dep_ in ("nsubj", "nsubjpass") and not _is_pronoun(c)
        ]
        dobjs = [c for c in token.children if c.dep_ == "dobj"]
        if not subjects or not dobjs:
            continue

        sent_text = sent.text.strip()
        # Also look for "from" prep hanging off the dobj or the verb
        from_pobjs: list[Any] = []
        for dobj_tok in dobjs:
            from_pobjs.extend(
                c2
                for c in dobj_tok.children
                if c.dep_ == "prep" and c.text.lower() == "from"
                for c2 in c.children
                if c2.dep_ == "pobj"
            )
        if not from_pobjs:
            from_pobjs.extend(
                c2
                for c in token.children
                if c.dep_ == "prep" and c.text.lower() == "from"
                for c2 in c.children
                if c2.dep_ == "pobj"
            )

        for subj in subjects:
            src_text = _span_text(doc, subj)
            src_text, src_label = _ground_span(src_text, gliner_entities, spacy_ent_map, "AGENT")
            for dobj_tok in dobjs:
                tgt1_text = _span_text(doc, dobj_tok)
                tgt1_text, tgt1_label = _ground_span(
                    tgt1_text, gliner_entities, spacy_ent_map, "CONCEPT"
                )
                results.append(
                    _triple(
                        src_text,
                        src_label,
                        "DISTINGUISHES",
                        tgt1_text,
                        tgt1_label,
                        sent_text,
                        0.85,
                        "F-distinction",
                    )
                )
            for pobj_tok in from_pobjs:
                tgt2_text = _span_text(doc, pobj_tok)
                tgt2_text, tgt2_label = _ground_span(
                    tgt2_text, gliner_entities, spacy_ent_map, "CONCEPT"
                )
                results.append(
                    _triple(
                        src_text,
                        src_label,
                        "DISTINGUISHES",
                        tgt2_text,
                        tgt2_label,
                        sent_text,
                        0.85,
                        "F-distinction-from",
                    )
                )
    return results


def _pattern_G(
    sent: Any,
    doc: Any,
    gliner_entities: list[dict],
    spacy_ent_map: dict,
) -> list[dict[str, Any]]:
    """Deontic modal AUX → DIRECTS (Stratum IV: directive illocution)."""
    results: list[dict] = []
    for token in sent:
        if token.dep_ != "aux" or token.pos_ != "AUX":
            continue
        lemma = (token.lemma_ or token.text).lower()
        if lemma not in _DEONTIC_MODALS:
            continue
        verb = token.head
        if verb.pos_ not in ("VERB", "AUX"):
            continue
        subjects = [
            c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass") and not _is_pronoun(c)
        ]
        if not subjects:
            continue
        # VP target: from the main verb token to the right edge of its subtree,
        # which excludes the subject and modal (both are left children).
        tgt_text = _norm(doc[verb.i : verb.right_edge.i + 1].text)
        if not tgt_text:
            continue
        sent_text = sent.text.strip()
        negated = _has_negation(verb)
        for subj in subjects:
            src_text = _span_text(doc, subj)
            src_text, src_label = _ground_span(src_text, gliner_entities, spacy_ent_map, "AGENT")
            results.append(
                _triple(
                    src_text,
                    src_label,
                    "DIRECTS",
                    tgt_text,
                    "PROPOSITION",
                    sent_text,
                    0.80,
                    "G-deontic",
                    negated,
                )
            )
    return results


def _pattern_H(
    sent: Any,
    doc: Any,
    gliner_entities: list[dict],
    spacy_ent_map: dict,
) -> list[dict[str, Any]]:
    """Causal advcl (because/since) → EXPLAINS (Stratum V: abductive reasoning)."""
    results: list[dict] = []
    for token in sent:
        if token.dep_ != "advcl":
            continue
        marks = [
            c
            for c in token.children
            if c.dep_ == "mark" and (c.lemma_ or c.text).lower() in _CAUSAL_MARKS
        ]
        if not marks:
            continue
        main_verb = token.head
        main_subjects = [
            c for c in main_verb.children if c.dep_ in ("nsubj", "nsubjpass") and not _is_pronoun(c)
        ]
        if not main_subjects:
            continue
        # Causal clause text, stripped of the leading mark word.
        causal_raw = _span_text(doc, token)
        for mark in marks:
            prefix = mark.text + " "
            if causal_raw.lower().startswith(prefix.lower()):
                causal_raw = causal_raw[len(prefix) :]
        causal_text = _norm(causal_raw)
        sent_text = sent.text.strip()
        for subj in main_subjects:
            tgt_text = _span_text(doc, subj)
            tgt_text, tgt_label = _ground_span(tgt_text, gliner_entities, spacy_ent_map, "ENTITY")
            # "causal_clause EXPLAINS main_subject"  (B explains A)
            results.append(
                _triple(
                    causal_text,
                    "PROPOSITION",
                    "EXPLAINS",
                    tgt_text,
                    tgt_label,
                    sent_text,
                    0.85,
                    "H-causal",
                )
            )
    return results


def _pattern_I(
    sent: Any,
    doc: Any,
    gliner_entities: list[dict],
    spacy_ent_map: dict,
) -> list[dict[str, Any]]:
    """Discourse connective (therefore/thus/hence) → ENTAILS (Stratum V: deductive).

    SpaCy en_core_web parses compound sentences with "therefore/thus/hence" in two ways:
      Case A: "A; therefore B" — B is ROOT, A is ccomp of B, connective is advmod of B.
      Case B: "A, therefore B" — A is ROOT, B is ccomp of A, connective is advmod of B.

    In both cases the connective position in the text splits the sentence: everything
    before the connective is the antecedent (A), everything after is the consequent (B).
    The dep parse is used only to confirm a multi-clause structure exists.
    """
    results: list[dict] = []
    sent_text = sent.text.strip()
    for token in sent:
        lemma = (token.lemma_ or token.text).lower()
        if lemma not in _ENTAILMENT_CONNECTIVES:
            continue
        if token.dep_ != "advmod":
            continue
        consequent_verb = token.head

        # Confirm multi-clause structure: the connective's verb must be ROOT with a
        # ccomp child (Case A), or a ccomp/parataxis/conj child of another verb (Case B).
        has_structure = (
            consequent_verb.dep_ == "ROOT"
            and any(c.dep_ == "ccomp" for c in consequent_verb.children)
        ) or consequent_verb.dep_ in ("ccomp", "parataxis", "conj")
        if not has_structure:
            continue

        # Slice the sentence at the connective boundary.
        sent_start_char = sent[0].idx
        conn_start_char = token.idx
        sent_end_char = sent[-1].idx + len(sent[-1].text)

        ant_text = _norm(doc.text[sent_start_char:conn_start_char]).rstrip(" ;,")
        con_text = _norm(doc.text[conn_start_char + len(token.text) : sent_end_char]).lstrip(" ,")

        if not ant_text or not con_text:
            continue
        results.append(
            _triple(
                ant_text,
                "PROPOSITION",
                "ENTAILS",
                con_text,
                "PROPOSITION",
                sent_text,
                0.75,
                "I-connective",
            )
        )
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_intentional_relations(
    text: str,
    gliner_entities: list[dict[str, Any]],
    spacy_model: str = "en_core_web_lg",
) -> list[dict[str, Any]]:
    """Extract intentionality-typed triples from text using SpaCy dep parsing.

    Args:
        text: Raw chunk text.
        gliner_entities: GLiNER2-extracted entities with 'text' and 'label'/'ontology_label'.
        spacy_model: SpaCy model name (must have dep parser and NER).

    Returns:
        List of typed relation triples (see module docstring for format).
    """
    nlp = get_spacy_nlp(spacy_model)
    doc = nlp(text)
    spacy_ent_map = _build_spacy_ent_map(doc)

    triples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for sent in doc.sents:
        for fn in (
            _pattern_A,
            _pattern_B,
            _pattern_C,
            _pattern_D,
            _pattern_E,
            _pattern_F,
            _pattern_G,
            _pattern_H,
            _pattern_I,
        ):
            for t in fn(sent, doc, gliner_entities, spacy_ent_map):
                key = (_norm(t["source"]).lower(), t["relation"], _norm(t["target"]).lower())
                if key not in seen:
                    seen.add(key)
                    triples.append(t)

    return triples
