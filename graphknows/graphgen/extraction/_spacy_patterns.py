"""SpaCy dependency-pattern extractors (patterns a-i) for intentionality-typed triples."""

from __future__ import annotations

import re
from typing import Any

from graphknows.graphgen.extraction._spacy_constants import (
    _CAUSAL_MARKS,
    _COPULA_LEMMAS,
    _DEFINITIONAL_LEMMAS,
    _DEONTIC_MODALS,
    _ENTAILMENT_CONNECTIVES,
    COGNITIVE_VERB_MAP,
)
from graphknows.graphgen.extraction._spacy_helpers import (
    _ground_span,
    _has_negation,
    _is_pronoun,
    _norm,
    _span_text,
    _strip_leading_complementizer,
    _triple,
)


def _pattern_a(
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

            ccomp = next((c for c in token.children if c.dep_ == "ccomp"), None)
            xcomp = next((c for c in token.children if c.dep_ == "xcomp"), None)
            dobj = next((c for c in token.children if c.dep_ == "dobj"), None)

            if ccomp is not None:
                raw = _span_text(doc, ccomp)
                tgt_text = _strip_leading_complementizer(raw)
                results.append(
                    _triple(src_text, src_label, relation, tgt_text, "PROPOSITION",
                            sent_text, 0.90, "A-ccomp", negated)
                )
            elif xcomp is not None:
                raw = _span_text(doc, xcomp)
                tgt_text = re.sub(r"^to\s+", "", raw, flags=re.IGNORECASE)
                results.append(
                    _triple(src_text, src_label, relation, tgt_text, "PROPOSITION",
                            sent_text, 0.85, "A-xcomp", negated)
                )
            elif dobj is not None:
                tgt_text = _span_text(doc, dobj)
                tgt_text, tgt_label = _ground_span(tgt_text, gliner_entities, spacy_ent_map, "ENTITY")
                results.append(
                    _triple(src_text, src_label, relation, tgt_text, tgt_label,
                            sent_text, 0.80, "A-dobj", negated)
                )
    return results


def _pattern_b(
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
                _triple(src_text, src_label, "DEFINED_AS", tgt_text, tgt_label,
                        sent_text, 0.85, "B-copula")
            )
    return results


def _pattern_c(
    sent: Any,
    doc: Any,
    gliner_entities: list[dict],
    spacy_ent_map: dict,
) -> list[dict[str, Any]]:
    """Passive definition: nsubjpass + defined/characterized + prep(as) + pobj."""
    results: list[dict] = []
    root = next((t for t in sent if t.dep_ == "ROOT"), None)
    if root is None:
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
                tgt_text, tgt_label = _ground_span(tgt_text, gliner_entities, spacy_ent_map, "CONCEPT")
                results.append(
                    _triple(src_text, src_label, "DEFINED_AS", tgt_text, tgt_label,
                            sent_text, 0.90, "C-passive-def")
                )
    return results


def _pattern_d(
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
        head_text = head.text.strip()
        appos_text = _span_text(doc, token)

        head_text, head_label = _ground_span(head_text, gliner_entities, spacy_ent_map, "ENTITY")
        appos_text, appos_label = _ground_span(appos_text, gliner_entities, spacy_ent_map, "ENTITY")
        results.append(
            _triple(head_text, head_label, "EXEMPLIFIES", appos_text, appos_label,
                    sent_text, 0.75, "D-appos")
        )
    return results


def _pattern_e(
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
                tgt_text, tgt_label = _ground_span(tgt_text, gliner_entities, spacy_ent_map, "CONCEPT")
                results.append(
                    _triple(src_text, src_label, relation, tgt_text, tgt_label,
                            sent_text, 0.85, "E-referential")
                )
    return results


def _pattern_f(
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
                tgt1_text, tgt1_label = _ground_span(tgt1_text, gliner_entities, spacy_ent_map, "CONCEPT")
                results.append(
                    _triple(src_text, src_label, "DISTINGUISHES", tgt1_text, tgt1_label,
                            sent_text, 0.85, "F-distinction")
                )
            for pobj_tok in from_pobjs:
                tgt2_text = _span_text(doc, pobj_tok)
                tgt2_text, tgt2_label = _ground_span(tgt2_text, gliner_entities, spacy_ent_map, "CONCEPT")
                results.append(
                    _triple(src_text, src_label, "DISTINGUISHES", tgt2_text, tgt2_label,
                            sent_text, 0.85, "F-distinction-from")
                )
    return results


def _pattern_g(
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
        tgt_text = _norm(doc[verb.i : verb.right_edge.i + 1].text)
        if not tgt_text:
            continue
        sent_text = sent.text.strip()
        negated = _has_negation(verb)
        for subj in subjects:
            src_text = _span_text(doc, subj)
            src_text, src_label = _ground_span(src_text, gliner_entities, spacy_ent_map, "AGENT")
            results.append(
                _triple(src_text, src_label, "DIRECTS", tgt_text, "PROPOSITION",
                        sent_text, 0.80, "G-deontic", negated)
            )
    return results


def _pattern_h(
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
            c for c in main_verb.children
            if c.dep_ in ("nsubj", "nsubjpass") and not _is_pronoun(c)
        ]
        if not main_subjects:
            continue
        causal_raw = _span_text(doc, token)
        for mark in marks:
            prefix = mark.text + " "
            if causal_raw.lower().startswith(prefix.lower()):
                causal_raw = causal_raw[len(prefix):]
        causal_text = _norm(causal_raw)
        sent_text = sent.text.strip()
        for subj in main_subjects:
            tgt_text = _span_text(doc, subj)
            tgt_text, tgt_label = _ground_span(tgt_text, gliner_entities, spacy_ent_map, "ENTITY")
            results.append(
                _triple(causal_text, "PROPOSITION", "EXPLAINS", tgt_text, tgt_label,
                        sent_text, 0.85, "H-causal")
            )
    return results


def _pattern_i(
    sent: Any,
    doc: Any,
    gliner_entities: list[dict],
    spacy_ent_map: dict,
) -> list[dict[str, Any]]:
    """Discourse connective (therefore/thus/hence) → ENTAILS (Stratum V: deductive).

    SpaCy en_core_web parses compound sentences with "therefore/thus/hence" in two ways:
      Case A: "A; therefore B" — B is ROOT, A is ccomp of B, connective is advmod of B.
      Case B: "A, therefore B" — A is ROOT, B is ccomp of A, connective is advmod of B.
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

        has_structure = (
            consequent_verb.dep_ == "ROOT"
            and any(c.dep_ == "ccomp" for c in consequent_verb.children)
        ) or consequent_verb.dep_ in ("ccomp", "parataxis", "conj")
        if not has_structure:
            continue

        sent_start_char = sent[0].idx
        conn_start_char = token.idx
        sent_end_char = sent[-1].idx + len(sent[-1].text)

        ant_text = _norm(doc.text[sent_start_char:conn_start_char]).rstrip(" ;,")
        con_text = _norm(doc.text[conn_start_char + len(token.text) : sent_end_char]).lstrip(" ,")

        if not ant_text or not con_text:
            continue
        results.append(
            _triple(ant_text, "PROPOSITION", "ENTAILS", con_text, "PROPOSITION",
                    sent_text, 0.75, "I-connective")
        )
    return results
