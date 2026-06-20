"""SpaCy model loading, text utilities, entity grounding, and triple builder."""

from __future__ import annotations

import logging
import re
from typing import Any

from graphknows.graphgen.extraction._spacy_constants import _SPACY_NER_TO_LABEL

log = logging.getLogger(__name__)

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


def _overlaps(a: str, b: str) -> bool:
    na, nb = a.lower().strip(), b.lower().strip()
    return bool(na and nb and (na in nb or nb in na))


def _ground_span(
    span_text: str,
    gliner_entities: list[dict[str, Any]],
    spacy_ent_map: dict[tuple[int, int], str],
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
    for (_start, _end), ner_type in spacy_ent_map.items():
        span_lower = span_text.lower()
        if _overlaps(span_text, span_lower):
            mapped = _SPACY_NER_TO_LABEL.get(ner_type, default_label)
            return span_text.strip(), mapped

    return span_text.strip(), default_label


def _build_spacy_ent_map(doc: Any) -> dict[tuple[int, int], str]:
    return {(ent.start, ent.end): ent.label_ for ent in doc.ents}


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
