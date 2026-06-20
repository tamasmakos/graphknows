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
A  Cognitive verb + complement
B  Copular definition
C  Passive definition
D  Appositive
E  Referential verb + prepositional object
F  Distinction verb
G  Deontic modal obligation (Stratum IV)
H  Causal subordinate clause (Stratum V — abductive)
I  Discourse connective entailment (Stratum V — deductive)

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

from typing import Any

from graphknows.graphgen.extraction._spacy_helpers import (
    _build_spacy_ent_map,
    _norm,
    get_spacy_nlp,
)
from graphknows.graphgen.extraction._spacy_patterns import (
    _pattern_a,
    _pattern_b,
    _pattern_c,
    _pattern_d,
    _pattern_e,
    _pattern_f,
    _pattern_g,
    _pattern_h,
    _pattern_i,
)

__all__ = ["extract_intentional_relations", "get_spacy_nlp"]

_PATTERNS = (
    _pattern_a,
    _pattern_b,
    _pattern_c,
    _pattern_d,
    _pattern_e,
    _pattern_f,
    _pattern_g,
    _pattern_h,
    _pattern_i,
)


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
        for fn in _PATTERNS:
            for t in fn(sent, doc, gliner_entities, spacy_ent_map):
                key = (_norm(t["source"]).lower(), t["relation"], _norm(t["target"]).lower())
                if key not in seen:
                    seen.add(key)
                    triples.append(t)

    return triples
