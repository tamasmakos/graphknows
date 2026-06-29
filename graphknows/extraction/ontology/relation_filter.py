"""Ontology-driven relation filter — post-extraction curation.

Drops extracted relations whose label is not part of the configured ontology's
property labels. Matching is case- and punctuation-insensitive so that surface
forms like ``member_of``, ``MEMBEROF`` and ``member of`` all map to the ontology
property ``memberOf``.

An empty allowlist disables the filter (pass-through), so extraction modes that
do not constrain relations behave as before.

Extract-then-map
~~~~~~~~~~~~~~~~
:class:`OntologyPredicateMapper` extends the basic filter with an embedding
similarity step: when a free-form predicate does not match any controlled
vocabulary entry by exact normalised comparison, it is mapped to the closest
ontology property using cosine similarity of character-n-gram TF-IDF vectors
(no external embedding service required).  Predicates that score below
``min_similarity`` are dropped as unmappable.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

log = logging.getLogger("graphknows.extraction.ontology.relation_filter")

_NON_ALNUM = re.compile(r"[^a-z0-9]")

# ---------------------------------------------------------------------------
# Default mapping threshold
# ---------------------------------------------------------------------------

_DEFAULT_MIN_SIMILARITY: float = 0.25
"""Cosine similarity threshold below which a predicate is considered unmappable.

Chosen empirically: surface aliases like "is friends with" → friend_of score
well above this; genuinely unrelated predicates (e.g. "filed_complaint") score
below it when the ontology is personal/conversational.
"""


def _normalize(label: str) -> str:
    """Lower-case and strip all non-alphanumeric characters for comparison."""
    return _NON_ALNUM.sub("", str(label).lower())


# ---------------------------------------------------------------------------
# Lightweight character-n-gram similarity (no heavy ML deps)
# ---------------------------------------------------------------------------


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    """Return all character n-grams of length *n* from *text*."""
    padded = f"_{text}_"
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def _tfidf_vector(text: str, idf: dict[str, float], n: int = 3) -> dict[str, float]:
    """Build a TF-IDF weighted character-n-gram vector for *text*."""
    tokens = _char_ngrams(text, n)
    if not tokens:
        return {}
    tf = Counter(tokens)
    total = len(tokens)
    return {tok: (count / total) * idf.get(tok, 1.0) for tok, count in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not a or not b:
        return 0.0
    dot = sum(a.get(tok, 0.0) * val for tok, val in b.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_idf(corpus: list[str], n: int = 3) -> dict[str, float]:
    """Compute IDF weights from a corpus of strings (character n-grams)."""
    N = len(corpus)
    if N == 0:
        return {}
    df: Counter[str] = Counter()
    for doc in corpus:
        df.update(set(_char_ngrams(doc, n)))
    return {tok: math.log(N / (count + 1)) + 1.0 for tok, count in df.items()}


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------


class RelationOntologyFilter:
    """Keep only relations whose label matches an allowed ontology property.

    Args:
        allowed_labels: Ontology property labels. An empty list yields an
            inactive (pass-through) filter.
    """

    def __init__(self, allowed_labels: list[str]) -> None:
        self._allowed = {_normalize(label) for label in allowed_labels if _normalize(label)}

    @property
    def active(self) -> bool:
        """Whether the filter constrains anything (False = pass-through)."""
        return bool(self._allowed)

    def is_allowed(self, label: str) -> bool:
        """Return True if *label* maps to an allowed property (or filter inactive)."""
        if not self.active:
            return True
        return _normalize(label) in self._allowed

    def filter(self, relations: list[dict[str, Any]], label_key: str) -> list[dict[str, Any]]:
        """Return only relations whose ``label_key`` value is allowed."""
        if not self.active:
            return relations
        return [r for r in relations if self.is_allowed(r.get(label_key, ""))]


# ---------------------------------------------------------------------------
# Extract-then-map: free-form predicate → controlled vocabulary
# ---------------------------------------------------------------------------


_SURFACE_ALIASES: dict[str, str] = {
    # Common natural-language surface forms → normalised ontology property names
    # This table is matched before character-n-gram similarity so that semantically
    # close but lexically distant pairs are correctly resolved.
    "is friends with": "friend_of",
    "is a friend of": "friend_of",
    "are friends": "friend_of",
    "is related to": "family_of",
    "is family of": "family_of",
    "is the partner of": "partner_of",
    "is married to": "partner_of",
    "is in a relationship with": "partner_of",
    "belongs to": "member_of",
    "is a member of": "member_of",
    "is part of": "member_of",
    "works for": "works_as",
    "is employed by": "works_as",
    "is employed at": "works_as",
    "works at": "works_as",
    "resides in": "lives_in",
    "lives at": "lives_in",
    "moved to": "lives_in",
    "is located in": "located_in",
    "took place in": "located_in",
    "happened at": "happened_on",
    "occurred on": "happened_on",
    "was held on": "happened_on",
    "went to": "attended",
    "participated in": "attended",
    "took part in": "attended",
    "has": "owns",
    "possesses": "owns",
    "enjoys": "likes",
    "loves": "likes",
    "hates": "dislikes",
    "is planning to": "plans_to",
    "intends to": "plans_to",
    "wants to": "plans_to",
}
"""Surface alias table for common natural-language predicate phrases.

Keys are normalised (lowercase, alphanumeric) forms of surface phrases; values
are the normalised ontology property names they should map to.  This table is
consulted *before* character-n-gram similarity to handle semantically similar
but lexically distant pairs.
"""


class OntologyPredicateMapper:
    """Map free-form LLM predicates to the closest controlled-vocabulary property.

    The mapping is a three-stage pipeline:
    1. **Exact match** (normalised, case/punctuation-insensitive).
    2. **Surface alias lookup** — matches common natural-language phrases
       (e.g. "belongs to" → ``member_of``) without any model calls.
    3. **Character-n-gram TF-IDF cosine similarity** against all ontology
       property labels.  Purely local — no network calls.

    Args:
        ontology_labels: Controlled-vocabulary property labels from the ontology.
        min_similarity: Cosine similarity threshold below which a predicate is
            considered unmappable and dropped.  Defaults to ``_DEFAULT_MIN_SIMILARITY``.
        ngram_size: Character n-gram size for similarity vectors.
    """

    def __init__(
        self,
        ontology_labels: list[str],
        min_similarity: float = _DEFAULT_MIN_SIMILARITY,
        ngram_size: int = 3,
    ) -> None:
        self._labels = list(ontology_labels)
        self._norm_to_label: dict[str, str] = {_normalize(lb): lb for lb in ontology_labels}
        self._min_similarity = min_similarity
        self._n = ngram_size

        # Build a normalised-alias → label lookup restricted to the provided
        # ontology labels so aliases for labels not in this ontology are ignored.
        # Note: alias target values in _SURFACE_ALIASES are normalised forms
        # (underscore-separated, like "member_of") — normalise them before lookup.
        norm_label_set = set(self._norm_to_label.keys())
        self._alias_to_label: dict[str, str] = {}
        for surface, target in _SURFACE_ALIASES.items():
            target_norm = _normalize(target)
            if target_norm in norm_label_set:
                norm_surface = _normalize(surface)
                # The canonical label for target_norm (preserves original casing)
                self._alias_to_label[norm_surface] = self._norm_to_label[target_norm]

        # Pre-compute TF-IDF vectors for each ontology label using label text
        # treated as a bag of character n-grams.  We use the normalised forms
        # (underscores/spaces stripped) as input so "friend_of" == "friend of".
        label_texts = [_normalize(lb) for lb in self._labels]
        self._idf = _build_idf(label_texts, ngram_size)
        self._label_vectors: list[dict[str, float]] = [
            _tfidf_vector(t, self._idf, ngram_size) for t in label_texts
        ]

    @property
    def ontology_labels(self) -> list[str]:
        """Return the controlled-vocabulary property labels."""
        return list(self._labels)

    def map_predicate(self, predicate: str) -> tuple[str | None, float]:
        """Map a free-form predicate to the closest ontology property.

        Returns:
            ``(canonical_label, similarity_score)`` where ``canonical_label``
            is ``None`` if no property meets the ``min_similarity`` threshold.
        """
        if not self._labels:
            return None, 0.0

        norm = _normalize(predicate)

        # Stage 1: exact normalised match (score = 1.0)
        if norm in self._norm_to_label:
            return self._norm_to_label[norm], 1.0

        # Stage 2: surface alias lookup (score = 0.95 to distinguish from exact)
        if norm in self._alias_to_label:
            canonical = self._alias_to_label[norm]
            log.debug(
                "Predicate '%s' alias-mapped → '%s' (score=0.95)", predicate, canonical
            )
            return canonical, 0.95

        # Stage 3: character-n-gram TF-IDF cosine similarity
        query_vec = _tfidf_vector(norm, self._idf, self._n)
        best_score = 0.0
        best_label: str | None = None
        for label, vec in zip(self._labels, self._label_vectors, strict=False):
            score = _cosine(query_vec, vec)
            if score > best_score:
                best_score = score
                best_label = label

        if best_score >= self._min_similarity:
            log.debug(
                "Predicate '%s' mapped → '%s' (similarity=%.3f)", predicate, best_label, best_score
            )
            return best_label, best_score

        log.debug(
            "Predicate '%s' unmappable (best_score=%.3f < threshold=%.3f)",
            predicate,
            best_score,
            self._min_similarity,
        )
        return None, best_score

    def map_relations(
        self,
        relations: list[dict[str, Any]],
        label_key: str = "relation",
    ) -> list[dict[str, Any]]:
        """Map all relations in *relations* and drop unmappable ones.

        For each relation dict, the ``label_key`` field is replaced with the
        canonical ontology property name when a mapping is found.  Relations
        that cannot be mapped (score below threshold) are dropped.

        A ``_mapped_from`` key is added when the predicate was remapped (not
        an exact match) so callers can audit the mapping decisions.
        """
        result: list[dict[str, Any]] = []
        for rel in relations:
            raw_pred = rel.get(label_key, "")
            canonical, score = self.map_predicate(raw_pred)
            if canonical is None:
                log.debug("Dropping relation with unmappable predicate '%s'", raw_pred)
                continue
            mapped = dict(rel)
            mapped[label_key] = canonical
            if canonical != raw_pred:
                mapped["_mapped_from"] = raw_pred
                mapped["_map_score"] = round(score, 4)
            result.append(mapped)
        return result


def build_predicate_mapper(
    ontology_id: str = "personal",
    min_similarity: float = _DEFAULT_MIN_SIMILARITY,
) -> OntologyPredicateMapper:
    """Build an :class:`OntologyPredicateMapper` from a bundled ontology.

    Falls back to an empty mapper (no constraints) when the ontology cannot be
    loaded, so the pipeline never fails due to a missing ontology asset.
    """
    from graphknows.extraction.ontology.loader import OntologyLoader

    try:
        manifest = OntologyLoader().load(ontology_id)
        return OntologyPredicateMapper(manifest.property_labels, min_similarity=min_similarity)
    except (KeyError, FileNotFoundError) as exc:
        log.warning(
            "Predicate mapper inactive — ontology '%s' unavailable: %s", ontology_id, exc
        )
        return OntologyPredicateMapper([], min_similarity=min_similarity)


def build_relation_filter(ontology_id: str = "schema-org") -> RelationOntologyFilter:
    """Build a relation filter from a bundled ontology's property labels.

    Falls back to an inactive (pass-through) filter when the ontology cannot be
    loaded, so ingestion never fails because of a missing ontology asset.
    """
    from graphknows.extraction.ontology.loader import OntologyLoader

    try:
        manifest = OntologyLoader().load(ontology_id)
    except (KeyError, FileNotFoundError) as exc:
        log.warning("Relation filter inactive — ontology '%s' unavailable: %s", ontology_id, exc)
        return RelationOntologyFilter([])
    return RelationOntologyFilter(manifest.property_labels)
