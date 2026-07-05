"""Dynamic spaCy-hinted GLiNER2 extractor for the memory ingestion pipeline.

spaCy runs once per chunk and mines the ENTITY schema GLiNER2 receives for that
same chunk (entity labels come from spaCy NER). RELATION labels, by contrast,
come from the configured taxonomy's canonical predicate vocabulary
(schema-guided relation extraction), falling back to mined dependency-path
triggers only when no taxonomy relation schema is available.
"""

from __future__ import annotations

import contextlib
import logging
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from graphknows.config.adapters.gliner_model import load_gliner2_model
from graphknows.extraction.relations.object_cleaner import clean_relation_object

log = logging.getLogger(__name__)

_BASE_THRESHOLD: float = 0.35
_SPACY_MODEL = "en_core_web_lg"
_MIN_ENTITY_LABEL_COUNT = 1
_MIN_RELATION_TRIGGER_COUNT = 1
_MAX_ENTITY_LABELS = 24
_MAX_RELATION_LABELS = 64
_MAX_RELATION_FRAMES = 4
_TRIGGER_POS = frozenset({"VERB", "NOUN"})


@dataclass
class ExtractionResult:
    """Structured result from GLiNER2 extraction."""

    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _EntityHint:
    text: str
    label: str
    root: Any
    sent: Any


@dataclass(frozen=True)
class _DynamicSchema:
    entity_spec: dict[str, dict[str, Any]]
    relation_spec: dict[str, str]
    label_to_type: dict[str, str]


def _default_relation_schema() -> dict[str, str]:
    """Load the configured taxonomy's relation schema (predicate -> description).

    Soft-fails to an empty dict (miner-derived relations) if the taxonomy or its
    descriptions are unavailable, so extraction never breaks on a config issue.
    """
    try:
        from graphknows.ingestion.domain.taxonomy import (
            build_relation_schema,
            get_default_taxonomy,
        )

        return build_relation_schema(get_default_taxonomy())
    except Exception as exc:  # pragma: no cover - config guard
        log.warning("taxonomy relation schema unavailable, using mined lemmas: %s", exc)
        return {}


def _relationship_terms() -> frozenset[str]:
    """Kinship / social-role words from the taxonomy — never valid fact objects.

    Empty on config failure (object cleaning simply skips the role check).
    """
    try:
        from graphknows.ingestion.domain.taxonomy import get_default_taxonomy

        group = get_default_taxonomy().entity_groups.get("RELATIONSHIP")
        return frozenset(a.lower() for a in (group.aliases if group else []))
    except Exception as exc:  # pragma: no cover - config guard
        log.warning("relationship-role vocabulary unavailable: %s", exc)
        return frozenset()


def _schema_label(value: str) -> str:
    """Return a GLiNER-friendly label derived only from observed text metadata."""
    label = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return label or "entity"


def _dependency_path(left: Any, right: Any) -> list[Any]:
    """Return the shortest undirected dependency path between two spaCy tokens."""
    left_path = [left, *left.ancestors]
    right_path = [right, *right.ancestors]
    right_positions = {token.i: idx for idx, token in enumerate(right_path)}
    for left_idx, token in enumerate(left_path):
        right_idx = right_positions.get(token.i)
        if right_idx is None:
            continue
        return left_path[:left_idx] + right_path[right_idx::-1]
    return []


def _iter_entity_pairs(entities: list[_EntityHint]) -> Iterable[tuple[_EntityHint, _EntityHint]]:
    for idx, left in enumerate(entities):
        for right in entities[idx + 1 :]:
            if left.text.casefold() != right.text.casefold():
                yield left, right


def _is_ancestor(ancestor: Any, token: Any) -> bool:
    return any(item.i == ancestor.i for item in token.ancestors)


def _relation_trigger_candidates(path: list[Any]) -> list[Any]:
    """Return broad lexical relation triggers from a dependency path.

    Prefer content-bearing predicates. Bare prepositions/adjectives create labels
    that are too narrow for GLiNER2, and high-level verbs that dominate another
    candidate on the same path tend to express reporting/framing rather than the
    entity-to-entity relation itself.
    """
    candidates = [
        token
        for token in path
        if token.pos_ in _TRIGGER_POS and (token.lemma_ or token.text).strip()
    ]
    return [
        token
        for token in candidates
        if not any(other.i != token.i and _is_ancestor(token, other) for other in candidates)
    ]


def _token_text(token: Any) -> str:
    return str(getattr(token, "text", "") or "").strip()


def _path_text(path: list[Any]) -> str:
    parts = [_token_text(token) for token in path]
    return " -> ".join(part for part in parts if part)


def _relation_frame(left: _EntityHint, trigger: Any, right: _EntityHint, path: list[Any]) -> str:
    trigger_lemma = (trigger.lemma_ or trigger.text).strip()
    frame = f"{left.text} -> {trigger_lemma} -> {right.text}"
    path_hint = _path_text(path)
    if path_hint:
        frame = f"{frame} (dependency path: {path_hint})"
    return frame


def _relation_description(lemma: str, pos: str, frames: list[str]) -> str:
    description = (
        f"A relation between two concise entity or noun-phrase arguments expressed by "
        f"the observed {pos.lower()} trigger '{lemma}' or a close paraphrase. "
        "Prefer direct subject, object, complement, or prepositional-object arguments; "
        "avoid selecting whole clauses as relation heads or tails."
    )
    if frames:
        joined = "; ".join(frames[:_MAX_RELATION_FRAMES])
        description = f"{description} Observed dependency frames in this chunk: {joined}."
    return description


class _SpacySchemaMiner:
    """Mine GLiNER2 schema labels from spaCy NER and dependency structure."""

    def __init__(
        self,
        model_name: str = _SPACY_MODEL,
        nlp_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._nlp_loader = nlp_loader
        self._nlp: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._nlp is not None:
            return
        loader = self._nlp_loader
        if loader is None:
            import spacy

            loader = spacy.load
        try:
            self._nlp = loader(self._model_name)
        except Exception as exc:
            raise RuntimeError(
                f"spaCy model '{self._model_name}' is required for GLiNER2 schemas. "
                f"Install it with:  python -m spacy download {self._model_name}"
            ) from exc

    @property
    def nlp(self) -> Any:
        """The loaded spaCy pipeline (loads on first access)."""
        self._ensure_loaded()
        return self._nlp

    def build(self, text: str) -> _DynamicSchema:
        self._ensure_loaded()
        assert self._nlp is not None

        doc = self._nlp(text)
        hints = [
            _EntityHint(
                text=ent.text.strip(),
                label=ent.label_.strip(),
                root=ent.root,
                sent=ent.sent,
            )
            for ent in doc.ents
            if ent.text.strip() and ent.label_.strip()
        ]

        entity_counts = Counter(hint.label for hint in hints)
        label_to_type: dict[str, str] = {}
        entity_spec: dict[str, dict[str, Any]] = {}
        for label, count in entity_counts.most_common(_MAX_ENTITY_LABELS):
            if count < _MIN_ENTITY_LABEL_COUNT:
                continue
            schema_name = _schema_label(label)
            label_to_type[schema_name] = label
            entity_spec[schema_name] = {
                "description": (f"Named entity with the label {label}, observed in this chunk."),
                "threshold": _BASE_THRESHOLD,
            }

        trigger_counts: Counter[tuple[str, str]] = Counter()
        trigger_frames: dict[tuple[str, str], list[str]] = {}
        by_sentence: dict[int, list[_EntityHint]] = {}
        for hint in hints:
            by_sentence.setdefault(hint.sent.start, []).append(hint)

        for sent_entities in by_sentence.values():
            for left, right in _iter_entity_pairs(sent_entities):
                path = _dependency_path(left.root, right.root)
                for token in _relation_trigger_candidates(path):
                    lemma = (token.lemma_ or token.text).strip()
                    key = (lemma.casefold(), token.pos_)
                    trigger_counts[key] += 1
                    frame = _relation_frame(left, token, right, path)
                    frames = trigger_frames.setdefault(key, [])
                    if frame not in frames and len(frames) < _MAX_RELATION_FRAMES:
                        frames.append(frame)

        relation_spec: dict[str, str] = {}
        for (lemma, pos), count in trigger_counts.most_common(_MAX_RELATION_LABELS):
            if count < _MIN_RELATION_TRIGGER_COUNT:
                continue
            relation_name = _schema_label(lemma)
            relation_spec[relation_name] = _relation_description(
                lemma=lemma,
                pos=pos,
                frames=trigger_frames.get((lemma, pos), []),
            )

        return _DynamicSchema(
            entity_spec=entity_spec,
            relation_spec=relation_spec,
            label_to_type=label_to_type,
        )


class GLiNER2EntityExtractor:
    """GLiNER2 extractor with dynamic spaCy-derived schemas.

    The GLiNER2 model is loaded lazily once.  The schema is rebuilt for every
    chunk from spaCy signals, so entity and relation labels adapt to the input
    text without taxonomy files or domain-specific constants.
    """

    def __init__(
        self,
        model_loader: Callable[[], Any] = load_gliner2_model,
        schema_miner: _SpacySchemaMiner | None = None,
        spacy_model: str = "",
    ) -> None:
        self._gliner: Any | None = None
        self._model_loader = model_loader
        if not spacy_model:
            from graphknows.config import GraphKnowsSettings

            spacy_model = GraphKnowsSettings().spacy_model
        self._schema_miner = schema_miner or _SpacySchemaMiner(model_name=spacy_model)
        # Relation schema comes from the configured taxonomy (conversational
        # predicates + descriptions), not per-chunk mined lemmas — schema-guided
        # relation extraction against a stable, canonical vocabulary. Entities
        # stay dynamic (spaCy-mined). Empty schema -> fall back to mined lemmas.
        self._relation_schema: dict[str, str] = _default_relation_schema()
        self._relationship_terms: frozenset[str] = _relationship_terms()

    def _ensure_model_loaded(self) -> None:
        """Load GLiNER2 model on first extraction call."""
        if self._gliner is not None:
            return

        log.info("GLiNER2 model not loaded; loading now (first call, expect 15-30s)...")
        start = time.monotonic()
        with contextlib.redirect_stdout(sys.stderr):
            model = self._model_loader()

        if model is None:
            raise RuntimeError(
                "GLiNER2 model failed to load; check RAM (~2 GB required) and "
                "HuggingFace access. No silent fallback."
            )

        self._gliner = model
        log.info("GLiNER2 model loaded in %.1fs", time.monotonic() - start)

    def _build_gliner_schema(self, dynamic_schema: _DynamicSchema) -> Any:
        assert self._gliner is not None
        schema = self._gliner.create_schema()
        if dynamic_schema.entity_spec:
            schema.entities(dynamic_schema.entity_spec)
        relation_spec = self._relation_schema or dynamic_schema.relation_spec
        if relation_spec:
            schema.relations(relation_spec)
        return schema

    def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relations using a spaCy-mined GLiNER2 schema."""
        self._ensure_model_loaded()
        assert self._gliner is not None

        dynamic_schema = self._schema_miner.build(text)
        if not dynamic_schema.entity_spec:
            return ExtractionResult()

        gliner_schema = self._build_gliner_schema(dynamic_schema)
        raw = self._gliner.extract(
            text,
            gliner_schema,
            _BASE_THRESHOLD,
            True,  # format_results
            True,  # include_confidence
            True,  # include_spans
        )

        entities: list[dict[str, Any]] = []
        for raw_label, spans in (raw.get("entities") or {}).items():
            schema_label = _schema_label(raw_label)
            canonical = dynamic_schema.label_to_type.get(schema_label, raw_label.upper())
            for span in spans or []:
                conf = float(span.get("confidence", 0.0))
                if conf < _BASE_THRESHOLD:
                    continue
                name = span.get("text", "").strip()
                if not name:
                    continue
                entities.append({"name": name, "type": canonical, "score": conf})

        relations: list[dict[str, Any]] = []
        for rel_type, triplets in (raw.get("relation_extraction") or {}).items():
            for triplet in triplets or []:
                head = triplet.get("head", {})
                tail = triplet.get("tail", {})
                src = head.get("text", "").strip()
                tgt = tail.get("text", "").strip()
                if not src or not tgt or src.lower() == tgt.lower():
                    continue  # drop empty and self-relations (X -rel-> X is noise)
                # Canonicalize the object to a content noun; drop clause /
                # question-echo / wrong-type objects at extraction.
                tgt = clean_relation_object(tgt, self._schema_miner.nlp, self._relationship_terms)
                if not tgt or src.lower() == tgt.lower():
                    continue
                score = min(
                    float(head.get("confidence", 0.0)),
                    float(tail.get("confidence", 0.0)),
                )
                src_type = next((e["type"] for e in entities if e["name"] == src), "ENTITY")
                tgt_type = next((e["type"] for e in entities if e["name"] == tgt), "ENTITY")
                relations.append(
                    {
                        "head": src,
                        "head_type": src_type,
                        "relation": rel_type,
                        "tail": tgt,
                        "tail_type": tgt_type,
                        "score": score,
                    }
                )

        return ExtractionResult(entities=entities, relations=relations)
