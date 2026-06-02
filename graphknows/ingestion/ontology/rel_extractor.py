"""DSPy-based ontology-guided relation extractor.

Architecture
============
Three-pass hybrid relation extraction pipeline:

Pass 1 — Doc-type classifier
    Keyword-count heuristic routes each chunk to the most appropriate
    ontology: the graphknows-domain ontology (technical text) or the
    Schema.org general ontology (prose).

Pass 2 — Co-occurrence warm-start
    For every pair of extracted entities in the same sentence, a CO_OCCURS
    candidate triple is generated and passed to the LLM as a warm-start.
    Validated in the three-pass benchmark (2026-05-28): 2-3x recall gain.

Pass 3 — DSPy ChainOfThought extraction
    A typed DSPy Signature sends the text, entity list, ontology schema,
    and candidate triples to the configured LM (default:
    ``deepseek/deepseek-v4-flash`` via OpenRouter, temperature=0.6,
    max_tokens=3000).

Post-processing
    Self-loops dropped.  Triples below confidence threshold filtered.
    (subject, object) pairs deduplicated; highest-confidence triple wins.

Configuration
    Set ``EXTRACTION_MODEL`` in .env (default: ``deepseek/deepseek-v4-flash``).
    If ``OPENROUTER_API_KEY`` is absent the extractor disables itself gracefully.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

# ── Built-in domain ontologies ────────────────────────────────────────────────

DOMAIN_GK_ONTOLOGY: dict[str, str] = {
    "component_of": "software component or module X is part of larger system Y",
    "depends_on": "X requires Y to function; removing Y breaks X",
    "implements": "X is a concrete implementation of abstract interface or pattern Y",
    "wraps": "X provides a thin adapter layer around underlying system Y",
    "stores_in": "agent or pipeline X persists data into storage system Y",
    "reads_from": "agent or pipeline X retrieves data from storage system Y",
    "exposes_tool": "MCP server or service X exposes tool or API endpoint Y",
    "calls_tool": "agent or system X invokes tool, function, or API Y",
    "returns": "function or tool X returns value, object, or result type Y",
    "supersedes": "X makes Y obsolete; X is the current valid fact, Y is tombstoned",
    "valid_at": "fact or relation X was valid at time or date Y",
    "causes": "condition or bug X directly causes failure, error, or symptom Y",
    "fixes": "code change or patch X resolves bug, error, or issue Y",
    "diagnosed_as": "failure or anomaly X is classified under root cause Y",
    "triggers": "event or condition X triggers or initiates process or pipeline Y",
    "evaluated_by": "system or output X is assessed by agent role or method Y",
    "achieves_score": "system or method X achieves metric value or benchmark score Y",
    "outperforms": "approach X produces better results than baseline or alternative Y",
    "contrasts_with": "design X and design Y differ in approach or trade-off",
    "stored_in_stm": "entity or fact X is held in short-term memory store Y",
    "promoted_to_ltm": "entity or session X was flushed and promoted into long-term memory Y",
    "recalled_by": "fact or chunk X is retrieved by recall query Y",
    "entity_type": "extracted entity X is classified under ontology label or type Y",
    "extracted_from": "entity, relation, or chunk X was extracted from source document Y",
    "chunked_into": "document X was split into chunk Y by the ingest pipeline",
    "embedded_as": "text X was encoded and stored as vector embedding in Y",
    "designed_as": "system or component X was intentionally architected as approach Y",
    "replaces": "design decision or component X replaces previous approach Y",
    "validates": "test, assertion, or metric X confirms correctness of component Y",
    "extends": "X adds functionality on top of base class or interface Y",
    "preceded_by": "fact or component X was preceded or replaced by earlier version Y",
}

SCHEMA_ORG_ONTOLOGY: dict[str, str] = {
    "memberOf": "a group or organization that an agent belongs to",
    "member": "a member of an organization, group, or network",
    "founder": "a person who founded or created an organization",
    "employee": "a person who is employed by an organization",
    "affiliation": "an organization that a person is affiliated with",
    "subOrganization": "a division that is part of a larger organization",
    "parentOrganization": "the larger organization that a subsidiary belongs to",
    "knowsAbout": "a topic that an agent has knowledge or expertise about",
    "colleague": "a person who is a professional peer of another",
    "author": "the person or organization who wrote a creative work",
    "creator": "the person or entity who originally created something",
    "contributor": "a person or organization that contributed to a project",
    "publisher": "the organization responsible for publishing a work",
    "mentions": "something that is mentioned or discussed in a work",
    "about": "the subject or main theme of a creative work",
    "citation": "a citation or reference to another publication",
    "isBasedOn": "a work that this work is derived from",
    "hasPart": "a component included within a larger work",
    "isPartOf": "a larger collection that this item belongs to",
    "organizer": "the person or organization who organized an event",
    "performer": "a person or entity who performed at an event",
    "sponsor": "a person or organization that supports an event or person",
    "attendee": "a person who attended or participated in an event",
    "location": "the place or venue where an event took place",
    "result": "the outcome or output produced by an action or event",
    "instrument": "a tool, software, or method used to perform an action",
    "agent": "the person, organization, or agent that performed an action",
    "manufacturer": "the organization that manufactured or produced a product",
    "brand": "the brand or trademark associated with a product",
    "isRelatedTo": "a product, concept, or entity that is related to this one",
    "isSimilarTo": "a product or entity that is similar or comparable",
    "softwareRequirements": "software or libraries required to run this application",
    "applicationCategory": "the category or type that a software application belongs to",
    "teaches": "the skill or competency that a course or resource teaches",
    "assesses": "the skill or competency assessed by this resource",
    "inLanguage": "the natural language a creative work is written in",
    "containsPlace": "a location geographically contained within another place",
    "containedInPlace": "the larger geographic area this place is situated within",
    "category": "the category or classification that an entity belongs to",
    "sameAs": "a URL pointing to an equivalent entity in another KB",
    "subjectOf": "a creative work that this entity is the subject of",
}

# ── Doc-type classifier ───────────────────────────────────────────────────────

_GK_KEYWORDS: set[str] = {
    "memgraph",
    "neo4j",
    "pgvector",
    "graphknows",
    "gliner",
    "memory_ingest",
    "memory_recall",
    "ltm_query",
    "stm",
    "ltm",
    "mcp",
    "session_id",
    "chunk",
    "entity_extraction",
    "relation_extraction",
    "ingest",
    "flush",
    "cosine_score",
    "bi-temporal",
    "tombstone",
    "valid_from",
    "valid_to",
    "is_current",
}


def _classify_doc_type(text: str) -> str:
    """Return 'graphknows' if >= 2 domain keywords found, else 'general'."""
    tl = text.lower()
    hits = sum(1 for kw in _GK_KEYWORDS if kw in tl)
    return "graphknows" if hits >= 2 else "general"


# ── Co-occurrence warm-start ──────────────────────────────────────────────────


def _build_cooccurrence_candidates(text: str, entities: list[str]) -> list[dict[str, Any]]:
    """Emit CO_OCCURS candidates for entity pairs appearing in the same sentence."""
    candidates: list[dict[str, Any]] = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        sent_lower = sent.lower()
        sent_ents = [e for e in entities if e.lower() in sent_lower]
        for i, e1 in enumerate(sent_ents):
            for e2 in sent_ents[i + 1 :]:
                candidates.append(
                    {"subject": e1, "predicate": "CO_OCCURS", "object": e2, "confidence": 0.4}
                )
    return candidates


# ── DSPy Signature ────────────────────────────────────────────────────────────


def _make_extractor() -> Any:
    """Build a DSPy ChainOfThought module with a typed Signature."""
    import dspy  # type: ignore

    class OntologyExtractionSignature(dspy.Signature):  # type: ignore[misc]
        """Extract knowledge graph triples from the source text.

        You are a precise knowledge graph extractor.  Your job is to identify
        typed (subject, predicate, object) triples supported by the text.

        Rules:
        - Use ONLY predicates from the provided ontology_schema.
        - Extract relations that are clearly stated OR strongly implied.
        - Be generous: include all plausible relations supported by the text.
        - Aim for at least one triple per pair of co-occurring named entities.
        - Set confidence >= 0.6 for well-supported relations.
        - Set confidence 0.4-0.6 for implied but plausible relations.
        - NEVER invent facts not grounded in the text.
        - Use the candidate_relations as a warm-start — refine and extend them.
        - Deduplicate: keep all distinct predicates for the same (subject, object).
        - Output ONLY a valid JSON array.  No markdown fences, no preamble.
        - Each element: {"subject": str, "predicate": str, "object": str,
          "confidence": float, "rationale": str}
        """

        text: str = dspy.InputField(  # type: ignore[assignment]
            desc="Source text to extract relations from (<=2000 chars)"
        )
        entities_json: str = dspy.InputField(  # type: ignore[assignment]
            desc="JSON array of entity names found in the text"
        )
        ontology_schema: str = dspy.InputField(  # type: ignore[assignment]
            desc="JSON dict mapping predicate -> description; only these predicates are valid"
        )
        candidate_relations: str = dspy.InputField(  # type: ignore[assignment]
            desc="JSON array of warm-start CO_OCCURS candidates (subject, predicate, object)"
        )
        relations_json: str = dspy.OutputField(  # type: ignore[assignment]
            desc="JSON array of extracted triples: [{subject, predicate, object, confidence, rationale}]"
        )

    return dspy.ChainOfThought(OntologyExtractionSignature)


# ── Main extractor class ──────────────────────────────────────────────────────


class OntologyRelExtractor:
    """Extract typed relations from text guided by an injected relation schema.

    Uses DSPy ChainOfThought with DeepSeek v4 Flash via OpenRouter.
    Built-in domain routing is only used when no explicit relation schema is
    injected.
    Co-occurrence warm-start boosts recall 2-3x vs cold extraction.

    Args:
        lm: Pre-configured DSPy LM instance.  If None, auto-configured from
            ``OPENROUTER_API_KEY`` and ``EXTRACTION_MODEL`` env vars.
        confidence_threshold: Minimum confidence to keep a triple (default 0.5).
        taxonomy_file: Optional taxonomy file or bundled taxonomy short name.
        relation_schema: Optional predicate-to-description mapping.
    """

    def __init__(
        self,
        lm: Any | None = None,
        confidence_threshold: float = 0.5,
        taxonomy_file: str | None = None,
        relation_schema: dict[str, str] | None = None,
    ) -> None:
        self._available: bool = False
        self._extractor: Any = None
        self._threshold = confidence_threshold
        self._relation_schema: dict[str, str] | None = None

        if relation_schema is not None:
            self._relation_schema = relation_schema
        elif taxonomy_file is not None:
            from graphknows.ingestion.domain.taxonomy import load_taxonomy_artifacts

            self._relation_schema = load_taxonomy_artifacts(taxonomy_file).relation_schema

        try:
            import dspy  # type: ignore

            if lm is not None:
                dspy.configure(lm=lm)
                self._extractor = _make_extractor()
                self._available = True
                log.info("OntologyRelExtractor: DSPy ready (external LM)")
            elif os.environ.get("OPENROUTER_API_KEY"):
                model_name = os.environ.get("EXTRACTION_MODEL", "deepseek/deepseek-v4-flash")
                _lm = dspy.LM(
                    model=f"openrouter/{model_name}",
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    temperature=0.6,
                    max_tokens=3000,
                )
                dspy.configure(lm=_lm)
                self._extractor = _make_extractor()
                self._available = True
                log.info("OntologyRelExtractor: DSPy ready (model=%s)", model_name)
            else:
                log.warning("OntologyRelExtractor: OPENROUTER_API_KEY not set — disabled")
        except ImportError:
            log.warning("OntologyRelExtractor: DSPy not installed — disabled")
        except Exception as exc:
            log.warning("OntologyRelExtractor: init failed — %s", exc)

    # ── Routing helpers ───────────────────────────────────────────────────────

    @staticmethod
    def get_domain_ontology(text: str) -> dict[str, str]:
        """Return the appropriate ontology dict for the given text."""
        doc_type = _classify_doc_type(text)
        return DOMAIN_GK_ONTOLOGY if doc_type == "graphknows" else SCHEMA_ORG_ONTOLOGY

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_ontology(
        self, text: str, ontology_schema: str | dict[str, str] | None
    ) -> dict[str, str]:
        """Resolve the ontology to a predicate->description dict."""
        if ontology_schema is None:
            if self._relation_schema:
                return self._relation_schema
            return self.get_domain_ontology(text)
        if isinstance(ontology_schema, dict):
            return ontology_schema
        try:
            parsed = json.loads(ontology_schema)
            if isinstance(parsed, dict) and "properties" in parsed:
                props = parsed["properties"][:40]
                return {p: "" for p in props}
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return self.get_domain_ontology(text)

    @staticmethod
    def _strip_fences(raw: str) -> str:
        """Strip markdown code fences and skip to first JSON structure."""
        raw = raw.strip()
        for fence in ("```json", "```"):
            if raw.startswith(fence):
                raw = raw[len(fence) :]
                break
        raw = raw.removesuffix("```").strip()
        if raw and raw[0] not in ("[", "{"):
            for ch in ("[", "{"):
                idx = raw.find(ch)
                if idx >= 0:
                    raw = raw[idx:]
                    break
        return raw

    def _validate_relations(self, relations: list[Any], threshold: float) -> list[dict[str, Any]]:
        """Validate, filter, and deduplicate a raw relations list."""
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        for r in relations:
            if not isinstance(r, dict):
                continue
            subj = str(r.get("subject", "")).strip()
            pred_name = str(r.get("predicate", "")).strip()
            obj = str(r.get("object", "")).strip()
            if not subj or not pred_name or not obj:
                continue
            if subj.lower() == obj.lower():
                continue
            conf = float(r.get("confidence", 0.5))
            if conf < threshold:
                continue
            key = (subj.lower(), obj.lower())
            if key not in seen or seen[key]["confidence"] < conf:
                seen[key] = {
                    "subject": subj,
                    "predicate": pred_name,
                    "object": obj,
                    "confidence": conf,
                    "rationale": str(r.get("rationale", ""))[:300],
                }
        return list(seen.values())

    # ── Main extraction entry point ───────────────────────────────────────────

    def extract(
        self,
        text: str,
        entities: list[dict[str, Any]],
        ontology_schema: str | dict[str, str] | None = None,
        ontology_id: str = "auto",
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Extract ontology-typed relations from text using DSPy + DeepSeek.

        Args:
            text: Source document text (first 2000 chars used).
            entities: List of entity dicts with at minimum a ``name`` key.
            ontology_schema: Optional override.  May be a JSON string (legacy
                API compat), a ``dict[predicate, description]`` mapping, or None
                for auto-routing by doc type.
            ontology_id: Identifier for logging.
            confidence_threshold: Override instance-level threshold.

        Returns:
            List of dicts: ``{subject, predicate, object, confidence, rationale}``.
            Empty list on failure or when extractor is unavailable.
        """
        if not self._available or self._extractor is None:
            return []
        if not entities:
            return []

        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else getattr(self, "_threshold", 0.5)
        )
        ont_dict = self._resolve_ontology(text, ontology_schema)
        schema_compact = json.dumps(ont_dict, ensure_ascii=False)[:3000]

        entity_names = [e.get("name", "") for e in entities if e.get("name")]
        if not entity_names:
            return []

        candidates = _build_cooccurrence_candidates(text[:2000], entity_names[:30])
        try:
            pred = self._extractor(
                text=text[:2000],
                entities_json=json.dumps(entity_names[:30]),
                ontology_schema=schema_compact,
                candidate_relations=json.dumps(candidates[:40]),
            )
        except Exception as exc:
            log.warning("OntologyRelExtractor DSPy call failed [%s]: %s", ontology_id, exc)
            return []

        raw = self._strip_fences(getattr(pred, "relations_json", "") or "")
        try:
            relations = json.loads(raw)
        except json.JSONDecodeError:
            log.debug(
                "OntologyRelExtractor [%s]: JSON parse failed — raw=%r",
                ontology_id,
                raw[:200],
            )
            return []

        if not isinstance(relations, list):
            return []

        validated = self._validate_relations(relations, threshold)
        log.info(
            "OntologyRelExtractor [%s]: %d triples from %d entities (doc_type=%s)",
            ontology_id,
            len(validated),
            len(entity_names),
            _classify_doc_type(text),
        )
        return validated
