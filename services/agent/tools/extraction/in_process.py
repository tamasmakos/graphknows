"""In-process GLiNER2 + DSPy extraction module.

Eliminates subprocess overhead by loading GLiNER2 once and reusing it
across extraction calls within the same agent process.
"""

from __future__ import annotations

import logging
import os
import sys
import time

log = logging.getLogger("extractor")

# ── Chunking helper ───────────────────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break
    return chunks[:10]


_TRAILING_DIGIT_RE = __import__("re").compile(r"\d+$")

# Short words that indicate a truncated span when they appear at the end
_NOISE_LAST_WORDS = frozenset(
    {
        "pub",
        "publ",
        "impl",
        "spec",
        "dept",  # truncations
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",  # prepositions / articles
        "and",
        "or",
        "but",
        "for",
    }
)

# Single title-cased words that are semantically too generic to be named entities
_GENERIC_SINGLE_WORDS = frozenset(
    {
        "digital",
        "central",
        "bank",
        "currency",
        "monetary",
        "financial",
        "economic",
        "political",
        "legal",
        "national",
        "european",
        "global",
        "federal",
        "regional",
        "official",
        "public",
        "private",
        "social",
        "report",
        "document",
        "proposal",
        "reform",
        "review",
        "article",
        "section",
        "chapter",
        "point",
        "paragraph",
        "clause",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "wikipedia",
        "source",
        "reference",
        "note",
        "see",
        "ibid",
    }
)


def _is_quality_span(name: str) -> bool:
    """Return True if the entity span is likely a real named entity.

    Rejects:
    - Spans ending with bare digits  (e.g. "report1", "section2")
    - Single lowercase words         (e.g. "ward", "moderniz")
    - Single generic English words   (e.g. "Digital", "Central", "Wikipedia")
    Accepts:
    - All-caps acronyms of 2+ chars  (e.g. "ECB", "CBDC", "EU")
    - Any multi-word phrase          (e.g. "digital euro", "central bank")
    - Single words starting uppercase that are not generic (e.g. "Eurozone", "Bundestag")
    """
    if not name or len(name) < 2:
        return False
    # All-caps acronym of 2+ chars (EU, UK, ECB, CBDC …)
    if name.isupper() and len(name) >= 2:
        return True
    if len(name) < 3:
        return False
    if _TRAILING_DIGIT_RE.search(name):
        return False
    words = name.split()
    if len(words) == 1:
        if name.isupper():  # acronym
            return True
        if name[0].islower():  # bare lowercase word — not a proper entity
            return False
        # Single title-cased or mixed-case word: reject if it is a generic term
        if name.lower() in _GENERIC_SINGLE_WORDS:
            return False
    else:
        # Reject spans ending with a known noise/truncation word
        if words[-1].lower() in _NOISE_LAST_WORDS:
            return False
    return True


class InProcessExtractor:
    """In-process GLiNER2 entity + DSPy relation extractor.

    Loads GLiNER2 once at construction and reuses it across all extract() calls.
    All entity types, descriptions, confidence thresholds, and relation types
    are read from the domain taxonomy (taxonomy.eu-politics.yaml); no labels or
    type names are hardcoded in this module.
    """

    def __init__(self) -> None:
        t0 = time.time()
        import contextlib

        from graphgen.config.taxonomy import get_default_taxonomy
        from graphgen.pipeline.entity_relation.extraction import get_gliner2_model

        with contextlib.redirect_stdout(sys.stderr):
            self._gliner = get_gliner2_model()

        # ── Load taxonomy — single source of truth for all labels ──────────
        self._taxonomy = get_default_taxonomy()

        # Build the GLiNER2Schema object from taxonomy descriptions + thresholds
        _base_threshold = 0.35
        entity_spec: dict[str, dict] = {
            label.lower(): {
                "description": grp.description.strip() or label.lower().replace("_", " "),
                "threshold": max(grp.min_ner_confidence, _base_threshold),
            }
            for label, grp in self._taxonomy.entity_groups.items()
            if grp.description.strip()
        }
        self._gliner_schema = self._gliner.create_schema()
        self._gliner_schema.entities(entity_spec)

        # Add taxonomy relation types to the schema for joint extraction
        rel_descs: dict[str, str] = {
            rt: rt.replace("_", " ").lower() for rt in self._taxonomy.relation_types
        }
        if rel_descs:
            self._gliner_schema.relations(rel_descs)

        # child label → canonical top-level group name mapping
        self._child_to_group: dict[str, str] = {}
        for group_label, grp in self._taxonomy.entity_groups.items():
            self._child_to_group[group_label.lower()] = group_label
            for child in grp.children:
                self._child_to_group[child.lower()] = group_label

        # Allowed relation type set (for DSPy relation normalisation)
        self._allowed_relation_set: frozenset[str] = frozenset(
            rt.upper() for rt in self._taxonomy.relation_types
        )
        # List form used as DSPy input hint
        self._allowed_relation_types: list[str] = list(self._taxonomy.relation_types)

        # ── DSPy setup ─────────────────────────────────────────────────────
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not openrouter_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set — DSPy relation extraction requires it"
            )
        import dspy
        from graphgen.pipeline.entity_relation.dspy_module import GraphExtractorModule

        lm = dspy.LM(
            "openrouter/google/gemini-2.0-flash-001",
            api_key=openrouter_key,
            max_tokens=1024,
        )
        dspy.configure(lm=lm)
        self._dspy_module = GraphExtractorModule()
        self._dspy_ok = True
        log.info("DSPy configured with OpenRouter LM")

        log.info("GLiNER2 loaded in %.1fs (taxonomy=%s)", time.time() - t0, self._taxonomy.name)

    def extract(self, documents: list[dict], session_id: str = "") -> dict:
        """Run GLiNER2 entity + DSPy relation extraction.

        Args:
            documents: List of dicts with keys: id, title, text, source.
            session_id: Session identifier (used for logging context).

        Returns:
            dict with keys: nodes, edges, lexical_edges, entities, relations
            (same schema as graphgen_worker.py).
        """
        node_map: dict[str, dict] = {}
        edge_list: list[dict] = []

        log.info(
            "extraction START  session=%s  docs=%d  dspy_ok=%s",
            session_id,
            len(documents),
            self._dspy_ok,
        )
        for doc_idx, doc in enumerate(documents[:15]):
            doc_id = doc.get("id", "") or doc.get("url", "")[:60]
            raw_text = str(doc.get("text", "") or doc.get("title", ""))
            if not raw_text.strip():
                continue

            log.debug(
                "  doc[%d/%d]  id=%r  chars=%d",
                doc_idx + 1,
                min(len(documents), 15),
                doc_id[:60],
                len(raw_text),
            )
            doc_entities: list[str] = []

            for chunk_idx, chunk in enumerate(_chunk_text(raw_text)):
                if not chunk.strip():
                    continue
                try:
                    # \u2500\u2500 Schema-guided extraction (taxonomy descriptions + thresholds) \u2500\u2500\u2500\u2500\u2500\u2500\u2500
                    _base_threshold = 0.35
                    raw = self._gliner.extract(
                        chunk, self._gliner_schema, _base_threshold, True, True, True
                    )

                    # Map raw labels \u2192 canonical group names; apply per-group confidence gate
                    flat_ents: list[dict] = []
                    for raw_label, spans in (raw.get("entities") or {}).items():
                        canonical = self._child_to_group.get(raw_label.lower(), raw_label.upper())
                        min_conf = self._taxonomy.get_min_ner_confidence(canonical)
                        eff_threshold = max(min_conf, _base_threshold)
                        for span in spans or []:
                            conf = float(span.get("confidence", 0.0))
                            if conf < eff_threshold:
                                continue
                            flat_ents.append(
                                {
                                    "text": span.get("text", "").strip(),
                                    "label": canonical,
                                    "ontology_label": canonical,
                                    "confidence": conf,
                                }
                            )

                    # GLiNER2 joint relation extraction candidates
                    gliner_rels: list[dict] = []
                    for rel_type, triplets in (raw.get("relation_extraction") or {}).items():
                        for triplet in triplets or []:
                            head = triplet.get("head") or {}
                            tail = triplet.get("tail") or {}
                            src = (head.get("text") or "").strip()
                            tgt = (tail.get("text") or "").strip()
                            if not src or not tgt or src == tgt:
                                continue
                            rel_conf = min(
                                float(head.get("confidence", 0.0)),
                                float(tail.get("confidence", 0.0)),
                            )
                            gliner_rels.append(
                                {
                                    "source": src,
                                    "relation": rel_type.upper(),
                                    "target": tgt,
                                    "confidence": rel_conf,
                                }
                            )

                    log.debug(
                        "    chunk[%d]  GLiNER2: %d spans (after conf gate)  %d relations",
                        chunk_idx + 1,
                        len(flat_ents),
                        len(gliner_rels),
                    )

                    # \u2500\u2500 Build chunk entity list with per-entity confidence tracking \u2500\u2500\u2500\u2500\u2500\u2500\u2500
                    chunk_ent_names: list[str] = []
                    chunk_ent_conf: dict[str, float] = {}  # name \u2192 max confidence
                    for ent in flat_ents:
                        label = ent.get("label", "ENTITY")
                        name = ent.get("text", "").strip()
                        conf = ent.get("confidence", 0.0)
                        if not name or len(name) < 3:
                            continue
                        if not _is_quality_span(name):
                            continue
                        node_key = f"{label}::{name.lower()}"
                        if node_key not in node_map:
                            node_map[node_key] = {
                                "id": name,
                                "type": label,
                                "name": name,
                                "doc_ids": [],
                                "count": 0,
                                "confidence": conf,
                            }
                        node_map[node_key]["count"] += 1
                        node_map[node_key]["confidence"] = max(
                            node_map[node_key]["confidence"], conf
                        )
                        if doc_id and doc_id not in node_map[node_key]["doc_ids"]:
                            node_map[node_key]["doc_ids"].append(doc_id)
                        if name not in chunk_ent_conf or conf > chunk_ent_conf[name]:
                            chunk_ent_conf[name] = conf
                        if name not in chunk_ent_names:
                            chunk_ent_names.append(name)
                        if name not in doc_entities:
                            doc_entities.append(name)

                    # High-confidence entities only: used for CO_OCCURS fallback
                    # (min 0.5 to avoid pairing low-quality partial spans)
                    high_conf_ents = [
                        n for n in chunk_ent_names if chunk_ent_conf.get(n, 0.0) >= 0.5
                    ]

                    # \u2500\u2500 DSPy relation extraction per chunk \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
                    if self._dspy_ok and self._dspy_module and len(chunk_ent_names) >= 2:
                        try:
                            ontology_classes = list(
                                {
                                    ent.get("label", "ENTITY")
                                    for ent in flat_ents
                                    if ent.get("text", "").strip()
                                }
                            ) or list(self._taxonomy.entity_groups.keys())
                            # Seed DSPy with GLiNER2 joint-extraction candidates +
                            # a small windowed CO_OCCURS hint from high-confidence entities
                            candidate_relations = gliner_rels[:20] + [
                                {
                                    "source": high_conf_ents[i],
                                    "relation": "CO_OCCURS",
                                    "target": high_conf_ents[j],
                                }
                                for i in range(len(high_conf_ents))
                                for j in range(i + 1, min(i + 4, len(high_conf_ents)))
                            ]
                            result = self._dspy_module(
                                text=chunk[:1500],
                                ontology_classes=ontology_classes,
                                entity_hints=chunk_ent_names[:20],
                                allowed_relation_types=self._allowed_relation_types,
                                candidate_relations=candidate_relations[:30],
                            )
                            triplets = getattr(result, "triplets", []) or []
                            accepted = 0
                            for triplet in triplets:
                                if not (triplet.source and triplet.target):
                                    continue
                                if triplet.confidence < 0.4:
                                    continue
                                relation = (triplet.relation or "RELATED_TO").upper()
                                if relation not in self._allowed_relation_set:
                                    log.debug(
                                        "    normalising unknown relation %r \u2192 RELATED_TO",
                                        relation,
                                    )
                                    relation = "RELATED_TO"
                                edge_list.append(
                                    {
                                        "source": triplet.source,
                                        "target": triplet.target,
                                        "type": relation,
                                        "evidence": (triplet.evidence or "")[:200],
                                        "confidence": round(triplet.confidence, 3),
                                    }
                                )
                                accepted += 1
                            log.debug(
                                "    chunk[%d]  DSPy: %d/%d triplets accepted",
                                chunk_idx + 1,
                                accepted,
                                len(triplets),
                            )
                        except Exception:
                            raise
                    else:
                        log.debug(
                            "    chunk[%d]  too few entities (%d) — skipping DSPy",
                            chunk_idx + 1,
                            len(chunk_ent_names),
                        )

                except Exception:
                    raise

        entity_nodes = [
            {
                "id": v["id"],
                "type": v["type"],
                "name": v["name"],
                "doc_id": v["doc_ids"][0] if v["doc_ids"] else "",
                "count": v["count"],
            }
            for v in node_map.values()
        ]

        # Back-fill stub nodes for DSPy-enhanced entities not extracted by GLiNER2.
        # DSPy may legitimately surface entity names that GLiNER2 missed — keep them.
        known_names_lower: set[str] = {v["id"].lower() for v in entity_nodes}
        dspy_stubs = 0
        for e in edge_list:
            for name in (e.get("source", ""), e.get("target", "")):
                name = name.strip()
                if not name or name.lower() in known_names_lower:
                    continue
                if not _is_quality_span(name):
                    continue
                entity_nodes.append(
                    {
                        "id": name,
                        "type": "Entity",
                        "name": name,
                        "doc_id": "",
                        "count": 0,
                    }
                )
                known_names_lower.add(name.lower())
                dspy_stubs += 1
        if dspy_stubs:
            log.debug("  DSPy back-fill: +%d stub nodes from relation endpoints", dspy_stubs)

        seen_edges: set[str] = set()
        deduped_edges: list[dict] = []
        for e in edge_list:
            key = f"{e['source']}\u2192{e['target']}:{e['type']}"
            if key not in seen_edges:
                seen_edges.add(key)
                deduped_edges.append(e)

        log.info(
            "extraction DONE  session=%s  docs=%d  entities=%d  raw_edges=%d  deduped_edges=%d",
            session_id,
            len(documents),
            len(entity_nodes),
            len(edge_list),
            len(deduped_edges),
        )
        return {
            "nodes": len(entity_nodes),
            "edges": len(deduped_edges),
            "lexical_edges": 0,
            "entities": entity_nodes[:300],
            "relations": deduped_edges[:600],
        }


def load_extractor() -> InProcessExtractor:
    """Load and return an InProcessExtractor. Raises on failure."""
    return InProcessExtractor()
