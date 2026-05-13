"""In-process GLiNER2 + DSPy extraction module.

Eliminates subprocess overhead by loading GLiNER2 once and reusing it
across extraction calls within the same agent process.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# ── Path setup — CRITICAL: package is at services/, not services/graphgen/ ─────
_SERVICES = Path(__file__).resolve().parents[2]
for _p in [str(_SERVICES), str(_SERVICES / "graphgen"), str(_SERVICES / "agent")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Load .env for GROQ_API_KEY ────────────────────────────────────────────────
_ENV = Path(__file__).resolve().parents[3] / ".env"
if _ENV.exists():
    for _line in _ENV.read_text().splitlines():
        _line = _line.strip()
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stderr,
    format="%(levelname)s %(name)s: %(message)s",
)
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


_DEFAULT_LABELS = [
    "Person", "Organization", "Location", "Country", "Institution",
    "Legislative Body", "Policy", "Law", "Regulation", "Directive",
    "Treaty", "Agreement", "Decision", "Action", "Event",
    "Date", "Concept", "Topic", "Product", "Technology",
]


class InProcessExtractor:
    """In-process GLiNER2 entity + DSPy relation extractor.

    Loads GLiNER2 once at construction and reuses it across all extract() calls.
    """

    def __init__(self) -> None:
        t0 = time.time()
        import contextlib

        from graphgen.pipeline.entity_relation.extraction import get_gliner2_model
        from graphgen.prototype_gliner2_ontology import (
            build_gliner2_schema,
            build_top_level_label_space,
            refine_entities_with_ontology,
            select_candidate_labels,
        )

        with contextlib.redirect_stdout(sys.stderr):
            self._gliner = get_gliner2_model()

        top_labels = build_top_level_label_space(_DEFAULT_LABELS)
        _children = list(top_labels.get("ontology", {}).get("children", []))
        if not _children:
            _children = _DEFAULT_LABELS
        self._schema = build_gliner2_schema(_children)
        self._top_labels = top_labels

        # DSPy setup
        self._dspy_ok = False
        self._dspy_module = None
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            try:
                import dspy
                from graphgen.pipeline.entity_relation.dspy_module import GraphExtractorModule
                lm = dspy.LM(
                    "groq/llama-3.3-70b-versatile",
                    api_key=groq_key,
                    max_tokens=1024,
                )
                dspy.configure(lm=lm)
                self._dspy_module = GraphExtractorModule()
                self._dspy_ok = True
                log.warning("DSPy configured with Groq LM")
            except Exception as exc:
                log.warning("DSPy setup failed, will use co-occurrence fallback: %s", exc)
        else:
            log.warning("No GROQ_API_KEY — DSPy relation extraction disabled")

        log.warning("GLiNER2 loaded in %.1fs", time.time() - t0)

    def extract(self, documents: list[dict], session_id: str = "") -> dict:
        """Run GLiNER2 entity + DSPy relation extraction.

        Args:
            documents: List of dicts with keys: id, title, text, source.
            session_id: Session identifier (used for logging context).

        Returns:
            dict with keys: nodes, edges, lexical_edges, entities, relations
            (same schema as graphgen_worker.py).
        """
        from graphgen.prototype_gliner2_ontology import (
            refine_entities_with_ontology,
            select_candidate_labels,
        )

        node_map: dict[str, dict] = {}
        edge_list: list[dict] = []

        for doc in documents[:15]:
            doc_id = doc.get("id", "") or doc.get("url", "")[:60]
            raw_text = str(doc.get("text", "") or doc.get("title", ""))
            if not raw_text.strip():
                continue

            doc_entities: list[str] = []

            for chunk in _chunk_text(raw_text):
                if not chunk.strip():
                    continue
                try:
                    candidate_labels = select_candidate_labels(chunk, self._top_labels, top_k=40)
                    raw_result = self._gliner.extract_entities(
                        chunk, candidate_labels, threshold=0.35
                    )
                    raw_dict = raw_result.get("entities", {}) if isinstance(raw_result, dict) else {}
                    flat_ents = [
                        {"text": name, "label": label}
                        for label, names in raw_dict.items()
                        for name in (names if isinstance(names, list) else [])
                        if name and name.strip()
                    ]
                    refined = refine_entities_with_ontology(flat_ents, self._schema)

                    chunk_ent_names: list[str] = []
                    for ent in refined:
                        label = ent.get("label", "CONCEPT")
                        name = ent.get("text", "").strip()
                        if not name or len(name) < 3:
                            continue
                        node_key = f"{label}::{name.lower()}"
                        if node_key not in node_map:
                            node_map[node_key] = {
                                "id": name,
                                "type": label,
                                "name": name,
                                "doc_ids": [],
                                "count": 0,
                            }
                        node_map[node_key]["count"] += 1
                        if doc_id and doc_id not in node_map[node_key]["doc_ids"]:
                            node_map[node_key]["doc_ids"].append(doc_id)
                        chunk_ent_names.append(name)
                        if name not in doc_entities:
                            doc_entities.append(name)

                    # DSPy relation extraction per chunk
                    if self._dspy_ok and self._dspy_module and len(chunk_ent_names) >= 2:
                        try:
                            ontology_classes = list({
                                ent.get("label", "CONCEPT")
                                for ent in refined
                                if ent.get("text", "").strip()
                            }) or ["PERSON", "ORGANIZATION", "POLICY", "LOCATION", "EVENT", "CONCEPT"]
                            result = self._dspy_module(
                                text=chunk[:1500],
                                ontology_classes=ontology_classes or ["PERSON", "ORGANIZATION", "POLICY", "CONCEPT"],
                                entity_hints=chunk_ent_names[:20],
                            )
                            triplets = getattr(result, "triplets", []) or []
                            for triplet in triplets:
                                if not (triplet.source and triplet.target):
                                    continue
                                if triplet.confidence < 0.4:
                                    continue
                                edge_list.append({
                                    "source": triplet.source,
                                    "target": triplet.target,
                                    "type": triplet.relation or "RELATED_TO",
                                    "evidence": (triplet.evidence or "")[:200],
                                    "confidence": round(triplet.confidence, 3),
                                })
                        except Exception as dspy_exc:
                            log.warning("DSPy extraction failed for chunk: %s", dspy_exc)
                            for i, a in enumerate(chunk_ent_names):
                                for b in chunk_ent_names[i + 1:i + 4]:
                                    edge_list.append({
                                        "source": a, "target": b,
                                        "type": "CO_OCCURS", "evidence": "", "confidence": 0.5,
                                    })
                    elif len(chunk_ent_names) >= 2:
                        for i, a in enumerate(chunk_ent_names):
                            for b in chunk_ent_names[i + 1:i + 4]:
                                edge_list.append({
                                    "source": a, "target": b,
                                    "type": "CO_OCCURS", "evidence": "", "confidence": 0.5,
                                })

                except Exception as chunk_exc:
                    log.warning("Chunk extraction failed: %s", chunk_exc)

        entity_nodes = [
            {"id": v["id"], "type": v["type"], "name": v["name"],
             "doc_id": v["doc_ids"][0] if v["doc_ids"] else "",
             "count": v["count"]}
            for v in node_map.values()
        ]

        seen_edges: set[str] = set()
        deduped_edges: list[dict] = []
        for e in edge_list:
            key = f"{e['source']}\u2192{e['target']}:{e['type']}"
            if key not in seen_edges:
                seen_edges.add(key)
                deduped_edges.append(e)

        return {
            "nodes": len(entity_nodes),
            "edges": len(deduped_edges),
            "lexical_edges": 0,
            "entities": entity_nodes[:300],
            "relations": deduped_edges[:600],
        }


def load_extractor() -> InProcessExtractor | None:
    """Load and return an InProcessExtractor, or None on failure."""
    try:
        return InProcessExtractor()
    except Exception as exc:
        log.warning("GLiNER2 load failed (agent will use subprocess fallback): %s", exc)
        return None
