#!/usr/bin/env python3
"""Isolated graphgen subprocess runner — GLiNER2 + DSPy pipeline.

Called by the agent's graphgen tool via asyncio.create_subprocess_exec.
Reads JSON docs from stdin, writes {nodes, edges, entities, relations} JSON to stdout.

Pipeline (from wiki pipeline-stages.md):
  text → GLiNER2 entity detection (ontology-guided) → DSPy relation extraction → graph assembly
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── Path setup — CRITICAL: package is at services/, not services/graphgen/ ─────
_SERVICES = Path(__file__).resolve().parents[2]  # /root/graphknows/services
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
log = logging.getLogger("graphgen_worker")

# ── Chunking helper ───────────────────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _chunk_text(text: str) -> list[str]:
    """Chunk text at CHUNK_SIZE with CHUNK_OVERLAP (matches wiki pipeline-stages.md)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break
    return chunks[:10]  # cap at 10 chunks per doc


# ── Main extraction ───────────────────────────────────────────────────────────

def main() -> None:
    payload: dict[str, Any] = json.load(sys.stdin)
    documents: list[dict] = payload.get("documents", [])
    session_id: str = payload.get("session_id", "unknown")

    if not documents:
        json.dump(
            {"nodes": 0, "edges": 0, "lexical_edges": 0, "entities": [], "relations": []},
            sys.stdout,
        )
        return

    # ── 1. Load GLiNER2 ───────────────────────────────────────────────────────
    try:
        t0 = time.time()
        from graphgen.pipeline.entity_relation.extraction import get_gliner2_model
        from graphgen.prototype_gliner2_ontology import (
            build_gliner2_schema,
            build_top_level_label_space,
            refine_entities_with_ontology,
            select_candidate_labels,
        )
        # GLiNER prints a banner to stdout which corrupts the JSON protocol.
        # Redirect stdout→stderr during model load, then restore.
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            gliner_model = get_gliner2_model()
        # Default label set covering EU-policy / general NER domains.
        # build_top_level_label_space([]) returns an empty dict (no ontology nodes),
        # so we seed it with a broad label set that the grouping rules can map.
        _DEFAULT_LABELS = [
            "Person", "Organization", "Location", "Country", "Institution",
            "Legislative Body", "Policy", "Law", "Regulation", "Directive",
            "Treaty", "Agreement", "Decision", "Action", "Event",
            "Date", "Concept", "Topic", "Product", "Technology",
        ]
        top_labels = build_top_level_label_space(_DEFAULT_LABELS)
        # If grouping produces nothing, fall back to the raw labels directly.
        _children = list(top_labels.get("ontology", {}).get("children", []))
        if not _children:
            _children = _DEFAULT_LABELS
        schema = build_gliner2_schema(_children)
        log.warning("GLiNER2 loaded in %.1fs", time.time() - t0)
    except Exception as exc:
        log.error("GLiNER2 load failed: %s", exc)
        json.dump(
            {"nodes": 0, "edges": 0, "lexical_edges": 0, "entities": [], "relations": [],
             "error": f"GLiNER2 load failed: {exc}"},
            sys.stdout,
        )
        return

    # ── 2. Configure DSPy with Groq ───────────────────────────────────────────
    groq_key = os.environ.get("GROQ_API_KEY", "")
    dspy_ok = False
    dspy_module = None
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
            dspy_module = GraphExtractorModule()
            dspy_ok = True
            log.warning("DSPy configured with Groq LM")
        except Exception as exc:
            log.warning("DSPy setup failed, will use co-occurrence fallback: %s", exc)
    else:
        log.warning("No GROQ_API_KEY — DSPy relation extraction disabled")

    # ── 3. Process documents ──────────────────────────────────────────────────
    node_map: dict[str, dict] = {}  # name → {id, type, name, doc_ids, count}
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
                candidate_labels = select_candidate_labels(chunk, top_labels, top_k=40)
                raw_result = gliner_model.extract_entities(
                    chunk, candidate_labels, threshold=0.35
                )
                raw_dict = raw_result.get("entities", {}) if isinstance(raw_result, dict) else {}
                flat_ents = [
                    {"text": name, "label": label}
                    for label, names in raw_dict.items()
                    for name in (names if isinstance(names, list) else [])
                    if name and name.strip()
                ]
                refined = refine_entities_with_ontology(flat_ents, schema)

                chunk_ent_names: list[str] = []
                for ent in refined:
                    label = ent.get("label", "CONCEPT")
                    name = ent.get("text", "").strip()
                    if not name or len(name) < 2:
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

                # ── DSPy relation extraction per chunk ────────────────────────
                if dspy_ok and dspy_module and len(chunk_ent_names) >= 2:
                    try:
                        ontology_classes = list({
                            node_map[f"{ent.get('label','CONCEPT')}::{ent.get('text','').lower()}"]["type"]
                            for ent in refined
                            if ent.get("text", "").strip()
                        })
                        result = dspy_module(
                            text=chunk[:1500],
                            ontology_classes=ontology_classes or ["PERSON", "ORGANIZATION", "POLICY", "CONCEPT"],
                            entity_hints=chunk_ent_names[:20],
                        )
                        triplets = getattr(result, "triplets", []) or []
                        for triplet in triplets:
                            if not (triplet.source and triplet.target):
                                continue
                            if triplet.confidence < 0.3:
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
                        # Fallback: co-occurrence edges for this chunk
                        for i, a in enumerate(chunk_ent_names):
                            for b in chunk_ent_names[i + 1:i + 4]:
                                edge_list.append({
                                    "source": a, "target": b,
                                    "type": "CO_OCCURS", "evidence": "", "confidence": 0.5,
                                })
                elif len(chunk_ent_names) >= 2:
                    # No DSPy — co-occurrence within chunk
                    for i, a in enumerate(chunk_ent_names):
                        for b in chunk_ent_names[i + 1:i + 4]:
                            edge_list.append({
                                "source": a, "target": b,
                                "type": "CO_OCCURS", "evidence": "", "confidence": 0.5,
                            })

            except Exception as chunk_exc:
                log.warning("Chunk extraction failed: %s", chunk_exc)

    # ── 4. Build final node list ──────────────────────────────────────────────
    entity_nodes = [
        {"id": v["id"], "type": v["type"], "name": v["name"],
         "doc_id": v["doc_ids"][0] if v["doc_ids"] else "",
         "count": v["count"]}
        for v in node_map.values()
    ]

    # Deduplicate edges
    seen_edges: set[str] = set()
    deduped_edges: list[dict] = []
    for e in edge_list:
        key = f"{e['source']}→{e['target']}:{e['type']}"
        if key not in seen_edges:
            seen_edges.add(key)
            deduped_edges.append(e)

    json.dump(
        {
            "nodes": len(entity_nodes),
            "edges": len(deduped_edges),
            "lexical_edges": 0,
            "entities": entity_nodes[:300],
            "relations": deduped_edges[:600],
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
