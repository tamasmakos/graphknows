#!/usr/bin/env python3
"""Isolated graphgen subprocess runner.

Called by the agent's graphgen_tool.py via asyncio.create_subprocess_exec.
Reads JSON docs from stdin, writes {nodes, edges, entities} JSON to stdout.
Crashes here don't affect the parent agent process.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Minimal logging — stderr only, JSON to keep it parseable
logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                    format="%(levelname)s %(name)s: %(message)s")

_SERVICES = Path(__file__).resolve().parents[1]
for p in [str(_SERVICES / "agent"), str(_SERVICES), str(_SERVICES / "graphgen")]:
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> None:
    payload = json.load(sys.stdin)
    documents = payload.get("documents", [])
    session_id = payload.get("session_id", "unknown")
    domain = payload.get("ontology_domain", "general")

    if not documents:
        json.dump({"nodes": 0, "edges": 0, "entities": [], "lexical_edges": 0}, sys.stdout)
        return

    from graphgen.pipeline.entity_relation.extraction import get_gliner2_model
    from graphgen.prototype_gliner2_ontology import (
        build_gliner2_schema,
        build_top_level_label_space,
        refine_entities_with_ontology,
        select_candidate_labels,
    )

    top_labels = build_top_level_label_space()
    schema = build_gliner2_schema(list(top_labels.get("ontology", {}).get("children", [])))
    model = get_gliner2_model()

    nodes: list[dict] = []
    edges: list[dict] = []

    for doc in documents[:20]:  # cap at 20 docs to limit RAM
        text = str(doc.get("text", ""))[:4000]
        if not text.strip():
            continue
        try:
            candidate_labels = select_candidate_labels(text, top_labels, top_k=40)
            raw_result = model.extract_entities(text, candidate_labels, threshold=0.4)
            # extract_entities returns {"entities": {"Label": ["name1", ...]}}
            raw_ents_dict = raw_result.get("entities", {}) if isinstance(raw_result, dict) else {}
            # Convert to flat list for refine_entities_with_ontology
            flat_ents = [
                {"text": name, "label": label}
                for label, names in raw_ents_dict.items()
                for name in names
                if name and name.strip()
            ]
            refined = refine_entities_with_ontology(flat_ents, schema)
            for ent in refined:
                label = ent.get("label", "CONCEPT")
                name = ent.get("text", "").strip()
                if not name:
                    continue
                nodes.append({"id": name, "type": label, "doc_id": doc.get("id", "")})
                # Co-occurrence edges within document
            # Simple co-occurrence: connect all entities in this doc
            doc_entities = [n["id"] for n in nodes if n.get("doc_id") == doc.get("id", "")]
            for i, a in enumerate(doc_entities):
                for b in doc_entities[i+1:i+4]:  # up to 3 neighbours
                    edges.append({"source": a, "target": b, "type": "co-occurs"})
        except Exception as exc:
            print(f"WARNING: entity extraction failed for doc {doc.get('id')}: {exc}", file=sys.stderr)

    json.dump({
        "nodes": len(nodes),
        "edges": len(edges),
        "lexical_edges": 0,
        "entities": nodes[:200],
        "relations": edges[:500],
    }, sys.stdout)


if __name__ == "__main__":
    main()
