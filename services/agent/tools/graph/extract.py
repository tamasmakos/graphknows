"""GraphGen tool — runs NER/KG pipeline on fetched documents, writes to session graph."""
from __future__ import annotations

import asyncio
import logging
import sys
from collections import Counter
import inspect
from pathlib import Path
from typing import Any

import networkx as nx

from agent.tools.base import BaseTool
from agent.session.manager import SessionGraphManager

_SERVICES_DIR = Path(__file__).resolve().parents[2]
if str(_SERVICES_DIR) not in sys.path:
    sys.path.append(str(_SERVICES_DIR))

from graphgen.pipeline.entity_relation.extraction import get_gliner2_model
from graphgen.pipeline.entity_relation.ontology import (
    build_gliner2_schema,
    build_top_level_label_space,
    refine_entities_with_ontology,
    select_candidate_labels,
)


log = logging.getLogger("agent.tools.graph.extract")

_DEFAULT_LABELS = [
    "Person", "Organization", "Location", "Country", "Institution",
    "Legislative Body", "Policy", "Law", "Regulation", "Directive",
    "Treaty", "Agreement", "Decision", "Action", "Event",
    "Date", "Concept", "Topic", "Product", "Technology",
]


class GraphGenTool(BaseTool):
    """Extract entities/relations and persist into session graph manager."""

    name = "graphgen"
    description = (
        "Extract entities and relations from fetched documents and update the session graph. "
        'Params: {"documents": [...], "session_id": "...", "ontology_domain": "debates|legislation|finance|general"}'
    )

    def __init__(self, sgm: SessionGraphManager, ws_broadcaster=None) -> None:
        self._sgm = sgm
        self._ws_broadcaster = ws_broadcaster

    async def run(self, params: dict) -> dict:
        try:
            session_id = str(params.get("session_id", "")).strip()
            documents = params.get("documents") or []
            if not session_id:
                return {"error": "session_id is required", "nodes": 0, "edges": 0, "lexical_edges": 0, "entities": []}
            if not isinstance(documents, list) or not documents:
                return {"error": "documents must be a non-empty list", "nodes": 0, "edges": 0, "lexical_edges": 0, "entities": []}

            domain = params.get("ontology_domain") or self._detect_domain(documents)
            docs_entities = await asyncio.gather(
                *[self._extract_document_entities(doc, domain) for doc in documents],
                return_exceptions=True,
            )

            graph = nx.DiGraph()
            doc_to_entities: dict[str, list[dict[str, Any]]] = {}
            all_entities: list[dict[str, Any]] = []

            for idx, result in enumerate(docs_entities):
                if isinstance(result, Exception):
                    log.warning("Graph extraction failed for document %s: %s", idx, result)
                    continue
                doc = documents[idx]
                doc_id = str(doc.get("id", f"doc-{idx}"))
                entities = result
                doc_to_entities[doc_id] = entities
                all_entities.extend(entities)

            capped_entities = self._cap_entities(all_entities, max_nodes=500)

            kept_keys = {
                (e["doc_id"], e["name"], e["label"])
                for e in capped_entities
            }
            by_doc: dict[str, list[dict[str, Any]]] = {}
            for e in capped_entities:
                by_doc.setdefault(e["doc_id"], []).append(e)
                graph.add_node(
                    e["name"],
                    id=e["name"],
                    label=e["label"],
                    source_doc=e["doc_id"],
                    session_id=session_id,
                    confidence=e["confidence"],
                )

            rel_edges = 0
            for doc_id, entities in doc_to_entities.items():
                kept = [e for e in entities if (e["doc_id"], e["name"], e["label"]) in kept_keys]
                for i, left in enumerate(kept):
                    for right in kept[i + 1 :]:
                        if left["name"] == right["name"]:
                            continue
                        relation_type = "CO_OCCURS"
                        weight = (left["confidence"] + right["confidence"]) / 2.0
                        graph.add_edge(left["name"], right["name"], type=relation_type, weight=weight, source_doc=doc_id)
                        rel_edges += 1

            lexical_edges = 0
            lexical_edge_list: list[dict[str, Any]] = []
            doc_ids = list(by_doc.keys())
            for i, doc_a in enumerate(doc_ids):
                ents_a = {e["name"] for e in by_doc.get(doc_a, [])}
                for doc_b in doc_ids[i + 1 :]:
                    ents_b = {e["name"] for e in by_doc.get(doc_b, [])}
                    shared = ents_a & ents_b
                    if len(shared) >= 2:
                        src = f"DOC::{doc_a}"
                        tgt = f"DOC::{doc_b}"
                        graph.add_node(src, id=src, label="Document", source_doc=doc_a, session_id=session_id)
                        graph.add_node(tgt, id=tgt, label="Document", source_doc=doc_b, session_id=session_id)
                        graph.add_edge(src, tgt, type="DOC_SIMILAR", weight=float(len(shared)))
                        lexical_edge_list.append(
                            {
                                "from": src,
                                "to": tgt,
                                "type": "DOC_SIMILAR",
                                "weight": float(len(shared)),
                                "shared_entities": sorted(shared),
                            }
                        )
                        lexical_edges += 1

            for _, data in graph.nodes(data=True):
                node_id = data.get("id") or ""
                self._sgm.add_node(
                    session_id=session_id,
                    node_id=node_id,
                    label=data.get("label", "Entity"),
                    properties={
                        "name": data.get("id", node_id),
                        "source_doc": data.get("source_doc", ""),
                        "session_id": session_id,
                        "confidence": data.get("confidence", 0.0),
                        "ontology_class": data.get("label", "Entity"),
                    },
                )

            edge_count = 0
            for source, target, data in graph.edges(data=True):
                self._sgm.add_edge(
                    session_id=session_id,
                    source_name=source,
                    target_name=target,
                    edge_type=str(data.get("type", "RELATION")),
                    properties={"weight": float(data.get("weight", 0.5))},
                )
                edge_count += 1

            top_entities = [name for name, _ in Counter([e["name"] for e in capped_entities]).most_common(20)]
            node_list = [
                {
                    "id": data.get("id") or "",
                    "label": data.get("label", "Entity"),
                    "name": data.get("id") or "",
                }
                for _, data in graph.nodes(data=True)
            ]
            edge_list = [
                {
                    "from": source,
                    "to": target,
                    "type": str(data.get("type", "RELATION")),
                }
                for source, target, data in graph.edges(data=True)
            ]

            if self._ws_broadcaster:
                payload = {
                    "nodes": node_list,
                    "edges": edge_list,
                    "lexical_edges": lexical_edge_list,
                }
                maybe_coro = self._ws_broadcaster(session_id, payload)
                if inspect.isawaitable(maybe_coro):
                    await maybe_coro

            return {
                "nodes": graph.number_of_nodes(),
                "edges": edge_count,
                "lexical_edges": lexical_edges,
                "lexical_edge_list": lexical_edge_list,
                "entities": top_entities,
                "ontology_domain": domain,
                "node_list": node_list,
                "edge_list": edge_list,
            }
        except Exception as exc:
            log.exception("graphgen tool failed")
            return {"error": str(exc), "nodes": 0, "edges": 0, "lexical_edges": 0, "entities": []}

    async def _extract_document_entities(self, document: dict[str, Any], domain: str) -> list[dict[str, Any]]:
        text = str(document.get("text", "") or "")
        if not text.strip():
            return []

        label_space = build_top_level_label_space(_DEFAULT_LABELS)
        candidate_labels = self._domain_labels(domain, label_space, text)
        _ = build_gliner2_schema(candidate_labels)
        model = get_gliner2_model({"extraction": {"ner_backend": "gliner2", "gliner_threshold": 0.45}})

        result = await asyncio.to_thread(
            model.extract_entities,
            text,
            candidate_labels,
            0.45,
            True,
            True,
            True,
        )

        flat: list[dict[str, Any]] = []
        for label, spans in (result.get("entities") or {}).items():
            for span in spans:
                flat.append(
                    {
                        "text": span.get("text", "").strip(),
                        "label": label,
                        "ontology_label": label,
                        "confidence": float(span.get("confidence", 0.0) or 0.0),
                    }
                )

        refined = refine_entities_with_ontology(flat, label_space)
        doc_id = str(document.get("id", ""))
        entities: list[dict[str, Any]] = []
        for item in refined:
            name = (item.get("text") or "").strip()
            if not name:
                continue
            entities.append(
                {
                    "name": name,
                    "label": str(item.get("ontology_label") or item.get("label") or "Entity"),
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                    "doc_id": doc_id,
                }
            )
        return entities

    def _detect_domain(self, documents: list[dict[str, Any]]) -> str:
        sources = {str(doc.get("source", "")).lower() for doc in documents}
        if "ep_debates" in sources:
            return "debates"
        if "eurlex" in sources or "cellar_sparql" in sources or "ep_resolutions" in sources:
            return "legislation"
        if sources & {"ecb_publications", "council_conclusions", "eca_reports"}:
            return "finance"
        return "general"

    def _domain_labels(self, domain: str, label_space: dict[str, dict[str, Any]], text: str) -> list[str]:
        base = select_candidate_labels(text, label_space, top_k=40)
        if domain == "legislation":
            bias = ["Directive", "Regulation", "Treaty", "Legal Document", "Decision"]
        elif domain == "debates":
            bias = ["Person", "Legislative Body", "Organization", "Action"]
        elif domain == "finance":
            bias = ["Organization", "Transaction", "Obligation", "Decision"]
        else:
            bias = []
        merged = []
        for label in [*bias, *base]:
            if label not in merged:
                merged.append(label)
        return merged[:40]

    def _cap_entities(self, entities: list[dict[str, Any]], max_nodes: int) -> list[dict[str, Any]]:
        sorted_entities = sorted(entities, key=lambda e: float(e.get("confidence", 0.0)), reverse=True)
        return sorted_entities[:max_nodes]
