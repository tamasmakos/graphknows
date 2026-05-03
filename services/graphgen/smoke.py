import argparse
import asyncio
import json
import logging
import os
import re
from datetime import date, datetime
from typing import Any, Dict, List, Tuple

import networkx as nx

from graphgen.config.settings import PipelineSettings
from graphgen.data_types import PipelineContext
from graphgen.pipeline.entity_relation.extraction import enrich_graph_per_segment
from graphgen.pipeline.lexical_graph_building.builder import build_lexical_graph
from graphgen.utils.schema_utils import save_graph_schema
from graphgen.utils.utils import create_output_directory, standardize_label

logger = logging.getLogger(__name__)


class LocalSmokeExtractor:
    """Small local extractor for smoke tests with no remote model calls."""

    PERSON_PATTERN = re.compile(
        r"\b([A-Z][a-z]+(?:\s+(?:von|van|de|der|del|la|le|da|di|du))*(?:\s+[A-Z][a-z]+)+)\b"
    )
    ORG_PATTERN = re.compile(
        r"\b(European Union|European Commission|European Parliament|Council of Europe)\b"
    )
    PLACE_PATTERN = re.compile(r"\b(Brussels|Europe|France|Germany|Hungary|Ukraine|Russia)\b")
    KEYWORD_PATTERN = re.compile(r"\b([a-z]{4,}(?:\s+[a-z]{4,})?)\b")
    STOPWORDS = {
        "with", "that", "this", "from", "have", "will", "into", "their", "there",
        "about", "after", "before", "under", "discuss", "discussed", "policy", "europe"
    }

    async def extract_relations(
        self,
        text: str,
        custom_prompt=None,
        keywords: List[str] = None,
        entities: List[str] = None,
        abstract_concepts: List[str] = None,
    ) -> Tuple[List[Tuple[str, str, str, Dict[str, Any]]], List[Dict[str, Any]]]:
        del custom_prompt, keywords, entities, abstract_concepts

        candidates: List[Tuple[str, str]] = []
        seen = set()
        for pattern, label in (
            (self.PERSON_PATTERN, "PERSON"),
            (self.ORG_PATTERN, "ORGANIZATION"),
            (self.PLACE_PATTERN, "LOCATION"),
        ):
            for match in pattern.findall(text):
                name = standardize_label(match)
                if name and name not in seen:
                    seen.add(name)
                    candidates.append((name, label))

        if len(candidates) < 2:
            for match in self.KEYWORD_PATTERN.findall(text.lower()):
                token = standardize_label(match)
                if not token or token.lower() in self.STOPWORDS or token in seen:
                    continue
                seen.add(token)
                candidates.append((token, "CONCEPT"))
                if len(candidates) >= 3:
                    break

        nodes = [{"id": node_id, "type": node_type, "properties": {}} for node_id, node_type in candidates]

        relations: List[Tuple[str, str, str, Dict[str, Any]]] = []
        for idx, (source, _) in enumerate(candidates):
            for target, _ in candidates[idx + 1:]:
                relations.append(
                    (
                        source,
                        "CO_OCCURS_WITH",
                        target,
                        {"confidence": 1.0, "evidence": text[:200]},
                    )
                )

        return relations, nodes


def build_smoke_settings(input_dir: str, output_dir: str, max_documents: int = 1) -> PipelineSettings:
    """Create a minimal local-only configuration for a fast smoke test."""
    config = {
        "test_mode": {"enabled": True, "max_documents": max_documents},
        "infra": {
            "graph_db_type": "none",
            "input_dir": input_dir,
            "output_dir": output_dir,
            "clean_start": False,
        },
        "llm": {
            "base_model": "local-smoke",
            "extraction_model": "local-smoke",
            "summarization_model": "local-smoke",
            "temperature": 0.0,
        },
        "extraction": {
            "backend": "smoke",
            "device": "cpu",
            "use_onnx": False,
            "chunk_size": 256,
            "chunk_overlap": 32,
            "max_concurrent_chunks": 1,
            "entity_labels": ["Person", "Organization", "Location", "Concept"],
            "ontology": {"enabled": False},
            "file_pattern": "*.txt",
        },
        "processing": {
            "enable_pruning": False,
            "pruning_threshold": 0.01,
            "prune_isolated_nodes": False,
            "min_component_size": 1,
            "similarity_threshold": 0.95,
        },
        "embedding": {"model_name": "all-MiniLM-L6-v2", "batch_size": 4, "device": "cpu"},
        "analysis": {
            "enabled": False,
            "topic_separation_test": False,
            "save_provenance": False,
            "save_sampling_manifest": False,
            "save_checkpoints": False,
            "save_topic_separation_inputs": False,
            "save_silhouette_samples": False,
            "save_anova_diagnostics": False,
            "save_manova_details": False,
            "save_raw_overlap_matrix": False,
            "visualization": {"interactive": False, "heatmap": False},
        },
        "analytics": {
            "enabled": False,
            "topic_separation_test": False,
            "save_provenance": False,
            "save_sampling_manifest": False,
            "save_checkpoints": False,
            "save_topic_separation_inputs": False,
            "save_silhouette_samples": False,
            "save_anova_diagnostics": False,
            "save_manova_details": False,
            "save_raw_overlap_matrix": False,
            "visualization": {"interactive": False, "heatmap": False},
        },
        "community": {
            "resolutions": [0.5],
            "n_iterations": 1,
            "min_community_size": 1,
            "seed": 42,
            "node2vec_enabled": False,
        },
        "iterative": {"enabled": False, "batch_size": 1, "iterations": 1, "random_seed": 42},
        "debug": False,
    }
    return PipelineSettings(**config)


def _serialize_graph_for_graphml(graph: nx.DiGraph) -> nx.DiGraph:
    clean_graph = graph.copy()
    for _, data in clean_graph.nodes(data=True):
        for key, value in list(data.items()):
            if value is None:
                del data[key]
            elif isinstance(value, (dict, list)):
                data[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (date, datetime)):
                data[key] = value.isoformat()
    for _, _, data in clean_graph.edges(data=True):
        for key, value in list(data.items()):
            if value is None:
                del data[key]
            elif isinstance(value, (dict, list)):
                data[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (date, datetime)):
                data[key] = value.isoformat()
    return clean_graph


def _save_smoke_artifacts(ctx: PipelineContext, output_dir: str) -> None:
    create_output_directory(output_dir)
    save_graph_schema(ctx.graph, output_dir)

    graph_path = os.path.join(output_dir, "knowledge_graph.graphml")
    clean_graph = _serialize_graph_for_graphml(ctx.graph)
    nx.write_graphml(clean_graph, graph_path)

    report_path = os.path.join(output_dir, "entity_resolution_report.json")
    report = {
        "extraction": ctx.stats.get("extraction", {}),
        "entity_resolution": ctx.stats.get("entity_resolution", {}),
        "graph": {
            "nodes": ctx.graph.number_of_nodes(),
            "edges": ctx.graph.number_of_edges(),
        },
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


async def _run_smoke_extraction(ctx: PipelineContext, extractor: LocalSmokeExtractor) -> Dict[str, Any]:
    results = []
    for task in ctx.extraction_tasks:
        relations, nodes = await extractor.extract_relations(task.chunk_text)
        gliner_entities = [
            {"text": node["id"], "label": node["type"]}
            for node in nodes
        ]
        ctx.graph.nodes[task.chunk_id].update(
            {
                "knowledge_triplets": relations,
                "raw_extraction": {"relations": relations, "nodes": nodes},
                "gliner_entities": gliner_entities,
                "extraction_successful": bool(relations or nodes),
            }
        )
        results.append({"chunk_id": task.chunk_id, "relations": len(relations), "nodes": len(nodes)})

    enrich_result = await enrich_graph_per_segment(ctx)
    return {
        "processed": len(results),
        "successful": sum(1 for item in results if item["relations"] or item["nodes"]),
        "total_entities_extracted": sum(item["nodes"] for item in results),
        "total_relations_extracted": sum(item["relations"] for item in results),
        "errors": enrich_result.get("errors", []),
    }


async def run_local_smoke(input_dir: str, output_dir: str = "output_smoke", max_documents: int = 1) -> Dict[str, Any]:
    settings = build_smoke_settings(input_dir=input_dir, output_dir=output_dir, max_documents=max_documents)
    ctx = PipelineContext(graph=nx.DiGraph())

    lexical_stats = await build_lexical_graph(ctx, settings.infra.input_dir, settings.model_dump())
    extractor = LocalSmokeExtractor()
    extraction_stats = await _run_smoke_extraction(ctx, extractor)

    ctx.stats["lexical"] = lexical_stats
    ctx.stats["extraction"] = extraction_stats
    ctx.stats["entity_resolution"] = {"merged_nodes": 0, "clusters_found": 0}

    _save_smoke_artifacts(ctx, settings.infra.output_dir)

    return {
        "documents_processed": lexical_stats.get("documents_processed", 0),
        "total_segments": lexical_stats.get("total_segments", 0),
        "total_chunks": lexical_stats.get("total_chunks", 0),
        "total_entities_extracted": extraction_stats.get("total_entities_extracted", 0),
        "total_relations_extracted": extraction_stats.get("total_relations_extracted", 0),
        "output_dir": settings.infra.output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small local GraphGen smoke test without remote models.")
    parser.add_argument("--input-dir", default="input/txt/translated", help="Input directory containing .txt files")
    parser.add_argument("--output-dir", default="output_smoke", help="Directory for smoke-test artifacts")
    parser.add_argument("--max-documents", type=int, default=1, help="Maximum number of documents to process")
    args = parser.parse_args()

    result = asyncio.run(
        run_local_smoke(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_documents=args.max_documents,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
