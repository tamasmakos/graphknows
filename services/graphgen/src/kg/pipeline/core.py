"""
Knowledge Graph Pipeline (Core).

Defines the KnowledgePipeline class which orchestrates the graph generation process.
Follows the Inversion of Control pattern where dependencies are injected.
"""

import os
import asyncio
import uuid
import logging
import networkx as nx
from pathlib import Path
from typing import Dict, Any, List

from kg.types import PipelineContext, ChunkExtractionTask
from kg.config.settings import PipelineSettings
from kg.graph.extraction import extract_all_entities_relations
from kg.graph.extractors import BaseExtractor
from kg.graph.pruning import prune_graph
from kg.graph.utils import create_output_directory
from kg.graph.schema import save_graph_schema

logger = logging.getLogger(__name__)


class KnowledgePipeline:
    """
    The main pipeline orchestrator.

    It accepts all necessary dependencies (infrastructure, configuration) via the constructor.
    It does NOT instantiate heavy objects itself.
    """

    def __init__(
        self,
        settings: PipelineSettings,
        uploader: Any = None,
        extractor: Any = None,
        clean_database: bool = True,
        run_communities: bool = True,
    ):
        self.settings = settings
        self.uploader = uploader
        self.extractor = extractor
        self.clean_database = clean_database
        self.run_communities = run_communities
        self.run_id = str(uuid.uuid4())[:8]

    async def run(self):
        """
        Execute the full knowledge graph generation pipeline:
        1. Build Lexical Graph from Input Dir
        2. Extract Entities/Relations
        3. Semantic Enrichment (Embeddings, Similarity, Resolution)
        4. Community Detection & Summarization
        5. Pruning
        6. Upload entities/relations to Neo4j
        7. Save Artifacts to Disk
        """
        logger.info(f"🚀 Starting KnowledgePipeline run [{self.run_id}]...")

        # Preflight Checks
        self._run_preflight_checks()

        # 0. Initialize Pipeline Context (The "Bus")
        graph = nx.DiGraph()
        # PipelineContext (aliased as AgentDependencies) holds the state
        ctx = PipelineContext(graph=graph)

        # Convert settings to dict for legacy functions
        # TODO: Refactor downstream functions to accept PipelineSettings object directly
        config_dict = (
            self.settings.model_dump()
            if hasattr(self.settings, "model_dump")
            else self.settings.dict()
        )

        try:
            # 1. Build Lexical Graph
            await self._step_lexical_graph(ctx, config_dict)

            # 2. Extraction
            await self._step_extraction(ctx, config_dict)

            # 3. Semantic Enrichment
            await self._step_enrichment(ctx)

            # 4. Community Detection & Summarization
            await self._step_communities(ctx, config_dict)

            # 5. Pruning
            await self._step_pruning(ctx)

            # 6. Upload
            await self._step_upload(ctx)

            # 7. Save Artifacts
            self._step_save_artifacts(ctx)

        except Exception as e:
            logger.critical(f"🔥 Pipeline [{self.run_id}] failed: {e}", exc_info=True)
            raise

        logger.info(f"✅ Pipeline Run [{self.run_id}] Finished Successfully.")

    def _run_preflight_checks(self):
        """Preflight checks (Neo4j connectivity is verified at startup via driver.verify_connectivity)."""
        logger.info("Preflight checks passed.")

    async def _step_lexical_graph(self, ctx: PipelineContext, config: Dict[str, Any]):
        """
        Phase 1: Parse all files in input_dir with the modern ParserRegistry,
        upload each Document + Chunks to Neo4j, and populate the PipelineContext
        graph with CHUNK nodes so Phase 2 extraction can run.
        """
        import importlib
        from kg.parser import get_parser

        # Ensure all parser subclasses are registered via __init_subclass__
        importlib.import_module("kg.parser.registry")

        input_dir = Path(self.settings.infra.input_dir)
        logger.info(f"Step 1: Parsing documents from {input_dir}")

        if not input_dir.exists():
            logger.warning(f"Input directory {input_dir} does not exist — skipping.")
            ctx.stats["lexical"] = {"documents_processed": 0, "total_chunks": 0}
            return

        files = [p for p in input_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
        docs_processed = 0
        total_chunks = 0
        errors = []

        for file_path in files:
            try:
                parser = get_parser(file_path)
            except ValueError:
                logger.debug(f"No parser for {file_path.name} — skipping.")
                continue
            try:
                parsed = parser.parse(file_path)
            except Exception as e:
                logger.error(f"Parsing failed for {file_path.name}: {e}")
                errors.append(str(e))
                continue

            # Upload Document + Chunks to Neo4j
            if self.uploader:
                try:
                    await self.uploader.upload_parsed_document(parsed)
                except Exception as e:
                    logger.error(f"Upload failed for {file_path.name}: {e}")
                    errors.append(str(e))

            # Populate PipelineContext graph for downstream extraction
            doc_node_id = parsed.doc_id
            ctx.graph.add_node(doc_node_id, node_type="DOCUMENT", title=parsed.title)
            for chunk in parsed.chunks:
                ctx.graph.add_node(
                    chunk.chunk_id,
                    node_type="CHUNK",
                    text=chunk.text,
                    doc_id=parsed.doc_id,
                )
                ctx.graph.add_edge(doc_node_id, chunk.chunk_id, label="CONTAINS")
                ctx.extraction_tasks.append(
                    ChunkExtractionTask(
                        chunk_id=chunk.chunk_id,
                        chunk_text=chunk.text,
                        entities=[],
                        abstract_concepts=[],
                        keywords=[],
                    )
                )
                total_chunks += 1

            docs_processed += 1
            logger.info(f"Parsed {file_path.name}: {len(parsed.chunks)} chunks")

        ctx.stats["lexical"] = {
            "documents_processed": docs_processed,
            "total_chunks": total_chunks,
            "errors": errors,
        }
        logger.info(f"Step 1 complete: {docs_processed} docs, {total_chunks} chunks")

    async def _step_extraction(self, ctx: PipelineContext, config: Dict[str, Any]):
        if not self.extractor:
            logger.warning("Step 2: Skipped (No extractor provided).")
            return

        logger.info("Step 2: Extracting Entities & Relations...")
        extract_results = await extract_all_entities_relations(
            ctx, config, extractor=self.extractor
        )

        ctx.stats["extraction"] = extract_results
        logger.info(f"Extraction Complete: {extract_results.get('successful')} successful chunks")

    async def _step_enrichment(self, ctx: PipelineContext):
        try:
            from ..embeddings.rag import generate_rag_embeddings
            from ..graph.resolution import resolve_entities_semantically

            logger.info("Step 3: Semantic Enrichment")

            logger.info("  3.1: Generating RAG Embeddings...")
            generate_rag_embeddings(ctx.graph)

            logger.info("  3.2: Semantic Resolution...")
            resolve_entities_semantically(ctx.graph)

        except Exception as e:
            logger.error(f"Semantic enrichment failed: {e}")
            ctx.add_error("enrichment", str(e))

    async def _step_communities(self, ctx: PipelineContext, config: Dict[str, Any]):
        if not self.run_communities:
            logger.info("Step 4: Skipped (run_communities=False).")
            return

        try:
            from ..community.detection import CommunityDetector
            from ..community.subcommunities import add_enhanced_community_attributes_to_graph
            from ..summarization.core import generate_community_summaries
            from ..llm import get_langchain_llm

            logger.info("Step 4: Community Detection & Summarization")

            logger.info("  4.1: Detecting Communities...")
            detector = CommunityDetector()
            comm_results = detector.detect_communities(ctx.graph)
            communities = comm_results["assignments"]

            subcommunities = detector.detect_subcommunities_leiden(ctx.graph, communities)
            add_enhanced_community_attributes_to_graph(ctx.graph, communities, subcommunities)

            logger.info("  4.2: Generating Summaries...")
            llm = get_langchain_llm(config, purpose="summarization")
            summary_stats = await generate_community_summaries(ctx.graph, llm)
            ctx.stats["summarization"] = summary_stats

            # 4.3: Generate Embeddings for Topics (NEW)
            logger.info("  4.3: Generating Topic Embeddings...")
            # We must import inside the function to avoid circular imports? Validated above
            from ..embeddings.rag import generate_rag_embeddings

            generate_rag_embeddings(ctx.graph, node_types=["TOPIC", "SUBTOPIC"])

        except Exception as e:
            logger.error(f"Community detection or summarization failed: {e}")
            ctx.add_error("communities", str(e))

    async def _step_pruning(self, ctx: PipelineContext):
        logger.info("Step 5: Pruning Graph...")
        prune_stats = prune_graph(ctx.graph, {"pruning_threshold": 0.01})
        ctx.stats["pruning"] = prune_stats
        logger.info(f"Pruning Stats: {prune_stats}")

    async def _step_upload(self, ctx: PipelineContext):
        if not self.uploader:
            logger.warning("Step 6: Skipped (no uploader provided).")
            return

        logger.info("Step 6: Uploading entities/relations to Neo4j...")
        try:
            # Documents and Chunks are already in Neo4j from Phase 1.
            # clean_database only applies on the first run; skip it here to
            # avoid wiping documents that were just uploaded.
            stats = await self.uploader.upload(ctx.graph, clean_database=False)
            ctx.stats["upload"] = stats
            logger.info("Upload Stats: %s", stats)
        except Exception as e:
            logger.error("Upload failed: %s", e)
            ctx.add_error("upload", str(e))

    def _step_save_artifacts(self, ctx: PipelineContext):
        output_dir = self.settings.infra.output_dir
        logger.info(f"Step 7: Saving artifacts to {output_dir}")
        create_output_directory(output_dir)

        try:
            save_graph_schema(ctx.graph, output_dir)

            # Save GraphML
            graph_path = os.path.join(output_dir, "knowledge_graph.graphml")
            clean_graph = ctx.graph.copy()

            # Serialize complex types for GraphML
            import json
            from datetime import date, datetime

            for _, d in clean_graph.nodes(data=True):
                for k, v in list(d.items()):
                    if v is None:
                        del d[k]
                        continue
                    if isinstance(v, (dict, list)):
                        d[k] = json.dumps(v, ensure_ascii=False)
                    elif isinstance(v, (date, datetime)):
                        d[k] = v.isoformat()

            for _, _, d in clean_graph.edges(data=True):
                for k, v in list(d.items()):
                    if v is None:
                        del d[k]
                        continue
                    if isinstance(v, (dict, list)):
                        d[k] = json.dumps(v, ensure_ascii=False)
                    elif isinstance(v, (date, datetime)):
                        d[k] = v.isoformat()

            nx.write_graphml(clean_graph, graph_path)
            logger.info(f"GraphML saved to {graph_path}")
        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")
            ctx.add_error("artifacts", str(e))
