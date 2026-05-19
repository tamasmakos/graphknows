"""Iterative pipeline runner for batch experiments."""

import logging
import json
import os
import hashlib
from datetime import datetime, date
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from graphknows.graphgen.config.settings import PipelineSettings
from graphknows.graphgen.data_types import PipelineContext
from graphknows.graphgen.pipeline.iterative_loader import IterativeLoader
from graphknows.graphgen.pipeline.lexical_graph_building.builder import add_segments_to_graph
from graphknows.graphgen.pipeline.entity_relation.extraction import extract_all_entities_relations
from graphknows.graphgen.pipeline.community.detection import CommunityDetector
from graphknows.graphgen.pipeline.community.subcommunities import (
    add_enhanced_community_attributes_to_graph,
)
from graphknows.graphgen.pipeline.graph_cleaning.resolution import resolve_entities_semantically
from graphknows.graphgen.pipeline.summarization.core import generate_community_summaries
from graphknows.graphgen.analytics.reporting import generate_topic_separation_report
from graphknows.graphgen.analytics.metrics import calculate_node2vec_significance
from graphknows.graphgen.analytics.visualizer import (
    plot_node2vec_impact,
    plot_community_centrality,
    plot_global_centrality,
)
from graphknows.graphgen.analytics.centrality import (
    calculate_centrality_measures,
    get_top_entities_global,
    get_top_entities_per_community,
)
from graphknows.graphgen.utils.vector_embedder.rag import generate_rag_embeddings
from graphknows.graphgen.config.llm import get_langchain_llm
from graphknows.graphgen.config.schema import GraphSchema
from graphknows.graphgen.utils.utils import create_output_directory
from graphknows.graphgen.utils.provenance import (
    write_analysis_run_manifest,
    write_pipeline_config_snapshot,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container types (lightweight dicts with typed aliases)
# ---------------------------------------------------------------------------
BatchResult = Dict[str, Any]
CommunityResult = Dict[str, Any]
AnalyticsResult = Dict[str, Any]
IterationResult = Dict[str, Any]


class IterativeOrchestrator:
    """Run the pipeline in iterative batches for experiments."""

    # ------------------------------------------------------------------
    # Class-level constants — replaces magic literals scattered in run()
    # ------------------------------------------------------------------
    NODE2VEC_DEFAULT_WORKERS: int = 4
    CENTRALITY_GLOBAL_TOP_K: int = 20
    CENTRALITY_COMMUNITY_TOP_K: int = 10
    ENTITY_CONCEPT_NODE_TYPE: str = "ENTITY_CONCEPT"
    TOPIC_NODE_TYPES: List[str] = ["TOPIC", "SUBTOPIC"]
    TOPIC_SNAPSHOT_NODE_TYPES: frozenset = frozenset(
        {"COMMUNITY", "SUBCOMMUNITY", "TOPIC", "SUBTOPIC"}
    )
    DOCUMENT_NODE_TYPE: str = "Document"
    ENTITY_RELATION_GRAPH_TYPE: str = "entity_relation"
    BASELINE_EDGE_WEIGHT: float = 1.0
    CENTRALITY_PLOTS_SUBDIR: str = "centrality_plots"
    CHECKPOINTS_SUBDIR: str = "checkpoints"
    SAMPLES_SUBDIR: str = "samples"

    def __init__(self, settings: PipelineSettings, uploader=None, extractor=None) -> None:
        self.settings = settings
        self.uploader = uploader
        self.extractor = extractor
        file_pattern = settings.extraction.file_pattern if settings.extraction else "*.txt"
        self.loader = IterativeLoader(settings.infra.input_dir, settings.iterative, file_pattern)
        self.results: List[IterationResult] = []

    # ==================================================================
    # Public entry-point — thin pipeline orchestrator (≤ 40 lines)
    # ==================================================================

    async def run(self) -> None:
        if not self.settings.iterative.enabled:
            logger.info("Iterative mode disabled.")
            return

        logger.info(
            f"Starting Iterative Experiment: {self.settings.iterative.iterations} iterations, "
            f"batch size {self.settings.iterative.batch_size}"
        )
        self.run_started_at = datetime.now()

        thesis_output_dir = self._setup_run_environment()
        graph = nx.DiGraph()
        ctx = PipelineContext(graph=graph)
        schema = self._parse_schema()
        config_dict = self.settings.model_dump()
        detector = CommunityDetector(self.settings.community)

        for i in range(self.settings.iterative.iterations):
            logger.info(f"=== Iteration {i + 1}/{self.settings.iterative.iterations} ===")

            segments = self._stage_load_batch(i, thesis_output_dir)
            if not segments:
                logger.warning("No segments loaded. Stopping.")
                break

            await self._stage_build_graph(ctx, segments, i, config_dict, schema)
            await self._stage_extract(ctx, config_dict)
            structural_embeddings = self._stage_resolve_entities(ctx, i)
            comm_results, modularity, modularity_baseline = self._stage_detect_communities(
                ctx, detector, i, structural_embeddings
            )
            await self._stage_summarize(ctx, comm_results, config_dict)
            stats_report = self._stage_export(ctx, i, comm_results, thesis_output_dir)

            result = self._collect_iteration_result(
                i, ctx, comm_results, modularity, modularity_baseline, stats_report
            )
            self.results.append(result)

        self._finalize_run(ctx, thesis_output_dir)

    # ==================================================================
    # Private stage methods
    # ==================================================================

    def _setup_run_environment(self) -> str:
        """Create output directories and write provenance start marker."""
        create_output_directory(self.settings.infra.output_dir)
        thesis_output_dir = os.path.join(
            self.settings.infra.output_dir,
            self.settings.analytics.outputs_subdir,
        )
        create_output_directory(thesis_output_dir)
        if self.settings.analytics.save_provenance:
            write_pipeline_config_snapshot(thesis_output_dir, self.settings)
            write_analysis_run_manifest(
                thesis_output_dir,
                self.settings,
                stage="started",
                run_id="iterative",
                started_at=self.run_started_at.isoformat(),
            )
        return thesis_output_dir

    def _parse_schema(self) -> Optional[GraphSchema]:
        """Parse GraphSchema from settings if configured."""
        if self.settings.schema_config:
            try:
                return GraphSchema(**self.settings.schema_config)
            except Exception:
                pass
        return None

    def _stage_load_batch(self, iteration: int, thesis_output_dir: str) -> List[Any]:
        """Load the next batch of segments and optionally persist a sampling manifest."""
        segments = self.loader.get_batch(iteration)
        if not segments:
            return segments

        logger.info(f"Loaded batch of {len(segments)} segments.")

        if self.settings.analytics.save_sampling_manifest:
            self._save_sample_manifest(iteration, segments, thesis_output_dir)

        return segments

    def _save_sample_manifest(
        self, iteration: int, segments: List[Any], thesis_output_dir: str
    ) -> None:
        """Persist a JSON manifest describing which segments were sampled."""
        samples_dir = os.path.join(thesis_output_dir, self.SAMPLES_SUBDIR)
        create_output_directory(samples_dir)
        sample_manifest = {
            "iteration": iteration + 1,
            "batch_size": len(segments),
            "random_seed": self.settings.iterative.random_seed + iteration,
            "samples": [
                {
                    "segment_id": seg.segment_id,
                    "filename": seg.metadata.get("filename"),
                    "line_number": seg.line_number,
                    "content_sha256": hashlib.sha256(seg.content.encode("utf-8")).hexdigest(),
                }
                for seg in segments
            ],
        }
        manifest_path = os.path.join(samples_dir, f"iteration_{iteration + 1}_sample_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(sample_manifest, f, indent=2, ensure_ascii=False)

    async def _stage_build_graph(
        self,
        ctx: PipelineContext,
        segments: List[Any],
        iteration: int,
        config_dict: Dict[str, Any],
        schema: Optional[GraphSchema],
    ) -> None:
        """Add batch segments to the cumulative lexical graph."""
        by_file: Dict[str, list] = defaultdict(list)
        for seg in segments:
            filename = seg.metadata.get("filename", f"batch_{iteration}")
            by_file[filename].append(seg)

        for filename, file_segments in by_file.items():
            doc_id = f"DOC_{filename}"
            if not ctx.graph.has_node(doc_id):
                ctx.graph.add_node(
                    doc_id,
                    node_type=self.DOCUMENT_NODE_TYPE,
                    ontology_class="STRUCTURAL_DOCUMENT",
                    name=filename,
                )
            await add_segments_to_graph(ctx, file_segments, doc_id, config_dict, schema)

    async def _stage_extract(self, ctx: PipelineContext, config_dict: Dict[str, Any]) -> None:
        """Run NER + relation extraction on pending tasks, then clear the task queue."""
        await extract_all_entities_relations(ctx, config_dict, self.extractor)
        ctx.extraction_tasks = []

    def _stage_resolve_entities(self, ctx: PipelineContext, iteration: int) -> Optional[Any]:
        """Compute optional structural embeddings, generate RAG embeddings, and resolve entities."""
        structural_embeddings = self._compute_structural_embeddings(ctx, iteration)
        generate_rag_embeddings(ctx.graph, node_types=[self.ENTITY_CONCEPT_NODE_TYPE])
        resolution_stats = resolve_entities_semantically(
            ctx.graph,
            similarity_threshold=self.settings.processing.similarity_threshold,
            structural_embeddings=structural_embeddings,
        )
        er_report_path = os.path.join(
            self.settings.infra.output_dir,
            f"iteration_{iteration + 1}_entity_resolution.json",
        )
        with open(er_report_path, "w", encoding="utf-8") as f:
            json.dump(resolution_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved entity resolution report to {er_report_path}")
        return structural_embeddings

    def _compute_structural_embeddings(self, ctx: PipelineContext, iteration: int) -> Optional[Any]:
        """Compute Node2Vec embeddings when the feature is enabled."""
        if not self.settings.community.node2vec_enabled:
            return None
        from graphknows.graphgen.pipeline.embeddings.node2vec_wrapper import compute_node_embeddings

        logger.debug("Computing Node2Vec embeddings for resolution...")
        return compute_node_embeddings(
            ctx.graph,
            dimensions=self.settings.community.node2vec_dimensions,
            workers=self.NODE2VEC_DEFAULT_WORKERS,
        )

    def _stage_detect_communities(
        self,
        ctx: PipelineContext,
        detector: CommunityDetector,
        iteration: int,
        structural_embeddings: Optional[Any],
    ) -> Tuple[CommunityResult, float, float]:
        """Run baseline + optional weighted community detection; return results and modularities."""
        # Baseline: strip entity-relation edge weights
        baseline_graph = ctx.graph.copy()
        for _u, _v, d in baseline_graph.edges(data=True):
            if d.get("graph_type") == self.ENTITY_RELATION_GRAPH_TYPE:
                d["weight"] = self.BASELINE_EDGE_WEIGHT
        baseline_res = detector.detect_communities(baseline_graph)
        modularity_baseline = baseline_res.get("modularity", 0.0)
        logger.info(f"Baseline Modularity (Unweighted): {modularity_baseline:.4f}")
        del baseline_graph

        # Optional Node2Vec weights
        if self.settings.community.node2vec_enabled:
            self._apply_node2vec_weights(ctx, detector, iteration)

        comm_results = detector.detect_communities(ctx.graph)
        modularity = comm_results.get("modularity", 0.0)

        if self.settings.community.node2vec_enabled:
            delta = modularity - modularity_baseline
            logger.debug(
                f"Node2Vec Impact: Delta={delta:+.4f} "
                f"(Baseline={modularity_baseline:.4f} -> Weighted={modularity:.4f})"
            )

        return comm_results, modularity, modularity_baseline

    def _apply_node2vec_weights(
        self, ctx: PipelineContext, detector: CommunityDetector, iteration: int
    ) -> None:
        """Compute Node2Vec edge weights and apply them to the graph."""
        from graphknows.graphgen.pipeline.embeddings.node2vec_wrapper import (
            compute_node2vec_weights,
        )

        logger.debug("Computing Node2Vec weights...")
        entity_relation_graph = detector._get_entity_graph(ctx.graph).copy()
        weights = compute_node2vec_weights(
            entity_relation_graph,
            dimensions=self.settings.community.node2vec_dimensions,
            walk_length=self.settings.community.node2vec_walk_length,
            num_walks=self.settings.community.node2vec_num_walks,
            seed=self.settings.iterative.random_seed + iteration,
        )
        weighted_count = 0
        for (u, v), w in weights.items():
            if ctx.graph.has_edge(u, v):
                ctx.graph[u][v]["weight"] = w
                weighted_count += 1
        logger.debug(f"Applied Node2Vec weights to {weighted_count} edges.")

    async def _stage_summarize(
        self,
        ctx: PipelineContext,
        comm_results: CommunityResult,
        config_dict: Dict[str, Any],
    ) -> None:
        """Detect subcommunities, annotate graph, generate LLM summaries, embed topics."""
        communities = comm_results["assignments"]
        subcommunities = CommunityDetector(self.settings.community).detect_subcommunities_leiden(
            ctx.graph, communities
        )
        add_enhanced_community_attributes_to_graph(ctx.graph, communities, subcommunities)
        llm = get_langchain_llm(config_dict, purpose="summarization")
        await generate_community_summaries(ctx.graph, llm)
        generate_rag_embeddings(ctx.graph, node_types=self.TOPIC_NODE_TYPES)

    def _stage_export(
        self,
        ctx: PipelineContext,
        iteration: int,
        comm_results: CommunityResult,
        thesis_output_dir: str,
    ) -> AnalyticsResult:
        """Run analytics, save checkpoints, and return the stats report."""
        stats_report = generate_topic_separation_report(
            ctx.graph,
            f"{self.settings.infra.output_dir}/iteration_{iteration + 1}_report.json",
            self.settings.analytics,
        )

        if self.settings.analytics.enabled:
            self._run_full_analytics(ctx, iteration, comm_results)

        if self.settings.analytics.save_checkpoints:
            self._save_checkpoint(ctx, iteration, thesis_output_dir)

        return stats_report

    def _run_full_analytics(
        self, ctx: PipelineContext, iteration: int, comm_results: CommunityResult
    ) -> None:
        """Run GraphAnalyzer heatmap/KGE analysis and centrality analytics."""
        from graphknows.graphgen.analytics.analyzer import GraphAnalyzer

        analyzer = GraphAnalyzer(
            config=self.settings.analytics.model_dump(),
            output_base_dir=self.settings.infra.output_dir,
        )
        communities_map = comm_results["assignments"]
        _ = analyzer.run_full_analysis(ctx.graph, communities_map)
        self._run_centrality_analytics(ctx, iteration, communities_map)

    def _run_centrality_analytics(
        self,
        ctx: PipelineContext,
        iteration: int,
        communities_map: Dict[str, Any],
    ) -> None:
        """Calculate centrality scores and save per-iteration JSON; plot on final iteration."""
        try:
            logger.info("Running advanced centrality analysis...")
            centrality_scores = calculate_centrality_measures(ctx.graph)
            if not centrality_scores:
                return

            global_top = get_top_entities_global(
                centrality_scores, ctx.graph, top_k=self.CENTRALITY_GLOBAL_TOP_K
            )
            with open(
                f"{self.settings.infra.output_dir}/iteration_{iteration + 1}_global_centrality.json",
                "w",
            ) as f:
                json.dump(global_top, f, indent=2)

            comm_top = get_top_entities_per_community(
                centrality_scores, communities_map, ctx.graph, top_k=self.CENTRALITY_COMMUNITY_TOP_K
            )
            with open(
                f"{self.settings.infra.output_dir}/iteration_{iteration + 1}_community_centrality.json",
                "w",
            ) as f:
                json.dump(comm_top, f, indent=2)

            is_final = iteration == self.settings.iterative.iterations - 1
            if is_final:
                plots_dir = os.path.join(
                    self.settings.infra.output_dir, self.CENTRALITY_PLOTS_SUBDIR
                )
                plot_global_centrality(global_top, plots_dir)
                plot_community_centrality(comm_top, plots_dir)
                logger.info("Final centrality plots saved to %s", plots_dir)

            logger.info("Centrality analysis completed and saved.")
        except Exception as e:
            logger.error(f"Centrality analysis failed: {e}", exc_info=True)

    def _save_checkpoint(
        self, ctx: PipelineContext, iteration: int, thesis_output_dir: str
    ) -> None:
        """Persist a GraphML snapshot and a compact topic JSON for this iteration."""
        checkpoints_dir = os.path.join(thesis_output_dir, self.CHECKPOINTS_SUBDIR)
        create_output_directory(checkpoints_dir)
        self._save_graphml_checkpoint(ctx, iteration, checkpoints_dir)
        self._save_topic_snapshot(ctx, iteration, checkpoints_dir)

    def _save_graphml_checkpoint(
        self, ctx: PipelineContext, iteration: int, checkpoints_dir: str
    ) -> None:
        """Write a GraphML file, serialising dicts/lists/dates to strings."""
        from datetime import date as _date, datetime as _dt

        graph_path = os.path.join(checkpoints_dir, f"iteration_{iteration + 1}_graph.graphml")
        clean_graph = ctx.graph.copy()
        for _, d in clean_graph.nodes(data=True):
            for k, v in list(d.items()):
                if v is None:
                    del d[k]
                elif isinstance(v, (dict, list)):
                    d[k] = json.dumps(v, ensure_ascii=False)
                elif isinstance(v, (_date, _dt)):
                    d[k] = v.isoformat()
        for _, _, d in clean_graph.edges(data=True):
            for k, v in list(d.items()):
                if v is None:
                    del d[k]
                elif isinstance(v, (dict, list)):
                    d[k] = json.dumps(v, ensure_ascii=False)
                elif isinstance(v, (_date, _dt)):
                    d[k] = v.isoformat()
        nx.write_graphml(clean_graph, graph_path)

    def _save_topic_snapshot(
        self, ctx: PipelineContext, iteration: int, checkpoints_dir: str
    ) -> None:
        """Persist a compact JSON snapshot of all community/topic nodes."""
        topics_snapshot = []
        for node_id, data in ctx.graph.nodes(data=True):
            node_type = str(data.get("node_type", "")).upper()
            if node_type not in self.TOPIC_SNAPSHOT_NODE_TYPES:
                continue
            embedding = data.get("embedding")
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            topics_snapshot.append(
                {
                    "node_id": node_id,
                    "node_type": node_type,
                    "title": data.get("title") or data.get("name"),
                    "summary": data.get("summary"),
                    "community_id": data.get("community_id"),
                    "subcommunity_id": data.get("subcommunity_id"),
                    "embedding": embedding,
                }
            )
        snapshot_path = os.path.join(checkpoints_dir, f"iteration_{iteration + 1}_topics.json")
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(
                {"iteration": iteration + 1, "topics": topics_snapshot},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _collect_iteration_result(
        self,
        iteration: int,
        ctx: PipelineContext,
        comm_results: CommunityResult,
        modularity: float,
        modularity_baseline: float,
        stats_report: AnalyticsResult,
    ) -> IterationResult:
        """Assemble the metrics dict for this iteration and log a summary line."""
        global_separation = stats_report.get("global_separation", 0.0) or 0.0
        global_overlap = stats_report.get("global_overlap", 0.0) or 0.0
        community_sil = 0.0
        if stats_report.get("community_level"):
            community_sil = stats_report["community_level"].get("silhouette_score", 0.0) or 0.0
        subcommunity_sil = 0.0
        if stats_report.get("subcommunity_level"):
            subcommunity_sil = (
                stats_report["subcommunity_level"].get("silhouette_score", 0.0) or 0.0
            )

        result: IterationResult = {
            "iteration": iteration + 1,
            "cumulative_speeches": (iteration + 1) * self.settings.iterative.batch_size,
            "modularity": modularity,
            "modularity_baseline": modularity_baseline,
            "topic_separation": global_separation,
            "topic_overlap": global_overlap,
            "community_silhouette": community_sil,
            "subcommunity_silhouette": subcommunity_sil,
            "nodes": ctx.graph.number_of_nodes(),
            "edges": ctx.graph.number_of_edges(),
            "communities": comm_results.get("community_count", 0),
        }
        logger.info(
            f"Iteration {iteration + 1} Stats: Modularity={modularity:.4f}, "
            f"Separation={global_separation:.4f}, "
            f"Silhouette(Com/Sub)={community_sil:.3f}/{subcommunity_sil:.3f}"
        )
        return result

    def _finalize_run(self, ctx: PipelineContext, thesis_output_dir: str) -> None:
        """Save CSV results, run Node2Vec significance tests, generate thesis plots, upload graph."""
        self._save_results_csv()
        self._run_node2vec_final_analysis()
        self._generate_thesis_plots()
        self._write_provenance_completed(ctx, thesis_output_dir)
        self._upload_final_graph(ctx)

    def _save_results_csv(self) -> None:
        """Write accumulated iteration results to a CSV file."""
        try:
            df = pd.DataFrame(self.results)
            output_csv = f"{self.settings.infra.output_dir}/iterative_experiment_results.csv"
            df.to_csv(output_csv, index=False)
            logger.info(f"Experiment complete. Results saved to {output_csv}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    def _run_node2vec_final_analysis(self) -> None:
        """Compute Node2Vec significance test and save impact plot (runs once at end)."""
        if not (self.settings.community.node2vec_enabled and len(self.results) >= 2):
            return
        try:
            n2v_stats = calculate_node2vec_significance(self.results)
            if n2v_stats:
                sig_msg = "SIGNIFICANT" if n2v_stats.get("significant") else "NOT SIGNIFICANT"
                logger.info(f"Node2Vec Improvement is {sig_msg} (p={n2v_stats.get('p_value'):.4f})")
                stats_path = f"{self.settings.infra.output_dir}/node2vec_stats.json"
                with open(stats_path, "w") as f:
                    json.dump(n2v_stats, f, indent=2)
            plot_node2vec_impact(
                self.results, f"{self.settings.infra.output_dir}/node2vec_impact.png"
            )
        except Exception as e:
            logger.error(f"Node2Vec analysis failed: {e}")

    def _generate_thesis_plots(self) -> None:
        """Generate thesis-quality summary plots from accumulated experiment data."""
        try:
            from graphknows.graphgen.pipeline.visualization.thesis_plots import (
                generate_all_thesis_plots,
            )

            thesis_results = generate_all_thesis_plots(self.settings.infra.output_dir)
            generated = len([v for v in thesis_results.values() if v])
            logger.info(f"Generated {generated} thesis plots.")
        except Exception as e:
            logger.warning(f"Thesis plot generation failed (non-critical): {e}")

    def _write_provenance_completed(self, ctx: PipelineContext, thesis_output_dir: str) -> None:
        """Write the completed provenance manifest."""
        if not self.settings.analytics.save_provenance:
            return
        write_analysis_run_manifest(
            thesis_output_dir,
            self.settings,
            stage="completed",
            run_id="iterative",
            started_at=self.run_started_at.isoformat(),
            completed_at=datetime.now().isoformat(),
            extra={
                "results": self.results,
                "graph": {
                    "nodes": ctx.graph.number_of_nodes(),
                    "edges": ctx.graph.number_of_edges(),
                },
            },
        )

    def _upload_final_graph(self, ctx: PipelineContext) -> None:
        """Upload the final cumulative graph to the configured graph database."""
        if not self.uploader:
            return
        db_type = self.settings.infra.graph_db_type
        logger.info(f"Uploading final iterative graph to {db_type}...")
        try:
            if self.uploader.connect():
                self.uploader.upload(ctx.graph, clean_database=self.settings.infra.clean_start)
                self.uploader.close()
                logger.info("Upload complete.")
            else:
                logger.warning(
                    "Failed to connect to graph database for upload. "
                    "Continuing without DB upload (output graphs saved locally)."
                )
        except Exception as e:
            logger.warning(
                f"Upload failed: {e}. Continuing without DB upload (output graphs saved locally)."
            )
