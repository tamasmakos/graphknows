"""Metrics for graph quality and embedding analysis."""

import logging
from collections import defaultdict
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def calculate_modularity(graph: nx.Graph, communities: Dict[str, int]) -> float:
    """Calculate modularity of the graph given community assignments."""
    try:
        # Create community sets for nodes that exist in both graph and communities
        community_sets = defaultdict(set)
        common_nodes = set()
        
        for node, comm_id in communities.items():
            if node in graph:
                community_sets[comm_id].add(node)
                common_nodes.add(node)
        
        # Filter out empty communities
        valid_communities = [nodes for nodes in community_sets.values() if nodes]
        
        if not valid_communities:
            logger.info(
                "Skipping modularity: no valid communities (nodes=%d, communities=%d).",
                graph.number_of_nodes(),
                len(community_sets),
            )
            return 0.0
            
        # Create subgraph of only the nodes involved in communities
        # Modularity requires partition to cover all nodes in the graph
        subgraph = graph.subgraph(common_nodes)
        if len(subgraph) == 0 or subgraph.number_of_edges() == 0:
             return 0.0

        return nx.community.modularity(subgraph, valid_communities)
    except ZeroDivisionError:
        logger.warning("Modularity calculation failed due to ZeroDivisionError (likely no edges).")
        return 0.0
    except Exception:
        logger.exception("Failed to calculate modularity.")
        return 0.0

def calculate_topic_overlap(topic_embeddings: Dict[str, np.ndarray]) -> float:
    """Calculate average cosine similarity between all pairs of topic embeddings."""
    try:
        if len(topic_embeddings) < 2:
            logger.info(
                "Skipping topic overlap: need at least 2 embeddings (found=%d).",
                len(topic_embeddings),
            )
            return 0.0
        
        embeddings = list(topic_embeddings.values())
        if not embeddings:
            return 0.0
            
        # Stack embeddings
        matrix = np.vstack(embeddings)
        
        # Calculate pairwise cosine similarity
        sim_matrix = cosine_similarity(matrix)
        
        # Get upper triangle excluding diagonal
        upper_tri = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
        
        if len(upper_tri) == 0:
            logger.info("Skipping topic overlap: no pairwise similarities computed.")
            return 0.0
            
        return float(np.mean(upper_tri))
    except Exception:
        logger.exception("Failed to calculate topic overlap.")
        return 0.0

def analyze_modularity_vs_overlap(
    results_history: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Correlate modularity with topic overlap across iterations.
    Expects list of dicts with keys 'modularity' and 'topic_overlap'.
    """
    if len(results_history) < 3:
        logger.info(
            "Skipping correlation analysis: need >= 3 data points (found=%d).",
            len(results_history),
        )
        return {}
        
    stats = {}
    try:
        mods = [r.get('modularity', 0) for r in results_history]
        overlaps = [r.get('topic_overlap', 0) for r in results_history]
        
        # Pearson correlation
        p_corr, p_val = pearsonr(mods, overlaps)
        stats['pearson_correlation'] = p_corr
        stats['pearson_p_value'] = p_val
        
        # Spearman correlation (robust to outliers)
        s_corr, s_val = spearmanr(mods, overlaps)
        stats['spearman_correlation'] = s_corr
        stats['spearman_p_value'] = s_val
        
    except Exception:
        logger.exception("Correlation analysis failed.")
        
    return stats

def calculate_node2vec_significance(
    results_history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Perform a paired t-test to determine if Node2Vec significantly improved modularity.
    Expects list of dicts with keys 'modularity' and 'modularity_baseline'.
    """
    stats_res = {}
    try:
        modularity = [r.get('modularity', 0.0) for r in results_history]
        baseline = [r.get('modularity_baseline', 0.0) for r in results_history]
        
        if len(modularity) < 2:
            logger.info("Skipping Node2Vec significance test: need at least 2 iterations.")
            return {}

        # Paired t-test
        t_stat, p_val = ttest_rel(modularity, baseline)
        
        stats_res['t_statistic'] = float(t_stat)
        stats_res['p_value'] = float(p_val)
        stats_res['significant'] = bool(p_val < 0.05)
        
        logger.info(
            f"Node2Vec Significance Test (n={len(modularity)}): t={t_stat:.4f}, p={p_val:.4f} "
            f"({'Significant' if stats_res['significant'] else 'Not Significant'})"
        )
        
    except Exception:
        logger.exception("Node2Vec significance analysis failed.")
        
    return stats_res
