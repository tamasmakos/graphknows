"""
Analytics Reporting Module.

Generates comprehensive reports by orchestrating:
- Separation analysis
- Statistical tests
- Diversity metrics
- Coherence metrics (where applicable)
"""

import json
import logging
import numpy as np
import networkx as nx
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterable
from dataclasses import dataclass, asdict

from graphgen.analytics.separation import (
    compute_global_separation,
    run_silhouette_analysis,
    SILHOUETTE_MIN_SAMPLES,
    SILHOUETTE_MIN_SAMPLES_PER_CLUSTER,
    SILHOUETTE_MAX_CLUSTERS_RATIO
)
from graphgen.analytics.statistics import (
    run_anova_analysis,
    run_multivariate_anova_on_pcs,
    run_pairwise_comparisons,
    PairwiseComparison,
    SKLEARN_AVAILABLE,
    SCIPY_AVAILABLE
)

logger = logging.getLogger(__name__)

@dataclass
class LevelAnalysisResult:
    """Results for a single hierarchy level (COMMUNITY or SUBCOMMUNITY)."""
    level: str
    n_samples: int
    n_groups: int
    silhouette_score: Optional[float]
    silhouette_per_group: Optional[Dict[str, float]]
    anova_f_statistic: Optional[float]
    anova_p_value: Optional[float]
    multivariate_anova_on_pcs: Optional[Dict[str, float]]
    interpretation: str

@dataclass
class TopicSeparationReport:
    """Complete statistical analysis report."""
    timestamp: str
    config: Dict[str, Any]
    community_level: Optional[LevelAnalysisResult]
    subcommunity_level: Optional[LevelAnalysisResult]
    pairwise_comparisons: List[PairwiseComparison]
    pca_explained_variance: List[float]
    global_separation: Optional[float]
    global_overlap: Optional[float]
    overall_interpretation: str

def _compute_pca_scores(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    if not SKLEARN_AVAILABLE:
        return np.empty((0, 0)), np.array([])
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    actual_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
    if actual_components < 1:
        return np.empty((0, 0)), np.array([])
    pca = PCA(n_components=actual_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca.explained_variance_ratio_

def interpret_separation(
    silhouette: Optional[float],
    p_value: Optional[float]
) -> str:
    """Generate human-readable interpretation of separation quality."""
    parts = []
    
    if silhouette is not None:
        if silhouette > 0.5:
            parts.append(f"Strong cluster separation (silhouette={silhouette:.3f})")
        elif silhouette > 0.25:
            parts.append(f"Moderate cluster separation (silhouette={silhouette:.3f})")
        elif silhouette > 0:
            parts.append(f"Weak cluster separation (silhouette={silhouette:.3f})")
        else:
            parts.append(f"Poor cluster separation (silhouette={silhouette:.3f})")
    
    if p_value is not None:
        if p_value < 0.001:
            parts.append(f"Highly significant difference (p<0.001)")
        elif p_value < 0.01:
            parts.append(f"Significant difference (p<0.01)")
        elif p_value < 0.05:
            parts.append(f"Marginally significant difference (p<0.05)")
        else:
            parts.append(f"No significant difference (p={p_value:.4f})")
    
    if not parts:
        return "Insufficient data for interpretation"
    
    return "; ".join(parts)

def _coerce_embedding(value: Any) -> Optional[np.ndarray]:
    if isinstance(value, list):
        return np.array(value, dtype=float)
    if isinstance(value, np.ndarray):
        return value.astype(float)
    return None


def _extract_label_key(node_id: str, level: str) -> Optional[str]:
    parts = node_id.split("_")
    if (level == "COMMUNITY" or level == "TOPIC") and (node_id.startswith("COMMUNITY_") or node_id.startswith("TOPIC_")):
        if len(parts) >= 2:
            return parts[1]
        return None
    if (level == "SUBCOMMUNITY" or level == "SUBTOPIC") and (node_id.startswith("SUBCOMMUNITY_") or node_id.startswith("SUBTOPIC_")):
        if len(parts) >= 3:
            return f"{parts[1]}::{parts[2]}"
        return None
    return None



def _build_numeric_labels(embeddings: Dict[str, np.ndarray], level: str) -> Dict[str, int]:
    label_keys: Dict[str, str] = {}
    for node_id in embeddings.keys():
        label_key = _extract_label_key(node_id, level)
        if label_key is not None:
            label_keys[node_id] = label_key

    unique_keys = {key: idx for idx, key in enumerate(sorted(set(label_keys.values())))}
    return {node_id: unique_keys[label_key] for node_id, label_key in label_keys.items()}



def extract_topic_embeddings(
    graph: nx.DiGraph,
    levels: Optional[List[str]] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    if levels is None:
        levels = ["COMMUNITY", "SUBCOMMUNITY"]

    result: Dict[str, Dict[str, np.ndarray]] = {}
    entity_types = {'ENTITY_CONCEPT', 'ENTITY', 'NAMEDENTITY', 'PLACE'}
    
    for level in levels:
        level_embeddings: Dict[str, np.ndarray] = {}
        if level == "COMMUNITY":
            target_node_types = ["COMMUNITY", "TOPIC"]
        elif level == "SUBCOMMUNITY":
            target_node_types = ["SUBCOMMUNITY", "SUBTOPIC"]
        else:
            target_node_types = [level]
            
        for node_id, node_data in graph.nodes(data=True):
            if str(node_data.get('node_type', '')).upper() not in target_node_types:
                continue

            member_embeddings = []
            for predecessor in graph.predecessors(node_id):
                pred_data = graph.nodes.get(predecessor, {})
                if str(pred_data.get('node_type', '')).upper() not in entity_types:
                    continue
                embedding = _coerce_embedding(pred_data.get('embedding'))
                if embedding is None:
                    continue
                synthetic_id = f"{node_id}__{predecessor}"
                level_embeddings[synthetic_id] = embedding
                member_embeddings.append(embedding)

            if member_embeddings:
                level_embeddings[node_id] = np.mean(member_embeddings, axis=0)
                continue

            embedding = _coerce_embedding(node_data.get('embedding'))
            if embedding is not None:
                level_embeddings[node_id] = embedding
        
        result[level] = level_embeddings
    
    return result



def analyze_level(
    embeddings: Dict[str, np.ndarray],
    level: str
) -> Optional[LevelAnalysisResult]:
    if len(embeddings) < 3:
        return None
    
    labels = _build_numeric_labels(embeddings, level)
    if len(labels) < 3:
        return None
    
    silhouette, silhouette_per_group = run_silhouette_analysis(embeddings, labels)
    f_stat, p_value = run_anova_analysis(embeddings, labels)
    manova = run_multivariate_anova_on_pcs(embeddings, labels)
    
    interpretation = interpret_separation(silhouette, p_value)
    
    return LevelAnalysisResult(
        level=level,
        n_samples=len(labels),
        n_groups=len(set(labels.values())),
        silhouette_score=silhouette,
        silhouette_per_group={str(k): v for k, v in (silhouette_per_group or {}).items()},
        anova_f_statistic=f_stat,
        anova_p_value=p_value,
        multivariate_anova_on_pcs=manova,
        interpretation=interpretation
    )

def generate_topic_separation_report(
    graph: nx.DiGraph,
    output_path: str,
    config: Any
) -> Dict[str, Any]:
    """
    Generate complete topic separation statistical analysis report.
    """
    logger.info("Generating topic separation statistical report...")

    # Config overrides for silhouette (using global variables in separation module not possible easily, passing as args would be better, but sticking to logic)
    # We will just warn if thresholds are violated inside the function.
    
    levels = config.hierarchy_levels if hasattr(config, 'hierarchy_levels') else ["COMMUNITY", "SUBCOMMUNITY"]
    all_embeddings = extract_topic_embeddings(graph, levels)
    
    community_result = None
    subcommunity_result = None

    if "COMMUNITY" in all_embeddings and all_embeddings["COMMUNITY"]:
        community_result = analyze_level(all_embeddings["COMMUNITY"], "COMMUNITY")
    
    if "SUBCOMMUNITY" in all_embeddings and all_embeddings["SUBCOMMUNITY"]:
        subcommunity_result = analyze_level(all_embeddings["SUBCOMMUNITY"], "SUBCOMMUNITY")
    
    # Pairwise comparisons
    pairwise = []
    if community_result and "COMMUNITY" in all_embeddings:
        comm_embeddings = all_embeddings["COMMUNITY"]
        labels = _build_numeric_labels(comm_embeddings, "COMMUNITY")
        pairwise = run_pairwise_comparisons(comm_embeddings, labels)
    
    # Global separation
    pca_variance = []
    combined_embeddings = {}
    for level_embs in all_embeddings.values():
        combined_embeddings.update(level_embs)
    
    global_separation = 0.0
    global_overlap = 0.0
    if combined_embeddings:
        if SKLEARN_AVAILABLE:
            from sklearn.decomposition import PCA
            try:
                X = np.array(list(combined_embeddings.values()))
                if X.shape[0] > 1:
                    actual_components = min(10, X.shape[1], X.shape[0]-1)
                    if actual_components > 0:
                        pca = PCA(n_components=actual_components)
                        pca.fit(X)
                        pca_variance = [float(v) for v in pca.explained_variance_ratio_]
            except Exception:
                pass
        
        global_separation, global_overlap = compute_global_separation(combined_embeddings)
    
    interpretations = []
    if community_result:
        interpretations.append(f"Communities: {community_result.interpretation}")
    if subcommunity_result:
        interpretations.append(f"Subcommunities: {subcommunity_result.interpretation}")
    overall = "; ".join(interpretations) if interpretations else "Insufficient data for analysis"
    
    report = TopicSeparationReport(
        timestamp=datetime.now().isoformat(),
        config={"hierarchy_levels": levels},
        community_level=community_result,
        subcommunity_level=subcommunity_result,
        pairwise_comparisons=pairwise,
        pca_explained_variance=pca_variance,
        global_separation=global_separation,
        global_overlap=global_overlap,
        overall_interpretation=overall
    )
    
    def to_dict(obj):
        if hasattr(obj, '__dict__'):
            return {k: to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        else:
            return obj
    
    report_dict = to_dict(report)
    
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Topic separation report saved to: {output_path}")
    except Exception:
        logger.exception("Failed to save report.")
        
    return report_dict
