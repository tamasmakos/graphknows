"""
Topic Separation Metrics.

Calculates metrics related to how distinct topics are from each other, including:
- Silhouette Scores
- Global Separation Indices
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Silhouette configuration
SILHOUETTE_MIN_SAMPLES: int = 3
SILHOUETTE_MIN_SAMPLES_PER_CLUSTER: int = 2
SILHOUETTE_MAX_CLUSTERS_RATIO: float = 0.5  # k <= n * ratio

try:
    from sklearn.metrics import silhouette_score, silhouette_samples
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Silhouette analysis will be skipped.")

def compute_global_separation(embeddings: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """
    Compute average pairwise cosine distance and similarity between all embeddings.
    
    Returns:
        Tuple[float, float]: (average_distance, average_similarity)
    """
    if len(embeddings) < 2:
        return 0.0, 0.0
    
    try:
        X = np.array(list(embeddings.values()))
        # Normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_norm = X / norms
        
        # Cosine Similarity Matrix
        sim_matrix = np.dot(X_norm, X_norm.T)
        
        # Cosine Distance = 1 - Similarity
        dist_matrix = 1.0 - sim_matrix
        
        # Average of upper triangle (excluding diagonal)
        n = X.shape[0]
        tri_indices = np.triu_indices(n, k=1)
        if len(tri_indices[0]) > 0:
            avg_dist = np.mean(dist_matrix[tri_indices])
            avg_sim = np.mean(sim_matrix[tri_indices])
            return float(avg_dist), float(avg_sim)
        return 0.0, 0.0
    except Exception as e:
        logger.error(f"Global separation calculation failed: {e}")
        return 0.0, 0.0

def run_silhouette_analysis(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, int]
) -> Tuple[Optional[float], Optional[Dict[int, float]]]:
    """
    Run silhouette analysis on embeddings.
    
    Silhouette score measures how similar an object is to its own cluster
    compared to other clusters. Score ranges from -1 to 1.
    """
    if not SKLEARN_AVAILABLE:
        return None, None

    # Align embeddings and labels
    common_ids = set(embeddings.keys()) & set(labels.keys())
    if len(common_ids) < SILHOUETTE_MIN_SAMPLES:
        logger.warning(
            "Insufficient samples for silhouette analysis: %d (require >=%d)",
            len(common_ids),
            SILHOUETTE_MIN_SAMPLES,
        )
        return None, None

    ids = list(common_ids)
    X = np.array([embeddings[id_] for id_ in ids])
    y = np.array([labels[id_] for id_ in ids])

    # Basic label statistics
    unique_labels = np.unique(y)
    n_samples = len(X)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        logger.warning("Need at least 2 clusters for silhouette analysis")
        return None, None

    if n_samples < SILHOUETTE_MIN_SAMPLES or n_clusters >= n_samples:
        logger.warning(
            "Silhouette undefined: n_samples=%d, n_clusters=%d",
            n_samples, n_clusters
        )
        return None, None
        
    # Interpretability guardrail
    max_allowed_clusters = max(1, int(n_samples * SILHOUETTE_MAX_CLUSTERS_RATIO))
    if n_clusters > max_allowed_clusters:
        logger.warning(
            "Silhouette likely meaningless: too many clusters (%d) relative to samples (%d)",
            n_clusters, n_samples
        )
        return None, None

    try:
        # Use cosine metric
        overall = silhouette_score(X, y, metric='cosine')

        # Per-sample scores for per-cluster analysis
        sample_scores = silhouette_samples(X, y, metric='cosine')

        per_cluster = {}
        for label in unique_labels:
            mask = y == label
            if np.sum(mask) > 0:
                per_cluster[int(label)] = float(np.mean(sample_scores[mask]))

        return overall, per_cluster

    except Exception as e:
        logger.error(f"Silhouette analysis failed: {e}")
        return None, None
