"""
Statistical Tests for Topic Analysis.

Includes:
- ANOVA / MANOVA on embeddings
- Pairwise comparisons
- Permutation tests
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Statistical tests will be limited.")

try:
    from scipy import stats
    from scipy.stats import f_oneway, ttest_ind
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Statistical tests will be limited.")

@dataclass
class PairwiseComparison:
    """Result of a pairwise comparison between two groups."""
    group1: str
    group2: str
    t_statistic: float
    p_value: float
    effect_size: float  # Cohen's d / Hedges' g
    significant: bool

def run_anova_analysis(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, int]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Run one-way ANOVA on the first Principal Component of embeddings.
    Only valid if groups are roughly normally distributed on PC1.
    """
    if not SCIPY_AVAILABLE or not SKLEARN_AVAILABLE:
        return None, None
    
    common_ids = set(embeddings.keys()) & set(labels.keys())
    if len(common_ids) < 3:
        return None, None
    
    ids = list(common_ids)
    X = np.array([embeddings[id_] for id_ in ids])
    y = np.array([labels[id_] for id_ in ids])
    
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return None, None
    
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=1)
        X_pca = pca.fit_transform(X_scaled).flatten()
        
        groups = [X_pca[y == label] for label in unique_labels]
        groups = [g for g in groups if len(g) > 1]
        
        if len(groups) < 2:
            return None, None
            
        f_stat, p_value = f_oneway(*groups)
        return float(f_stat), float(p_value)
        
    except Exception as e:
        logger.error(f"ANOVA analysis failed: {e}")
        return None, None

def run_multivariate_anova_on_pcs(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, int],
    n_components: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Run multiple univariate ANOVAs on top PCs with Bonferroni correction.
    Proxy for MANOVA when full MANOVA is too expensive or complex.
    """
    if not SCIPY_AVAILABLE or not SKLEARN_AVAILABLE:
        return None

    common_ids = set(embeddings.keys()) & set(labels.keys())
    if len(common_ids) < 10:
        return None

    ids = list(common_ids)
    X = np.array([embeddings[id_] for id_ in ids])
    y = np.array([labels[id_] for id_ in ids])

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return None

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        actual_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
        
        if actual_components < 1:
            return None

        pca = PCA(n_components=actual_components)
        X_pca = pca.fit_transform(X_scaled)

        f_stats = []
        p_values = []
        eta_squared_values = []

        for i in range(actual_components):
            groups = [X_pca[y == label, i] for label in unique_labels]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                continue

            f_stat, p_val = f_oneway(*groups)
            f_stats.append(f_stat)
            p_values.append(p_val)

            # Eta squared
            all_vals = X_pca[:, i]
            ss_total = np.sum((all_vals - np.mean(all_vals)) ** 2)
            ss_between = sum(len(g) * (np.mean(g) - np.mean(all_vals)) ** 2 for g in groups)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
            eta_squared_values.append(eta_sq)

        if not p_values:
            return None

        # Bonferroni correction on minimum p-value
        min_p = min(p_values) * len(p_values)

        return {
            "method": "multiple_univariate_anovas_on_pcs",
            "mean_f_statistic": float(np.mean(f_stats)),
            "min_p_value_corrected": float(min(min_p, 1.0)),
            "mean_eta_squared": float(np.mean(eta_squared_values)),
            "n_components_tested": len(p_values)
        }

    except Exception as e:
        logger.error(f"Multivariate ANOVA failed: {e}")
        return None

def run_pairwise_comparisons(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, int],
    alpha: float = 0.05,
    max_pairs: int = 20
) -> List[PairwiseComparison]:
    """
    Run pairwise t-tests on PC1 between groups.
    """
    if not SCIPY_AVAILABLE or not SKLEARN_AVAILABLE:
        return []
    
    common_ids = set(embeddings.keys()) & set(labels.keys())
    if len(common_ids) < 4:
        return []
    
    ids = list(common_ids)
    X = np.array([embeddings[id_] for id_ in ids])
    y = np.array([labels[id_] for id_ in ids])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_scaled).flatten()
    
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return []
        
    results = []
    
    # Generate all pairs
    import itertools
    pairs = list(itertools.combinations(unique_labels, 2))
    
    # If too many pairs, maybe sample or select largest groups?
    # For now, just truncate
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    
    n_total_comparisons = len(unique_labels) * (len(unique_labels) - 1) // 2
    corrected_alpha = alpha / max(1, n_total_comparisons)
    
    for label1, label2 in pairs:
        group1 = X_pca[y == label1]
        group2 = X_pca[y == label2]
        
        if len(group1) < 2 or len(group2) < 2:
            continue
            
        try:
            t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
            
            # Hedges' g
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            
            if pooled_std > 0:
                d = abs(np.mean(group1) - np.mean(group2)) / pooled_std
                # Hedges' correction
                j = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
                g = d * j
            else:
                g = 0.0
                
            results.append(PairwiseComparison(
                group1=str(label1),
                group2=str(label2),
                t_statistic=float(t_stat),
                p_value=float(p_val),
                effect_size=float(g),
                significant=p_val < corrected_alpha
            ))
            
        except Exception as e:
            logger.warning(f"Pairwise comparison failed for {label1}-{label2}: {e}")
            
    return results

def run_permutation_test(
    group1_embeddings: List[np.ndarray],
    group2_embeddings: List[np.ndarray],
    n_permutations: int = 1000
) -> Dict[str, float]:
    """
    Run a permutation test to check if two groups of embeddings are significantly different.
    Metric: Distance between centroids (Euclidean).
    
    Null hypothesis: Group labels are interchangeable.
    """
    if not group1_embeddings or not group2_embeddings:
        return {}
        
    g1 = np.array(group1_embeddings)
    g2 = np.array(group2_embeddings)
    
    # Observed statistic: distance between means
    clean_g1 = g1[~np.isnan(g1).any(axis=1)]
    clean_g2 = g2[~np.isnan(g2).any(axis=1)]
    
    if len(clean_g1) == 0 or len(clean_g2) == 0:
        return {}
        
    mean1 = np.mean(clean_g1, axis=0)
    mean2 = np.mean(clean_g2, axis=0)
    observed_diff = np.linalg.norm(mean1 - mean2)
    
    # Pooled data
    combined = np.vstack([clean_g1, clean_g2])
    n1 = len(clean_g1)
    
    count_extreme = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_g1 = combined[:n1]
        perm_g2 = combined[n1:]
        
        perm_mean1 = np.mean(perm_g1, axis=0)
        perm_mean2 = np.mean(perm_g2, axis=0)
        perm_diff = np.linalg.norm(perm_mean1 - perm_mean2)
        
        if perm_diff >= observed_diff:
            count_extreme += 1
            
    p_value = (count_extreme + 1) / (n_permutations + 1)
    
    return {
        "observed_centroid_distance": float(observed_diff),
        "p_value": float(p_value),
        "n_permutations": n_permutations
    }
