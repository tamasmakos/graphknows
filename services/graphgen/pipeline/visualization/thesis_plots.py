"""
Thesis-quality visualization module for Knowledge Graph Embedding analysis.

Generates publication-ready figures demonstrating:
1. How KGE affects modularity (structural improvement)
2. How modularity relates to topic separation (semantic clustering)
3. Statistical evidence of community quality (silhouette, ANOVA, MANOVA)
4. Dimensionality analysis (PCA scree plots)
5. Per-community clustering breakdown

All plots follow academic publication standards:
- High DPI (300+), vector-compatible
- Consistent color palette and font sizing
- Proper axis labels, titles, legends
- LaTeX-compatible font rendering where possible

Usage:
    from graphgen.pipeline.visualization.thesis_plots import generate_all_thesis_plots
    generate_all_thesis_plots("/app/output")
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from graphgen.utils.logging import configure_logging  # type: ignore[import-not-found]

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import MaxNLocator
    import matplotlib.patheffects as pe
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Thesis plots cannot be generated.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import networkx as nx
    import powerlaw
    NETWORK_ANALYSIS_AVAILABLE = True
except ImportError:
    NETWORK_ANALYSIS_AVAILABLE = False
    logger.warning("networkx or powerlaw not available. Degree distribution plots cannot be generated.")

# ---------------------------------------------------------------------------
# Style Configuration (imported from shared module)
# ---------------------------------------------------------------------------

from graphgen.analytics.plot_style import (
    COLORS,
    COMM_PALETTE,
    apply_thesis_style as _apply_thesis_style,
)




# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _load_csv(output_dir: str) -> Optional[pd.DataFrame]:
    """Load the iterative experiment results CSV."""
    csv_path = Path(output_dir) / "iterative_experiment_results.csv"
    if not csv_path.exists():
        logger.warning(f"CSV not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with {len(df)} iterations and columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return None


def _load_iteration_reports(output_dir: str) -> Dict[int, Dict]:
    """Load all iteration_X_report.json files."""
    reports = {}
    output_path = Path(output_dir)
    for report_file in sorted(output_path.glob("iteration_*_report.json")):
        try:
            # Extract iteration number from filename
            stem = report_file.stem  # e.g. "iteration_2_report"
            parts = stem.split('_')
            iteration = int(parts[1])
            with open(report_file, 'r', encoding='utf-8') as f:
                reports[iteration] = json.load(f)
            logger.info(f"Loaded report for iteration {iteration}")
        except (ValueError, IndexError, json.JSONDecodeError) as e:
            logger.warning(f"Skipping report {report_file}: {e}")
    return reports


def _load_silhouette_samples(diagnostics_dir: Path) -> List[Dict[str, Any]]:
    """Load silhouette sample diagnostics from JSON files."""
    samples: List[Dict[str, Any]] = []
    for file_path in sorted(diagnostics_dir.glob("*_silhouette_samples.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            iteration = None
            match = re.search(r"iteration_(\d+)", file_path.stem)
            if match:
                iteration = int(match.group(1))
            samples.append({
                "file": str(file_path),
                "iteration": iteration,
                "level": payload.get("level"),
                "samples": payload.get("samples", []),
            })
        except Exception as e:
            logger.warning("Failed to load silhouette samples %s: %s", file_path, e)
    return samples


# ---------------------------------------------------------------------------
# Figure 1: KGE Impact on Modularity (Removed)
# ---------------------------------------------------------------------------

# plot_kge_modularity_impact removed


# ---------------------------------------------------------------------------
# Figure 2: Modularity vs Topic Separation (Dual-Axis)
# ---------------------------------------------------------------------------

def plot_modularity_vs_separation(df: pd.DataFrame, output_dir: str) -> Optional[str]:
    """
    Dual-axis plot showing structural modularity alongside semantic topic separation.

    Hypothesis: As modularity increases, topic embeddings should become more
    distinct (higher cosine distance / lower overlap), supporting the claim that
    KGE-enhanced community detection produces semantically coherent clusters.

    Returns:
        Path to saved figure, or None on failure.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    required_cols = {'iteration', 'modularity', 'topic_separation', 'topic_overlap'}
    if not required_cols.issubset(df.columns):
        logger.warning(f"Missing columns for modularity-separation plot: {required_cols - set(df.columns)}")
        return None

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    iterations = df['iteration'].values

    # Left axis: Modularity
    line1, = ax1.plot(
        iterations, df['modularity'].values,
        color=COLORS['modularity'], marker='o', linewidth=2.5,
        label='Modularity (Q)', zorder=4
    )
    ax1.fill_between(
        iterations, df['modularity'].values, alpha=0.08,
        color=COLORS['modularity']
    )
    ax1.set_xlabel('Iteration (cumulative data)')
    ax1.set_ylabel('Modularity (Q)', color=COLORS['modularity'])
    ax1.tick_params(axis='y', labelcolor=COLORS['modularity'])
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Right axis: Topic Separation and Overlap
    ax2 = ax1.twinx()
    line2, = ax2.plot(
        iterations, df['topic_separation'].values,
        color=COLORS['separation'], marker='s', linewidth=2.0,
        linestyle='--', label='Topic Separation (cosine dist.)', zorder=3
    )
    line3, = ax2.plot(
        iterations, df['topic_overlap'].values,
        color=COLORS['overlap'], marker='^', linewidth=1.5,
        linestyle=':', label='Topic Overlap (cosine sim.)', zorder=3
    )
    ax2.set_ylabel('Semantic Distance / Similarity', color=COLORS['separation'])
    ax2.tick_params(axis='y', labelcolor=COLORS['separation'])

    # Combined legend
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(0.01, 0.35), framealpha=0.9)

    ax1.set_title('Structural Modularity vs. Semantic Topic Separation Across Iterations')

    plt.tight_layout()
    out_path = str(Path(output_dir) / "thesis_modularity_vs_separation.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved modularity vs separation plot: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: Silhouette Score Dashboard
# ---------------------------------------------------------------------------

def plot_silhouette_dashboard(
    df: pd.DataFrame,
    reports: Dict[int, Dict],
    output_dir: str
) -> Optional[str]:
    """
    Multi-panel figure showing silhouette scores at entity, community, and
    subcommunity levels across iterations.

    Silhouette score measures how similar an object is to its own cluster
    vs other clusters (-1 to +1). Positive values indicate well-separated clusters.

    Panel A: Line chart of silhouette scores across iterations (3 levels)
    Panel B: Per-group silhouette breakdown for the final iteration

    Returns:
        Path to saved figure, or None on failure.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.35, figure=fig)

    # -- Panel A: Silhouette trends across iterations --
    ax_a = fig.add_subplot(gs[0])

    iterations = df['iteration'].values
    levels = [
        ('community_silhouette', 'Community Level', COLORS['community']),
        ('subcommunity_silhouette', 'Subcommunity Level', COLORS['subcommunity']),
    ]

    has_data = False
    for col, label, color in levels:
        if col in df.columns:
            values = df[col].values
            ax_a.plot(iterations, values, marker='o', color=color, label=label, linewidth=2.0)
            has_data = True

    if has_data:
        ax_a.axhline(y=0, color='gray', linewidth=1.0, linestyle='-', alpha=0.5)
        ax_a.axhspan(-1, 0, alpha=0.04, color=COLORS['silhouette_neg'])
        ax_a.axhspan(0, 1, alpha=0.04, color=COLORS['silhouette_pos'])
        ax_a.set_xlabel('Iteration')
        ax_a.set_ylabel('Silhouette Score')
        ax_a.set_title('(A) Clustering Quality Across Iterations')
        ax_a.legend(loc='best', framealpha=0.9)
        ax_a.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Add interpretation bands
        ax_a.text(
            0.02, 0.95, 'Good separation',
            transform=ax_a.transAxes, fontsize=8, va='top', color=COLORS['accent'],
            alpha=0.7
        )
        ax_a.text(
            0.02, 0.05, 'Poor separation',
            transform=ax_a.transAxes, fontsize=8, va='bottom', color=COLORS['secondary'],
            alpha=0.7
        )

    # -- Panel B: Per-group silhouette for final iteration --
    ax_b = fig.add_subplot(gs[1])

    # Find the last iteration report that has subcommunity per-group data
    final_report = None
    for it in sorted(reports.keys(), reverse=True):
        rpt = reports[it]
        # Prefer subcommunity level (usually more groups)
        sub = rpt.get('subcommunity_level') or {}
        if sub.get('silhouette_per_group'):
            final_report = rpt
            final_iter = it
            break

    if final_report:
        # Combine available per-group data
        per_group_data = []
        for level_key, level_label, color in [
            ('subcommunity_level', 'Subcommunity', COLORS['subcommunity']),
        ]:
            level_data = final_report.get(level_key) or {}
            spg = level_data.get('silhouette_per_group') or {}
            for group_id, score in spg.items():
                per_group_data.append({
                    'group': f'{level_label} {group_id}',
                    'score': score,
                    'level': level_label,
                    'color': color,
                })

        if per_group_data:
            per_group_df = pd.DataFrame(per_group_data)
            per_group_df = per_group_df.sort_values('score', ascending=True)

            colors = [
                COLORS['silhouette_pos'] if s >= 0 else COLORS['silhouette_neg']
                for s in per_group_df['score']
            ]

            y_pos = np.arange(len(per_group_df))
            ax_b.barh(y_pos, per_group_df['score'], color=colors, edgecolor='white', linewidth=0.5)
            ax_b.set_yticks(y_pos)
            ax_b.set_yticklabels(per_group_df['group'], fontsize=8)
            ax_b.axvline(x=0, color='gray', linewidth=1.0)
            ax_b.set_xlabel('Silhouette Score')
            ax_b.set_title(f'(B) Per-Group Silhouette (Iteration {final_iter})')

            # Color-code positive/negative regions
            ax_b.axvspan(0, ax_b.get_xlim()[1], alpha=0.03, color=COLORS['silhouette_pos'])
            ax_b.axvspan(ax_b.get_xlim()[0], 0, alpha=0.03, color=COLORS['silhouette_neg'])
    else:
        ax_b.text(0.5, 0.5, 'No per-group data available',
                  transform=ax_b.transAxes, ha='center', va='center', fontsize=11, color='gray')
        ax_b.set_title('(B) Per-Group Silhouette Breakdown')

    # constrained_layout is set at figure creation; no tight_layout needed here
    out_path = str(Path(output_dir) / "thesis_silhouette_dashboard.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved silhouette dashboard: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 4: PCA Explained Variance (Scree Plot)
# ---------------------------------------------------------------------------

def plot_pca_scree(reports: Dict[int, Dict], output_dir: str) -> Optional[str]:
    """
    Scree plot showing PCA explained variance ratio for the embedding space.

    This demonstrates the effective dimensionality of the topic embedding space.
    A steep drop-off suggests the topics can be well-represented in fewer
    dimensions, supporting the claim of distinct topical clusters.

    Returns:
        Path to saved figure, or None on failure.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    # Collect PCA data from all iterations
    fig, ax = plt.subplots(figsize=(8, 5))

    has_data = False
    for iteration in sorted(reports.keys()):
        pca_var = reports[iteration].get('pca_explained_variance', [])
        if not pca_var:
            continue

        has_data = True
        components = np.arange(1, len(pca_var) + 1)
        cumulative = np.cumsum(pca_var)

        alpha = 0.4 + 0.6 * (iteration / max(reports.keys()))
        color = COLORS['primary']

        ax.bar(
            components - 0.2 + 0.4 * (iteration - 1) / max(1, len(reports) - 1),
            pca_var, width=0.35, alpha=alpha, color=color,
            label=f'Iter {iteration} (individual)', edgecolor='white', linewidth=0.5
        )
        ax.plot(
            components, cumulative, marker='o', linewidth=1.8,
            color=COLORS['secondary'], alpha=alpha,
            label=f'Iter {iteration} (cumulative)' if iteration == max(reports.keys()) else None
        )

        # Annotate cumulative at last component
        if len(cumulative) > 0:
            ax.annotate(
                f'{cumulative[-1]:.1%}',
                xy=(components[-1], cumulative[-1]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, color=COLORS['secondary'], alpha=alpha
            )

    if not has_data:
        plt.close(fig)
        logger.warning("No PCA data available for scree plot.")
        return None

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Scree Plot of Topic Embedding Space')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add 80% threshold line
    ax.axhline(y=0.8, color=COLORS['accent'], linestyle='--', linewidth=1.0, alpha=0.5)
    ax.text(0.5, 0.82, '80% variance threshold', transform=ax.get_yaxis_transform(),
            fontsize=9, color=COLORS['accent'], alpha=0.7)

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='center right', framealpha=0.9)

    plt.tight_layout()
    out_path = str(Path(output_dir) / "thesis_pca_scree.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved PCA scree plot: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 5: Statistical Significance Summary
# ---------------------------------------------------------------------------

def plot_statistical_summary(
    df: pd.DataFrame,
    reports: Dict[int, Dict],
    output_dir: str
) -> Optional[str]:
    """
    Summary table-figure showing key statistical test results across iterations.

    Displays ANOVA F-statistics, p-values, MANOVA approximations, and
    silhouette scores in a structured visual format suitable for a thesis
    results section.

    Returns:
        Path to saved figure, or None on failure.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    # Build summary table data
    rows = []
    for iteration in sorted(reports.keys()):
        rpt = reports[iteration]
        row = {'Iteration': iteration}

        # Extract metrics from each level
        for level_key, level_label in [
            ('subcommunity_level', 'Subcommunity'),
        ]:
            level = rpt.get(level_key) or {}
            row[f'{level_label}\nSilhouette'] = level.get('silhouette_score')
            row[f'{level_label}\nANOVA F'] = level.get('anova_f_statistic')
            row[f'{level_label}\nANOVA p'] = level.get('anova_p_value')

            # Check both old and new field names for backward compatibility
            manova = level.get('multivariate_anova_on_pcs') or level.get('manova_approximation') or {}
            row[f'{level_label}\nMult.ANOVA F'] = manova.get('mean_f_statistic')
            row[f'{level_label}\nMult.ANOVA p'] = manova.get('min_p_value_corrected')
            row[f'{level_label}\nEta-sq'] = manova.get('mean_eta_squared')

        # Add modularity from CSV
        csv_row = df[df['iteration'] == iteration]
        if not csv_row.empty:
            row['Modularity'] = csv_row['modularity'].values[0]
            row['Mod. Baseline'] = csv_row['modularity_baseline'].values[0]
            row['Separation'] = csv_row['topic_separation'].values[0]

        rows.append(row)

    if not rows:
        logger.warning("No data for statistical summary.")
        return None

    summary_df = pd.DataFrame(rows)

    # Create figure with table
    fig, ax = plt.subplots(figsize=(16, 3 + len(rows) * 0.8))
    ax.axis('off')

    # Format cells
    cell_text = []
    col_labels = list(summary_df.columns)

    for _, row in summary_df.iterrows():
        formatted = []
        for col in col_labels:
            val = row[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                formatted.append('--')
            elif isinstance(val, float):
                if 'p' in col.lower() and val < 0.001:
                    formatted.append(f'{val:.2e}')
                elif 'p' in col.lower():
                    formatted.append(f'{val:.4f}')
                elif abs(val) < 0.01:
                    formatted.append(f'{val:.4f}')
                else:
                    formatted.append(f'{val:.3f}')
            else:
                formatted.append(str(val))
        cell_text.append(formatted)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)

    # Style header row
    for j, col in enumerate(col_labels):
        cell = table[0, j]
        cell.set_facecolor(COLORS['primary'])
        cell.set_text_props(color='white', fontweight='bold')

    # Highlight significant p-values
    for i in range(len(cell_text)):
        for j, col in enumerate(col_labels):
            if 'p' in col.lower():
                try:
                    val = summary_df.iloc[i][col]
                    if val is not None and isinstance(val, float):
                        if val < 0.01:
                            table[i + 1, j].set_facecolor('#d4edda')  # Green for significant
                        elif val < 0.05:
                            table[i + 1, j].set_facecolor('#fff3cd')  # Yellow for marginal
                except (KeyError, IndexError):
                    pass

    ax.set_title(
        'Statistical Test Results: Topic Separation Analysis',
        fontsize=13, fontweight='bold', pad=20
    )

    plt.tight_layout()
    out_path = str(Path(output_dir) / "thesis_statistical_summary.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved statistical summary: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 6: KGE Modularity Delta with Separation Overlay (Removed)
# ---------------------------------------------------------------------------

# plot_kge_delta_separation removed


# ---------------------------------------------------------------------------
# Figure 7: MANOVA Component Analysis
# ---------------------------------------------------------------------------

def plot_manova_components(reports: Dict[int, Dict], output_dir: str) -> Optional[str]:
    """
    Visualize MANOVA approximation results across iterations.

    Shows how the multivariate analysis of variance (approximated via
    multiple ANOVAs on principal components) evolves as more data is added.

    Returns:
        Path to saved figure, or None on failure.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    iterations_list = []
    sub_f = []
    sub_p = []
    sub_var = []

    for iteration in sorted(reports.keys()):
        rpt = reports[iteration]
        iterations_list.append(iteration)

        # Subcommunity level multivariate ANOVA on PCs
        sub_level = rpt.get('subcommunity_level') or {}
        sub = sub_level.get('multivariate_anova_on_pcs') or sub_level.get('manova_approximation') or {}
        sub_f.append(sub.get('mean_f_statistic'))
        sub_p.append(sub.get('min_p_value_corrected'))
        sub_var.append(sub.get('explained_variance_ratio'))

    iterations_arr = np.array(iterations_list)

    # Panel A: F-statistics
    valid_sf = [(it, f) for it, f in zip(iterations_list, sub_f) if f is not None]

    if valid_sf:
        ax1.plot(
            [v[0] for v in valid_sf], [v[1] for v in valid_sf],
            marker='s', color=COLORS['subcommunity'], label='Subcommunity Level', linewidth=2.0
        )

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean F-statistic')
    ax1.set_title('(A) Multivariate ANOVA F-statistic (Group Separation Strength)')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Panel B: P-values (log scale)
    valid_sp = [(it, p) for it, p in zip(iterations_list, sub_p) if p is not None]

    if valid_sp:
        ax2.semilogy(
            [v[0] for v in valid_sp], [v[1] for v in valid_sp],
            marker='s', color=COLORS['subcommunity'], label='Subcommunity Level', linewidth=2.0
        )

    # Significance thresholds
    ax2.axhline(y=0.05, color=COLORS['overlap'], linestyle='--', linewidth=1.0, alpha=0.7)
    ax2.axhline(y=0.01, color=COLORS['secondary'], linestyle='--', linewidth=1.0, alpha=0.7)
    ax2.text(ax2.get_xlim()[1], 0.05, ' p=0.05', va='center', fontsize=9, color=COLORS['overlap'])
    ax2.text(ax2.get_xlim()[1], 0.01, ' p=0.01', va='center', fontsize=9, color=COLORS['secondary'])

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Corrected p-value (log scale)')
    ax2.set_title('(B) Mult. ANOVA p-values (Bonferroni-corrected)')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    out_path = str(Path(output_dir) / "thesis_multivariate_anova_components.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved MANOVA components plot: {out_path}")
    return out_path


def plot_silhouette_distributions(
    diagnostics_dir: Path,
    output_dir: str
) -> Optional[str]:
    """Plot silhouette score distributions per iteration and level."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    samples_payload = _load_silhouette_samples(diagnostics_dir)
    if not samples_payload:
        return None

    # Build data by level and iteration
    levels = ["COMMUNITY", "SUBCOMMUNITY"]
    data_by_level: Dict[str, Dict[int, List[float]]] = {lvl: {} for lvl in levels}
    for payload in samples_payload:
        level = payload.get("level")
        if level not in levels:
            continue
        iteration = payload.get("iteration")
        scores = [s.get("score") for s in payload.get("samples", []) if s.get("score") is not None]
        if iteration is None or not scores:
            continue
        data_by_level[level].setdefault(iteration, []).extend(scores)

    if not any(data_by_level.values()):
        return None

    # Filter to only levels with data
    levels_with_data = [lvl for lvl in levels if data_by_level[lvl]]
    if not levels_with_data:
        return None

    fig, axes = plt.subplots(nrows=len(levels_with_data), ncols=1, figsize=(9, 5 * len(levels_with_data)), sharex=True)
    if len(levels_with_data) == 1:
        axes = [axes]  # Make it iterable
    for ax, level in zip(axes, levels_with_data):
        iterations = sorted(data_by_level[level].keys())
        if not iterations:
            ax.set_axis_off()
            continue
        values = [data_by_level[level][it] for it in iterations]
        ax.boxplot(values, labels=[f"Iter {i}" for i in iterations], showfliers=False)
        ax.set_title(f"Silhouette distributions: {level.lower()}")
        ax.set_ylabel("Silhouette score")
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()
    out_path = str(Path(output_dir) / "thesis_silhouette_distributions.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 8: Comprehensive Graph Growth
# ---------------------------------------------------------------------------

def plot_graph_growth(df: pd.DataFrame, output_dir: str) -> Optional[str]:
    """
    Show how the knowledge graph grows across iterations: nodes, edges,
    and communities, alongside modularity.

    Returns:
        Path to saved figure, or None on failure.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    required_cols = {'iteration', 'nodes', 'edges', 'communities', 'modularity'}
    if not required_cols.issubset(df.columns):
        return None

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    iterations = df['iteration'].values

    # Bar chart for nodes and edges
    x = np.arange(len(iterations))
    width = 0.3

    ax1.bar(x - width, df['nodes'], width, label='Nodes', color=COLORS['entity'],
            edgecolor='white', linewidth=0.8, alpha=0.85)
    ax1.bar(x, df['edges'], width, label='Edges', color=COLORS['subcommunity'],
            edgecolor='white', linewidth=0.8, alpha=0.85)
    ax1.bar(x + width, df['communities'], width, label='Communities', color=COLORS['accent'],
            edgecolor='white', linewidth=0.8, alpha=0.85)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Iter {int(it)}' for it in iterations])
    ax1.legend(loc='upper left', framealpha=0.9)

    # Overlay modularity on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, df['modularity'], color=COLORS['secondary'], marker='D',
             linewidth=2.5, label='Modularity (Q)', zorder=5)
    ax2.set_ylabel('Modularity (Q)', color=COLORS['secondary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['secondary'])
    ax2.legend(loc='upper right', framealpha=0.9)

    ax1.set_title('Knowledge Graph Growth and Community Structure Evolution')

    plt.tight_layout()
    out_path = str(Path(output_dir) / "thesis_graph_growth.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved graph growth plot: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 9: Degree Distribution (Scale-Free Analysis)
# ---------------------------------------------------------------------------

def plot_degree_distribution_comparison(output_dir: str) -> Optional[str]:
    """
    Generate Log-Log CCDF plot of degree distributions to demonstrate scale-free properties.
    
    Fits a power-law distribution to the node degrees of the entity-relation graphs
    for each iteration. Scale-free networks (power law) are characteristic of
    robust, complex systems (small-world, hub-and-spoke).
    
    Returns:
        Path to saved figure, or None on failure.
    """
    if not MATPLOTLIB_AVAILABLE or not NETWORK_ANALYSIS_AVAILABLE:
        return None

    checkpoints_dir = Path(output_dir) / "thesis_outputs" / "checkpoints"
    if not checkpoints_dir.exists():
        logger.warning(f"Checkpoints directory not found: {checkpoints_dir}")
        return None

    graph_files = sorted(checkpoints_dir.glob("iteration_*_graph.graphml"))
    if not graph_files:
        logger.warning("No graph checkpoints found for degree distribution analysis.")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use iteration-specific colors if available, or fallback to palette
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    has_data = False
    
    for i, filepath in enumerate(graph_files):
        try:
            # Extract iteration number
            match = re.search(r"iteration_(\d+)", filepath.stem)
            iteration = int(match.group(1)) if match else i + 1
            
            G = nx.read_graphml(str(filepath))
            degrees = [d for n, d in G.degree()]
            degrees = [d for d in degrees if d > 0] # powerlaw requires > 0
            
            if not degrees:
                continue
                
            has_data = True
            
            # Fit power law
            fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
            
            # Determine color based on iteration index/scheme
            color_idx = (iteration - 1) % len(COLORS) # Simple rotation
            # Or use a specific map if defined, for now use standard rotation
            color = plt.cm.viridis(0.2 + 0.6 * (i / max(1, len(graph_files)-1)))
            
            marker = markers[i % len(markers)]
            
            # Plot Data CCDF (Empirical)
            fit.plot_ccdf(
                ax=ax, color=color, linewidth=0, 
                marker=marker, markersize=6, alpha=0.6, 
                label=f'Iter {iteration} (Data)'
            )
            
            # Plot Fit Line
            fit.power_law.plot_ccdf(
                ax=ax, color=color, linestyle='--', linewidth=2, 
                label=f'Iter {iteration} Fit ($\\alpha={fit.alpha:.2f}$)'
            )
            
        except Exception as e:
            logger.error(f"Failed to process {filepath.name} for degree distribution: {e}")
            continue

    if not has_data:
        plt.close(fig)
        return None

    ax.set_xlabel('Degree ($k$)')
    ax.set_ylabel('CCDF $P(K \geq k)$')
    ax.set_title('Degree Distribution Scale-Free Analysis')
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.grid(True, which="minor", ls=":", alpha=0.1)
    
    plt.tight_layout()
    out_path = str(Path(output_dir) / "thesis_degree_distribution.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved degree distribution plot: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def generate_all_thesis_plots(output_dir: str) -> Dict[str, Optional[str]]:
    """
    Generate all thesis-quality plots from experiment data in the output directory.

    Reads:
        - {output_dir}/iterative_experiment_results.csv
        - {output_dir}/iteration_*_report.json

    Writes:
        - thesis_modularity_vs_separation.png
        - thesis_silhouette_dashboard.png
        - thesis_silhouette_distributions.png
        - thesis_pca_scree.png
        - thesis_statistical_summary.png
        - thesis_manova_components.png
        - thesis_graph_growth.png

    Args:
        output_dir: Path to the output directory containing experiment data.

    Returns:
        Dictionary mapping figure name to file path (or None if generation failed).
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib is required for thesis plots.")
        return {}

    _apply_thesis_style()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating thesis plots from: {output_dir}")

    # Load data
    df = _load_csv(output_dir)
    reports = _load_iteration_reports(output_dir)

    if df is None and not reports:
        logger.error("No experiment data found. Cannot generate plots.")
        return {}

    results = {}

    # Generate each figure
    if df is not None:
        results['modularity_vs_separation'] = plot_modularity_vs_separation(df, output_dir)
        results['graph_growth'] = plot_graph_growth(df, output_dir)

    if df is not None and reports:
        results['silhouette_dashboard'] = plot_silhouette_dashboard(df, reports, output_dir)
        results['statistical_summary'] = plot_statistical_summary(df, reports, output_dir)

    if reports:
        results['pca_scree'] = plot_pca_scree(reports, output_dir)
        results['multivariate_anova_components'] = plot_manova_components(reports, output_dir)

    # Degree Distribution Analysis (needs checkpoints)
    results['degree_distribution'] = plot_degree_distribution_comparison(output_dir)

    diagnostics_dir = output_path / "thesis_outputs" / "diagnostics"
    if diagnostics_dir.exists():
        results['silhouette_distributions'] = plot_silhouette_distributions(diagnostics_dir, output_dir)

    # Summary
    generated = {k: v for k, v in results.items() if v is not None}
    failed = {k for k, v in results.items() if v is None}

    logger.info(f"Thesis plots generated: {len(generated)}/{len(results)}")
    if failed:
        logger.warning(f"Failed to generate: {failed}")

    # Write manifest
    manifest_path = str(output_path / "thesis_plots_manifest.json")
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump({
                'generated_at': pd.Timestamp.now().isoformat(),
                'output_dir': str(output_dir),
                'figures': results,
                'total_generated': len(generated),
                'total_attempted': len(results),
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Plot manifest saved: {manifest_path}")
    except Exception as e:
        logger.error(f"Failed to save manifest: {e}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    configure_logging()

    parser = argparse.ArgumentParser(
        description="Generate thesis-quality plots from GraphGen experiment data."
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="/app/output",
        help="Path to the output directory containing experiment CSV and JSON reports."
    )
    args = parser.parse_args()

    results = generate_all_thesis_plots(args.output_dir)

    if results:
        generated_count = len([v for v in results.values() if v])
        logger.info("Generated %d thesis plots.", generated_count)
        for name, path in results.items():
            status = path if path else "FAILED"
            logger.info("Plot result: %s=%s", name, status)
    else:
        logger.error("No plots generated. Check logs for errors.")
        sys.exit(1)
