"""Shared academic plot styling for all graphgen visualizations."""

import logging

logger = logging.getLogger(__name__)

# Colorblind-safe academic palette
COLORS = {
    'primary': '#2c3e50',
    'secondary': '#e74c3c',
    'accent': '#27ae60',
    'baseline': '#95a5a6',
    'modularity': '#2c3e50',
    'separation': '#c0392b',
    'overlap': '#e67e22',
    'silhouette_pos': '#27ae60',
    'silhouette_neg': '#e74c3c',
    'entity': '#3498db',
    'community': '#2ecc71',
    'subcommunity': '#9b59b6',
    'grid': '#ecf0f1',
}

COMM_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

# Centrality measure display palette
CENTRALITY_PALETTE = ['#2c3e50', '#2980b9', '#27ae60', '#e67e22', '#8e44ad']


def apply_thesis_style():
    """Configure matplotlib for thesis-quality output."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'figure.figsize': (8, 5),
    })

    sns.set_context("paper", font_scale=1.1)
    sns.set_style("whitegrid", {
        'axes.edgecolor': '.3',
        'grid.color': COLORS['grid'],
    })


def truncate_label(text: str, max_len: int = 25) -> str:
    """Truncate a label for plot readability."""
    if not text:
        return ""
    text = str(text).strip()
    if len(text) > max_len:
        return text[:max_len - 1] + "…"
    return text
