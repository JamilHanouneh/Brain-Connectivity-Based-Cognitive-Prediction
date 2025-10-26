"""
Plotting functions for model performance results.

Creates publication-quality visualizations of prediction performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")


def plot_performance_violin(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    metric: str = "r2_scores",
    output_path: Optional[str] = None,
    figsize: tuple = (12, 6),
    colors: Optional[Dict[str, str]] = None
) -> plt.Figure:
    """
    Create violin plot of model performance.
    
    Parameters
    ----------
    results_dict : dict
        Nested dictionary: {conn_type: {score_name: results}}
    metric : str
        Metric to plot
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    colors : dict, optional
        Color mapping for connectivity types
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    logger.info(f"Creating violin plot for {metric}")
    
    # Prepare data
    data_list = []
    conn_types = []
    score_names = []
    
    for conn_type in results_dict.keys():
        for score_name in results_dict[conn_type].keys():
            scores = results_dict[conn_type][score_name][metric]
            data_list.extend(scores)
            conn_types.extend([conn_type] * len(scores))
            score_names.extend([score_name] * len(scores))
    
    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame({
        'Score': data_list,
        'Connectivity': conn_types,
        'Cognitive': score_names
    })
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if colors is None:
        colors = {'FC': '#FF6B6B', 'SC': '#4ECDC4', 'HC': '#95E1D3'}
    
    sns.violinplot(
        data=df,
        x='Cognitive',
        y='Score',
        hue='Connectivity',
        palette=colors,
        ax=ax,
        inner='box'
    )
    
    ax.set_xlabel('Cognitive Score', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(title='Connectivity Type', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved violin plot to {output_path}")
    
    return fig


def plot_performance_boxplot(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    metric: str = "r2_scores",
    output_path: Optional[str] = None,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Create boxplot of model performance.
    
    Parameters
    ----------
    results_dict : dict
        Nested dictionary: {conn_type: {score_name: results}}
    metric : str
        Metric to plot
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    logger.info(f"Creating boxplot for {metric}")
    
    # Prepare data
    data = []
    labels = []
    
    for conn_type in results_dict.keys():
        for score_name in results_dict[conn_type].keys():
            scores = results_dict[conn_type][score_name][metric]
            data.append(scores)
            labels.append(f"{conn_type}\n{score_name}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True
    )
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Model Performance Across Splits', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved boxplot to {output_path}")
    
    return fig


def plot_scatter_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs. Actual",
    output_path: Optional[str] = None,
    figsize: tuple = (8, 8)
) -> plt.Figure:
    """
    Create scatter plot of predicted vs. actual values.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    title : str
        Plot title
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score
    
    logger.info(f"Creating scatter plot: {title}")
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    corr, p_value = pearsonr(y_true, y_pred)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=50, edgecolors='k', linewidths=0.5)
    
    # Identity line
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, 'k--', alpha=0.5, linewidth=2, label='Perfect prediction')
    
    # Regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(y_true, p(y_true), "r-", alpha=0.8, linewidth=2, label='Regression line')
    
    ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add metrics as text
    textstr = f'$R^2$ = {r2:.3f}\nr = {corr:.3f}\np < {p_value:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved scatter plot to {output_path}")
    
    return fig


def plot_comparison_heatmap(
    comparison_results: Dict,
    metric: str = "p_value",
    output_path: Optional[str] = None,
    figsize: tuple = (12, 10)
) -> plt.Figure:
    """
    Create heatmap of model comparison results.
    
    Parameters
    ----------
    comparison_results : dict
        Pairwise comparison results
    metric : str
        Metric to display ('p_value', 'mean_difference', 'cohens_d')
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    logger.info(f"Creating comparison heatmap for {metric}")
    
    # Extract model names
    model_names = set()
    for (model1, model2) in comparison_results.keys():
        model_names.add(model1)
        model_names.add(model2)
    model_names = sorted(list(model_names))
    
    # Create matrix
    n_models = len(model_names)
    matrix = np.zeros((n_models, n_models))
    matrix[:] = np.nan
    
    for (model1, model2), results in comparison_results.items():
        i = model_names.index(model1)
        j = model_names.index(model2)
        matrix[i, j] = results[metric]
        if metric == "mean_difference":
            matrix[j, i] = -results[metric]  # Symmetric with sign
        else:
            matrix[j, i] = results[metric]  # Symmetric
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if metric == "p_value":
        cmap = "RdYlGn_r"
        vmin, vmax = 0, 0.1
    else:
        cmap = "RdBu_r"
        vmax = np.nanmax(np.abs(matrix))
        vmin = -vmax
    
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_yticklabels(model_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    
    # Add values as text
    for i in range(n_models):
        for j in range(n_models):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison heatmap to {output_path}")
    
    return fig
