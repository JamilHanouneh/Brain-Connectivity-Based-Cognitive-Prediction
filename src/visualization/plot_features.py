"""
Feature importance visualization functions.

Creates plots for connectivity feature importance and brain network patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def plot_feature_importance_matrix(
    importance_matrix: np.ndarray,
    title: str = "Feature Importance",
    region_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 10),
    cmap: str = "RdBu_r",
    show_colorbar: bool = True
) -> plt.Figure:
    """
    Plot feature importance as a connectivity matrix.
    
    Parameters
    ----------
    importance_matrix : np.ndarray
        Importance matrix (n_regions x n_regions)
    title : str
        Plot title
    region_names : list of str, optional
        Names of brain regions
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    show_colorbar : bool
        Whether to show colorbar
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    logger.info(f"Creating feature importance matrix plot: {title}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Symmetric colormap limits
    vmax = np.abs(importance_matrix).max()
    vmin = -vmax
    
    # Plot heatmap
    im = ax.imshow(
        importance_matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
        interpolation='nearest'
    )
    
    # Set ticks
    n_regions = importance_matrix.shape[0]
    
    if region_names is not None and len(region_names) == n_regions:
        # Show region names if provided and not too many
        if n_regions <= 20:
            ax.set_xticks(np.arange(n_regions))
            ax.set_yticks(np.arange(n_regions))
            ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(region_names, fontsize=8)
        else:
            # Too many regions - show every nth
            step = n_regions // 10
            tick_positions = np.arange(0, n_regions, step)
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels([region_names[i] for i in tick_positions], 
                              rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels([region_names[i] for i in tick_positions], fontsize=8)
    else:
        # Show region indices
        if n_regions <= 30:
            ax.set_xticks(np.arange(n_regions))
            ax.set_yticks(np.arange(n_regions))
            ax.set_xticklabels(np.arange(n_regions), fontsize=8)
            ax.set_yticklabels(np.arange(n_regions), fontsize=8)
    
    # Colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Importance', fontsize=12, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Brain Region', fontsize=12, fontweight='bold')
    ax.set_ylabel('Brain Region', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance matrix to {output_path}")
    
    return fig


def plot_regional_importance(
    regional_importance: np.ndarray,
    region_names: Optional[List[str]] = None,
    title: str = "Regional Importance",
    output_path: Optional[str] = None,
    figsize: tuple = (14, 6),
    top_n: Optional[int] = None
) -> plt.Figure:
    """
    Plot importance scores aggregated by brain region.
    
    Parameters
    ----------
    regional_importance : np.ndarray
        Importance scores per region (n_regions,)
    region_names : list of str, optional
        Names of brain regions
    title : str
        Plot title
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    top_n : int, optional
        Show only top N regions
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    logger.info(f"Creating regional importance plot: {title}")
    
    n_regions = len(regional_importance)
    
    # Sort by importance
    sorted_idx = np.argsort(regional_importance)[::-1]
    
    if top_n is not None and top_n < n_regions:
        sorted_idx = sorted_idx[:top_n]
    
    sorted_importance = regional_importance[sorted_idx]
    
    # Create labels
    if region_names is not None:
        sorted_labels = [region_names[i] for i in sorted_idx]
    else:
        sorted_labels = [f"Region {i}" for i in sorted_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color bars by importance magnitude
    colors = plt.cm.RdYlBu_r(np.linspace(0.3, 0.9, len(sorted_importance)))
    
    bars = ax.barh(np.arange(len(sorted_importance)), sorted_importance, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(np.arange(len(sorted_importance)))
    ax.set_yticklabels(sorted_labels, fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Brain Region', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Invert y-axis to show highest at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved regional importance plot to {output_path}")
    
    return fig


def plot_top_connections(
    top_connections: List[tuple],
    title: str = "Top Important Connections",
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Plot top important connections as a bar chart.
    
    Parameters
    ----------
    top_connections : list of tuple
        List of (region1, region2, importance) tuples
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
    logger.info(f"Creating top connections plot: {title}")
    
    # Extract data
    connection_labels = [f"{r1} - {r2}" for r1, r2, _ in top_connections]
    importance_values = [imp for _, _, imp in top_connections]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by positive/negative importance
    colors = ['#FF6B6B' if imp > 0 else '#4ECDC4' for imp in importance_values]
    
    bars = ax.barh(np.arange(len(connection_labels)), importance_values, color=colors, 
                   edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(np.arange(len(connection_labels)))
    ax.set_yticklabels(connection_labels, fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Brain Connection', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Invert y-axis to show highest at top
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', edgecolor='black', label='Positive'),
        Patch(facecolor='#4ECDC4', edgecolor='black', label='Negative')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved top connections plot to {output_path}")
    
    return fig


def plot_network_importance(
    importance_matrix: np.ndarray,
    network_labels: np.ndarray,
    network_names: Optional[List[str]] = None,
    title: str = "Network-Level Importance",
    output_path: Optional[str] = None,
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot importance aggregated by brain networks.
    
    Parameters
    ----------
    importance_matrix : np.ndarray
        Importance matrix (n_regions x n_regions)
    network_labels : np.ndarray
        Network assignment for each region (n_regions,)
    network_names : list of str, optional
        Names of networks
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
    logger.info(f"Creating network importance plot: {title}")
    
    # Get unique networks
    unique_networks = np.unique(network_labels)
    n_networks = len(unique_networks)
    
    # Create network importance matrix
    network_importance = np.zeros((n_networks, n_networks))
    
    for i, net_i in enumerate(unique_networks):
        for j, net_j in enumerate(unique_networks):
            # Get connections between these networks
            mask_i = network_labels == net_i
            mask_j = network_labels == net_j
            
            # Sum absolute importance of connections
            connections = importance_matrix[np.ix_(mask_i, mask_j)]
            network_importance[i, j] = np.sum(np.abs(connections))
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(
        network_importance,
        cmap='YlOrRd',
        aspect='auto',
        interpolation='nearest'
    )
    
    # Set ticks
    ax.set_xticks(np.arange(n_networks))
    ax.set_yticks(np.arange(n_networks))
    
    if network_names is not None:
        ax.set_xticklabels(network_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(network_names, fontsize=10)
    else:
        ax.set_xticklabels([f"Network {i+1}" for i in range(n_networks)], 
                          rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels([f"Network {i+1}" for i in range(n_networks)], fontsize=10)
    
    # Add values
    for i in range(n_networks):
        for j in range(n_networks):
            text = ax.text(j, i, f'{network_importance[i, j]:.1f}',
                         ha="center", va="center", color="black", fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Total Absolute Importance', fontsize=12, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Network', fontsize=12, fontweight='bold')
    ax.set_ylabel('Network', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved network importance plot to {output_path}")
    
    return fig
