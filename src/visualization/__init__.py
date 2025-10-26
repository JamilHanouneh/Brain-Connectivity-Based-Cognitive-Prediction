"""
Visualization module for result plots and reports.

Provides plotting functions and HTML report generation.
"""

from .plot_results import (
    plot_performance_violin,
    plot_performance_boxplot,
    plot_scatter_prediction,
    plot_comparison_heatmap
)
from .plot_features import (
    plot_feature_importance_matrix,
    plot_regional_importance,
    plot_top_connections
)
from .report_generator import (
    generate_html_report,
    create_results_summary
)

__all__ = [
    "plot_performance_violin",
    "plot_performance_boxplot",
    "plot_scatter_prediction",
    "plot_comparison_heatmap",
    "plot_feature_importance_matrix",
    "plot_regional_importance",
    "plot_top_connections",
    "generate_html_report",
    "create_results_summary"
]
