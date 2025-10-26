"""
Evaluation module for model performance assessment.

Provides metrics computation and statistical model comparison.
"""

from .metrics import (
    compute_r2_score,
    compute_explained_variance,
    compute_correlation,
    compute_all_metrics
)
from .compare_models import (
    compare_two_models,
    compare_all_models,
    exact_test,
    fdr_correction
)

__all__ = [
    "compute_r2_score",
    "compute_explained_variance",
    "compute_correlation",
    "compute_all_metrics",
    "compare_two_models",
    "compare_all_models",
    "exact_test",
    "fdr_correction"
]
