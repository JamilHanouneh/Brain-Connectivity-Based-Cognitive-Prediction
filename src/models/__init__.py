"""
Machine learning models for cognitive prediction.

Implements ridge regression with nested cross-validation,
permutation testing, and feature importance extraction.
"""

from .ridge_prediction import (
    RidgePredictionModel,
    nested_cross_validation,
    train_test_prediction
)
from .permutation_test import (
    permutation_test,
    run_permutation_tests
)
from .feature_importance import (
    compute_activation_patterns,
    compute_feature_importance
)

__all__ = [
    "RidgePredictionModel",
    "nested_cross_validation",
    "train_test_prediction",
    "permutation_test",
    "run_permutation_tests",
    "compute_activation_patterns",
    "compute_feature_importance"
]
