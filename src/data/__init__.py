"""
Data handling module for Brain Connectivity Prediction.

Provides functionality for:
    - Downloading datasets
    - Loading connectivity matrices
    - Generating synthetic data
    - Data preprocessing
"""

from .download import download_data
from .load_connectivity import load_connectivity_matrix, load_all_subjects
from .generate_synthetic import generate_synthetic_data
from .preprocess import preprocess_connectivity, create_hybrid_connectivity

__all__ = [
    "download_data",
    "load_connectivity_matrix",
    "load_all_subjects",
    "generate_synthetic_data",
    "preprocess_connectivity",
    "create_hybrid_connectivity"
]
