"""
Brain Connectivity-Based Cognitive Prediction Package

This package implements the methodology from:
Dhamala, E., et al. (2021). Distinct functional and structural connections 
predict crystallised and fluid cognition in healthy adults. 
Brain Structure and Function, 226, 1669-1691.

Modules:
    - data: Data loading, preprocessing, and generation
    - models: Ridge regression, permutation testing, feature importance
    - evaluation: Performance metrics and model comparison
    - visualization: Plotting and report generation
    - utils: Logging, I/O, and utility functions
"""

__version__ = "1.0.0"
__author__ = "Based on Dhamala et al. (2021)"

# Package-level imports for convenience
from . import data
from . import models
from . import evaluation
from . import visualization
from . import utils

__all__ = [
    "data",
    "models",
    "evaluation",
    "visualization",
    "utils"
]
