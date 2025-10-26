"""
Feature importance extraction using activation patterns.

Implements the Haufe et al. (2014) method for computing activation patterns
from backward models (predictive models).
"""

import numpy as np
from typing import Optional, Dict, List  # ← FIXED: Added Dict import
import logging

logger = logging.getLogger(__name__)


def compute_activation_patterns(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray
) -> np.ndarray:
    """
    Compute activation patterns from model coefficients.
    
    This implements Haufe et al. (2014) method to transform backward model
    weights into interpretable forward model (activation) patterns.
    
    Reference:
    Haufe, S., et al. (2014). On the interpretation of weight vectors of linear 
    models in multivariate neuroimaging. NeuroImage, 87, 96-110.
    
    Parameters
    ----------
    X : np.ndarray
        Features (n_samples x n_features)
    y : np.ndarray
        Target values (n_samples,)
    coefficients : np.ndarray
        Model coefficients (n_features,)
    
    Returns
    -------
    np.ndarray
        Activation patterns (n_features,)
    """
    # Center data
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()
    
    # Compute covariance matrices
    cov_X = np.cov(X_centered.T)  # Feature covariance
    cov_y = np.var(y_centered)     # Target variance
    
    # Compute activation patterns
    # A = Cov(X) * w / Var(y_pred)
    # where y_pred = X * w
    
    # Predicted values
    y_pred = X_centered @ coefficients
    var_y_pred = np.var(y_pred)
    
    if var_y_pred > 0:
        activation_patterns = (cov_X @ coefficients) / var_y_pred
    else:
        activation_patterns = np.zeros_like(coefficients)
    
    return activation_patterns


def compute_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    coefficients_array: np.ndarray,
    method: str = "haufe"
) -> Dict[str, np.ndarray]:
    """
    Compute feature importance across multiple models.
    
    Parameters
    ----------
    X : np.ndarray
        Features (n_samples x n_features)
    y : np.ndarray
        Target values (n_samples,)
    coefficients_array : np.ndarray
        Array of coefficients from multiple models (n_models x n_features)
    method : str
        Method to use ('haufe', 'coefficients', 'abs_coefficients')
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'importance': Mean importance scores
        - 'std': Standard deviation of importance
        - 'all_importance': All importance scores
    """
    logger.info(f"Computing feature importance using method: {method}")
    
    n_models = coefficients_array.shape[0]
    all_importance = []
    
    for i in range(n_models):
        coefficients = coefficients_array[i]
        
        if method == "haufe":
            importance = compute_activation_patterns(X, y, coefficients)
        elif method == "coefficients":
            importance = coefficients
        elif method == "abs_coefficients":
            importance = np.abs(coefficients)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        all_importance.append(importance)
    
    all_importance = np.array(all_importance)
    
    results = {
        'importance': np.mean(all_importance, axis=0),
        'std': np.std(all_importance, axis=0),
        'all_importance': all_importance
    }
    
    logger.info(f"Feature importance computed for {n_models} models")
    
    return results


def feature_vector_to_matrix(
    feature_vector: np.ndarray,
    n_regions: int,
    symmetric: bool = True
) -> np.ndarray:
    """
    Convert feature vector back to connectivity matrix.
    
    Parameters
    ----------
    feature_vector : np.ndarray
        Vectorized upper triangle (n_features,)
    n_regions : int
        Number of brain regions
    symmetric : bool
        Whether to make matrix symmetric
    
    Returns
    -------
    np.ndarray
        Connectivity matrix (n_regions x n_regions)
    """
    matrix = np.zeros((n_regions, n_regions))
    
    # Fill upper triangle
    upper_indices = np.triu_indices(n_regions, k=1)
    matrix[upper_indices] = feature_vector
    
    if symmetric:
        # Mirror to lower triangle
        matrix = matrix + matrix.T
    
    return matrix


def identify_top_connections(
    importance_matrix: np.ndarray,
    n_top: int = 10,
    region_names: Optional[List[str]] = None
) -> List[tuple]:
    """
    Identify top important connections.
    
    Parameters
    ----------
    importance_matrix : np.ndarray
        Importance matrix (n_regions x n_regions)
    n_top : int
        Number of top connections to identify
    region_names : list, optional
        Names of brain regions
    
    Returns
    -------
    list
        List of tuples: (region1, region2, importance_value)
    """
    # Get absolute importance
    abs_importance = np.abs(importance_matrix)
    
    # Get upper triangle indices (excluding diagonal)
    upper_indices = np.triu_indices_from(abs_importance, k=1)
    
    # Get importance values
    importance_values = abs_importance[upper_indices]
    
    # Sort by importance
    sorted_idx = np.argsort(importance_values)[::-1]
    
    # Get top connections
    top_connections = []
    
    for idx in sorted_idx[:n_top]:
        i = upper_indices[0][idx]
        j = upper_indices[1][idx]
        value = importance_matrix[i, j]
        
        if region_names is not None:
            region1 = region_names[i]
            region2 = region_names[j]
        else:
            region1 = f"Region_{i}"
            region2 = f"Region_{j}"
        
        top_connections.append((region1, region2, value))
    
    return top_connections


def compute_regional_importance(
    importance_matrix: np.ndarray,
    aggregation: str = "sum"
) -> np.ndarray:
    """
    Aggregate connection importance by brain region.
    
    Parameters
    ----------
    importance_matrix : np.ndarray
        Importance matrix (n_regions x n_regions)
    aggregation : str
        Aggregation method ('sum', 'mean', 'max')
    
    Returns
    -------
    np.ndarray
        Regional importance scores (n_regions,)
    """
    abs_importance = np.abs(importance_matrix)
    
    if aggregation == "sum":
        regional_importance = np.sum(abs_importance, axis=1)
    elif aggregation == "mean":
        regional_importance = np.mean(abs_importance, axis=1)
    elif aggregation == "max":
        regional_importance = np.max(abs_importance, axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return regional_importance
