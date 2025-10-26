"""
Preprocessing functions for connectivity matrices.

Implements preprocessing steps from Dhamala et al. (2021):
- Fisher z-transformation for FC
- Log transformation for SC
- Normalization
- Hybrid connectivity creation
"""

import numpy as np
from typing import Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def fisher_z_transform(correlation_matrix: np.ndarray) -> np.ndarray:
    """
    Apply Fisher z-transformation to correlation matrix.
    
    Parameters
    ----------
    correlation_matrix : np.ndarray
        Correlation matrix
    
    Returns
    -------
    np.ndarray
        Fisher z-transformed matrix
    """
    # Clip to avoid numerical issues with arctanh
    correlation_matrix = np.clip(correlation_matrix, -0.9999, 0.9999)
    return np.arctanh(correlation_matrix)


def inverse_fisher_z(z_matrix: np.ndarray) -> np.ndarray:
    """
    Inverse Fisher z-transformation.
    
    Parameters
    ----------
    z_matrix : np.ndarray
        Fisher z-transformed matrix
    
    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    return np.tanh(z_matrix)


def log_transform(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Apply log transformation to structural connectivity.
    
    Parameters
    ----------
    matrix : np.ndarray
        Structural connectivity matrix
    epsilon : float
        Small constant to avoid log(0)
    
    Returns
    -------
    np.ndarray
        Log-transformed matrix
    """
    return np.log(matrix + epsilon)


def normalize_matrix(matrix: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize connectivity matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Connectivity matrix
    method : str
        Normalization method ('zscore', 'minmax', 'robust')
    
    Returns
    -------
    np.ndarray
        Normalized matrix
    """
    if method == "zscore":
        # Z-score normalization
        mean = np.mean(matrix)
        std = np.std(matrix)
        if std > 0:
            return (matrix - mean) / std
        else:
            return matrix - mean
    
    elif method == "minmax":
        # Min-max normalization to [0, 1]
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        if max_val > min_val:
            return (matrix - min_val) / (max_val - min_val)
        else:
            return matrix - min_val
    
    elif method == "robust":
        # Robust normalization using median and IQR
        median = np.median(matrix)
        q75, q25 = np.percentile(matrix, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            return (matrix - median) / iqr
        else:
            return matrix - median
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_connectivity(
    matrices: np.ndarray,
    connectivity_type: str,
    config: dict
) -> np.ndarray:
    """
    Preprocess connectivity matrices according to configuration.
    
    Parameters
    ----------
    matrices : np.ndarray
        Connectivity matrices (n_subjects x n_regions x n_regions)
    connectivity_type : str
        Type of connectivity ('FC' or 'SC')
    config : dict
        Preprocessing configuration
    
    Returns
    -------
    np.ndarray
        Preprocessed connectivity matrices
    """
    logger.info(f"Preprocessing {connectivity_type} matrices...")
    
    n_subjects = matrices.shape[0]
    processed_matrices = matrices.copy()
    
    # Get type-specific config
    if connectivity_type == "FC":
        type_config = config['preprocessing']['fc']
    elif connectivity_type == "SC":
        type_config = config['preprocessing']['sc']
    else:
        raise ValueError(f"Unknown connectivity type: {connectivity_type}")
    
    # Process each subject
    for i in range(n_subjects):
        matrix = processed_matrices[i]
        
        # Remove diagonal if specified
        if type_config.get('remove_diagonal', True):
            np.fill_diagonal(matrix, 0)
        
        # Apply transformations
        if connectivity_type == "FC":
            # Fisher z-transform for functional connectivity
            if type_config.get('fisher_z_transform', True):
                matrix = fisher_z_transform(matrix)
            
            # Threshold weak connections
            threshold = type_config.get('threshold', 0.0)
            if threshold > 0:
                matrix[np.abs(matrix) < threshold] = 0
        
        elif connectivity_type == "SC":
            # Log transform for structural connectivity
            if type_config.get('log_transform', True):
                matrix = log_transform(matrix)
            
            # Normalize
            if type_config.get('normalize', True):
                matrix = normalize_matrix(matrix, method='minmax')
        
        processed_matrices[i] = matrix
    
    logger.info(f"Preprocessing complete. Shape: {processed_matrices.shape}")
    return processed_matrices


def create_hybrid_connectivity(
    fc_matrices: np.ndarray,
    sc_matrices: np.ndarray,
    method: str = "upper_lower",
    fc_weight: float = 0.5,
    sc_weight: float = 0.5
) -> np.ndarray:
    """
    Create hybrid connectivity matrices combining FC and SC.
    
    Parameters
    ----------
    fc_matrices : np.ndarray
        Functional connectivity matrices (n_subjects x n_regions x n_regions)
    sc_matrices : np.ndarray
        Structural connectivity matrices (n_subjects x n_regions x n_regions)
    method : str
        Combination method:
        - 'upper_lower': FC in upper triangle, SC in lower triangle
        - 'weighted_average': Weighted average of FC and SC
        - 'concatenate': Concatenate FC and SC features
    fc_weight : float
        Weight for FC in weighted average
    sc_weight : float
        Weight for SC in weighted average
    
    Returns
    -------
    np.ndarray
        Hybrid connectivity matrices
    """
    if fc_matrices.shape != sc_matrices.shape:
        raise ValueError("FC and SC matrices must have the same shape")
    
    logger.info(f"Creating hybrid connectivity using method: {method}")
    
    n_subjects, n_regions, _ = fc_matrices.shape
    
    if method == "upper_lower":
        # Upper triangle: FC, Lower triangle: SC, Diagonal: 0
        hc_matrices = np.zeros_like(fc_matrices)
        
        for i in range(n_subjects):
            # Upper triangle from FC (excluding diagonal)
            upper_indices = np.triu_indices(n_regions, k=1)
            hc_matrices[i][upper_indices] = fc_matrices[i][upper_indices]
            
            # Lower triangle from SC (excluding diagonal)
            lower_indices = np.tril_indices(n_regions, k=-1)
            hc_matrices[i][lower_indices] = sc_matrices[i][lower_indices]
    
    elif method == "weighted_average":
        # Weighted average of FC and SC
        hc_matrices = fc_weight * fc_matrices + sc_weight * sc_matrices
    
    elif method == "concatenate":
        # This will be handled during feature extraction
        # Return stacked array along last dimension
        hc_matrices = np.concatenate([fc_matrices, sc_matrices], axis=-1)
    
    else:
        raise ValueError(f"Unknown hybrid method: {method}")
    
    logger.info(f"Hybrid connectivity created. Shape: {hc_matrices.shape}")
    return hc_matrices


def remove_confounds(
    data: np.ndarray,
    confounds: np.ndarray
) -> np.ndarray:
    """
    Remove confounding variables from data using linear regression.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (n_samples x n_features)
    confounds : np.ndarray
        Confound variables (n_samples x n_confounds)
    
    Returns
    -------
    np.ndarray
        Residuals after removing confounds
    """
    from sklearn.linear_model import LinearRegression
    
    logger.info("Removing confounds from data...")
    
    # Fit linear model
    model = LinearRegression()
    model.fit(confounds, data)
    
    # Get residuals
    residuals = data - model.predict(confounds)
    
    return residuals
