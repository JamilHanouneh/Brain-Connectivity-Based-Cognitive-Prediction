"""
Performance metrics for cognitive prediction.

Implements evaluation metrics from Dhamala et al. (2021).
"""

import numpy as np
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination) score.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        R² score
    """
    return r2_score(y_true, y_pred)


def compute_explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute explained variance score.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        Explained variance
    """
    return explained_variance_score(y_true, y_pred)


def compute_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "pearson"
) -> Tuple[float, float]:
    """
    Compute correlation between true and predicted values.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    method : str
        Correlation method ('pearson' or 'spearman')
    
    Returns
    -------
    tuple
        (correlation, p_value)
    """
    if method == "pearson":
        return pearsonr(y_true, y_pred)
    elif method == "spearman":
        return spearmanr(y_true, y_pred)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute root mean squared error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        RMSE
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean absolute error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        MAE
    """
    return mean_absolute_error(y_true, y_pred)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    dict
        Dictionary of all metrics
    """
    # R² and explained variance
    r2 = compute_r2_score(y_true, y_pred)
    exp_var = compute_explained_variance(y_true, y_pred)
    
    # Correlation
    pearson_r, pearson_p = compute_correlation(y_true, y_pred, method='pearson')
    spearman_r, spearman_p = compute_correlation(y_true, y_pred, method='spearman')
    
    # Error metrics
    rmse = compute_rmse(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    
    metrics = {
        'r2': r2,
        'explained_variance': exp_var,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'rmse': rmse,
        'mae': mae
    }
    
    return metrics


def aggregate_metrics_across_splits(
    metrics_list: list
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple train/test splits.
    
    Parameters
    ----------
    metrics_list : list
        List of metric dictionaries from each split
    
    Returns
    -------
    dict
        Aggregated metrics with mean and std
    """
    aggregated = {}
    
    # Get metric names from first dictionary
    metric_names = list(metrics_list[0].keys())
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in metrics_list]
        aggregated[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'all_values': np.array(values)
        }
    
    return aggregated


def compute_prediction_intervals(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute prediction intervals.
    
    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values
    residuals : np.ndarray
        Prediction residuals
    confidence_level : float
        Confidence level
    
    Returns
    -------
    tuple
        (lower_bounds, upper_bounds)
    """
    from scipy import stats
    
    # Estimate residual standard deviation
    residual_std = np.std(residuals)
    
    # Get critical value
    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha / 2)
    
    # Compute intervals
    margin = z_critical * residual_std
    lower_bounds = y_pred - margin
    upper_bounds = y_pred + margin
    
    return lower_bounds, upper_bounds
