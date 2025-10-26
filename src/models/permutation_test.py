"""
Permutation testing for statistical significance.

Implements permutation tests to determine if prediction performance
is significantly better than chance.
"""

import numpy as np
from typing import Callable, Dict, Optional
from tqdm import tqdm
import logging
from sklearn.metrics import r2_score
from scipy.stats import percentileofscore

logger = logging.getLogger(__name__)


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    prediction_func: Callable,
    n_permutations: int = 1000,
    metric: str = 'r2',
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Perform permutation test for prediction significance.
    
    Parameters
    ----------
    X : np.ndarray
        Features (n_samples x n_features)
    y : np.ndarray
        Target values (n_samples,)
    prediction_func : callable
        Function that takes (X, y) and returns predictions and actual values
    n_permutations : int
        Number of permutations
    metric : str
        Metric to use ('r2' or 'correlation')
    random_seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Whether to show progress bar
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'true_score': Score on true labels
        - 'null_scores': Scores on permuted labels
        - 'p_value': Permutation p-value
        - 'z_score': Z-score relative to null distribution
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    logger.info(f"Starting permutation test with {n_permutations} permutations...")
    
    # Get true score
    y_pred, y_true = prediction_func(X, y)
    
    if metric == 'r2':
        true_score = r2_score(y_true, y_pred)
    elif metric == 'correlation':
        true_score = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    logger.info(f"True score ({metric}): {true_score:.4f}")
    
    # Permutation testing
    null_scores = []
    
    iterator = range(n_permutations)
    if verbose:
        iterator = tqdm(iterator, desc="Permutation Test")
    
    for perm_idx in iterator:
        # Permute labels
        y_permuted = np.random.permutation(y)
        
        # Get score on permuted data
        y_pred_perm, y_true_perm = prediction_func(X, y_permuted)
        
        if metric == 'r2':
            perm_score = r2_score(y_true_perm, y_pred_perm)
        elif metric == 'correlation':
            perm_score = np.corrcoef(y_true_perm, y_pred_perm)[0, 1]
        
        null_scores.append(perm_score)
    
    null_scores = np.array(null_scores)
    
    # Calculate p-value
    p_value = (np.sum(null_scores >= true_score) + 1) / (n_permutations + 1)
    
    # Calculate z-score
    null_mean = np.mean(null_scores)
    null_std = np.std(null_scores)
    if null_std > 0:
        z_score = (true_score - null_mean) / null_std
    else:
        z_score = np.inf if true_score > null_mean else -np.inf
    
    results = {
        'true_score': true_score,
        'null_scores': null_scores,
        'null_mean': null_mean,
        'null_std': null_std,
        'p_value': p_value,
        'z_score': z_score
    }
    
    logger.info(f"Permutation test complete:")
    logger.info(f"  p-value = {p_value:.4f}")
    logger.info(f"  z-score = {z_score:.2f}")
    
    return results


def run_permutation_tests(
    X_dict: Dict[str, np.ndarray],
    y_dict: Dict[str, np.ndarray],
    prediction_func: Callable,
    n_permutations: int = 1000,
    random_seed: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run permutation tests for multiple connectivity types and cognitive scores.
    
    Parameters
    ----------
    X_dict : dict
        Dictionary of features: {'FC': X_fc, 'SC': X_sc, 'HC': X_hc}
    y_dict : dict
        Dictionary of targets: {'Crystallized': y_cryst, 'Fluid': y_fluid, ...}
    prediction_func : callable
        Prediction function
    n_permutations : int
        Number of permutations
    random_seed : int, optional
        Random seed
    
    Returns
    -------
    dict
        Nested dictionary of results for each combination
    """
    logger.info("=" * 80)
    logger.info("Running Permutation Tests for All Models")
    logger.info("=" * 80)
    
    results = {}
    
    for conn_type, X in X_dict.items():
        results[conn_type] = {}
        
        for score_name, y in y_dict.items():
            logger.info(f"\nTesting {conn_type} -> {score_name}")
            
            perm_results = permutation_test(
                X=X,
                y=y,
                prediction_func=prediction_func,
                n_permutations=n_permutations,
                random_seed=random_seed,
                verbose=True
            )
            
            results[conn_type][score_name] = perm_results
    
    logger.info("\n" + "=" * 80)
    logger.info("Permutation Testing Complete")
    logger.info("=" * 80)
    
    return results


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> tuple:
    """
    Calculate bootstrap confidence interval.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    statistic : callable
        Statistic function to compute
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_seed : int, optional
        Random seed
    
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n_samples, replace=True)
        bootstrap_stats.append(statistic(sample))
    
    # Calculate percentiles
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return lower, upper
