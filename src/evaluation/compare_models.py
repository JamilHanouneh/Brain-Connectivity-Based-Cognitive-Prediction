"""
Statistical comparison of models.

Implements methods for comparing prediction performance
across different connectivity types and cognitive scores.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def exact_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Perform exact test for paired samples.
    
    Uses Wilcoxon signed-rank test for paired comparisons.
    
    Parameters
    ----------
    scores1 : np.ndarray
        Scores from model 1
    scores2 : np.ndarray
        Scores from model 2
    alternative : str
        Alternative hypothesis ('two-sided', 'less', 'greater')
    
    Returns
    -------
    tuple
        (statistic, p_value)
    """
    if len(scores1) != len(scores2):
        raise ValueError("Score arrays must have same length")
    
    # Wilcoxon signed-rank test for paired samples
    statistic, p_value = stats.wilcoxon(scores1, scores2, alternative=alternative)
    
    return statistic, p_value


def paired_t_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Perform paired t-test.
    
    Parameters
    ----------
    scores1 : np.ndarray
        Scores from model 1
    scores2 : np.ndarray
        Scores from model 2
    alternative : str
        Alternative hypothesis
    
    Returns
    -------
    tuple
        (t_statistic, p_value)
    """
    statistic, p_value = stats.ttest_rel(scores1, scores2, alternative=alternative)
    return statistic, p_value


def compare_two_models(
    scores1: np.ndarray,
    scores2: np.ndarray,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    test: str = "wilcoxon"
) -> Dict[str, float]:
    """
    Compare two models statistically.
    
    Parameters
    ----------
    scores1 : np.ndarray
        Scores from model 1
    scores2 : np.ndarray
        Scores from model 2
    model1_name : str
        Name of model 1
    model2_name : str
        Name of model 2
    test : str
        Statistical test ('wilcoxon' or 'ttest')
    
    Returns
    -------
    dict
        Comparison results
    """
    logger.info(f"Comparing {model1_name} vs {model2_name}")
    
    # Descriptive statistics
    mean1 = np.mean(scores1)
    mean2 = np.mean(scores2)
    std1 = np.std(scores1)
    std2 = np.std(scores2)
    
    # Difference
    differences = scores1 - scores2
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    # Statistical test
    if test == "wilcoxon":
        statistic, p_value = exact_test(scores1, scores2)
        test_name = "Wilcoxon signed-rank"
    elif test == "ttest":
        statistic, p_value = paired_t_test(scores1, scores2)
        test_name = "Paired t-test"
    else:
        raise ValueError(f"Unknown test: {test}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    if pooled_std > 0:
        cohens_d = mean_diff / pooled_std
    else:
        cohens_d = 0.0
    
    results = {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'model1_mean': mean1,
        'model1_std': std1,
        'model2_mean': mean2,
        'model2_std': std2,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'cohens_d': cohens_d
    }
    
    logger.info(f"  {model1_name}: {mean1:.4f} ± {std1:.4f}")
    logger.info(f"  {model2_name}: {mean2:.4f} ± {std2:.4f}")
    logger.info(f"  Difference: {mean_diff:.4f} ± {std_diff:.4f}")
    logger.info(f"  {test_name} p-value: {p_value:.4f}")
    logger.info(f"  Cohen's d: {cohens_d:.4f}")
    
    return results


def compare_all_models(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    metric: str = "r2_scores"
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Compare all model pairs.
    
    Parameters
    ----------
    results_dict : dict
        Nested dictionary: {conn_type: {score_name: results}}
    metric : str
        Metric to compare
    
    Returns
    -------
    dict
        Pairwise comparison results
    """
    logger.info("=" * 80)
    logger.info("Comparing All Model Pairs")
    logger.info("=" * 80)
    
    # Extract all model keys
    model_keys = []
    for conn_type in results_dict.keys():
        for score_name in results_dict[conn_type].keys():
            model_keys.append((conn_type, score_name))
    
    # Pairwise comparisons
    comparisons = {}
    
    for i, (conn1, score1) in enumerate(model_keys):
        for j, (conn2, score2) in enumerate(model_keys):
            if i < j:  # Only compare each pair once
                # Get scores
                scores1 = results_dict[conn1][score1][metric]
                scores2 = results_dict[conn2][score2][metric]
                
                # Compare
                model1_name = f"{conn1}_{score1}"
                model2_name = f"{conn2}_{score2}"
                
                comparison = compare_two_models(
                    scores1, scores2,
                    model1_name, model2_name
                )
                
                comparisons[(model1_name, model2_name)] = comparison
    
    logger.info("\n" + "=" * 80)
    logger.info("Model Comparison Complete")
    logger.info("=" * 80)
    
    return comparisons


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05, method: str = "fdr_bh") -> Tuple[np.ndarray, float]:
    """
    Apply false discovery rate correction.
    
    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    alpha : float
        Significance level
    method : str
        FDR method ('bh' or 'fdr_bh' for Benjamini-Hochberg)
    
    Returns
    -------
    tuple
        (rejected, corrected_alpha)
        - rejected: Boolean array indicating which hypotheses to reject
        - corrected_alpha: Adjusted alpha threshold
    """
    n = len(p_values)
    
    # Handle both 'bh' and 'fdr_bh' as valid method names
    if method in ["bh", "fdr_bh"]:
        # Benjamini-Hochberg procedure
        # Sort p-values
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # Find largest i such that P(i) <= (i/n) * alpha
        threshold_values = np.arange(1, n + 1) / n * alpha
        comparisons = sorted_p <= threshold_values
        
        if np.any(comparisons):
            max_idx = np.where(comparisons)[0].max()
            corrected_alpha = threshold_values[max_idx]
            
            # Determine rejected hypotheses
            rejected = np.zeros(n, dtype=bool)
            rejected[sorted_idx[:max_idx + 1]] = True
        else:
            corrected_alpha = 0.0
            rejected = np.zeros(n, dtype=bool)
    
    else:
        raise ValueError(f"Unknown FDR method: {method}. Use 'bh' or 'fdr_bh'.")
    
    return rejected, corrected_alpha


def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    alpha : float
        Significance level
    
    Returns
    -------
    np.ndarray
        Boolean array indicating which hypotheses to reject
    """
    n = len(p_values)
    corrected_alpha = alpha / n
    rejected = p_values <= corrected_alpha
    
    return rejected
