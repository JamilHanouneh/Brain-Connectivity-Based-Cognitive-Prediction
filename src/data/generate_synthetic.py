"""
Generate synthetic connectivity matrices and cognitive scores.

Creates realistic synthetic data matching the statistical properties
of the Human Connectome Project dataset.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def generate_synthetic_connectivity(
    n_subjects: int,
    n_regions: int,
    connectivity_type: str = "FC",
    mean: float = 0.15,
    std: float = 0.25,
    sparsity: float = 0.3,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic connectivity matrices.
    
    Parameters
    ----------
    n_subjects : int
        Number of subjects
    n_regions : int
        Number of brain regions
    connectivity_type : str
        Type of connectivity ('FC' or 'SC')
    mean : float
        Mean connectivity strength
    std : float
        Standard deviation of connectivity
    sparsity : float
        Proportion of weak connections to set to zero
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Connectivity matrices (n_subjects x n_regions x n_regions)
    """
    if seed is not None:
        np.random.seed(seed)
    
    logger.info(f"Generating synthetic {connectivity_type} matrices for {n_subjects} subjects...")
    
    matrices = []
    
    for _ in range(n_subjects):
        # Generate random correlation/connectivity matrix
        if connectivity_type == "FC":
            # Functional connectivity: correlation matrix
            # Generate from multivariate normal with correlation structure
            base_corr = np.random.randn(n_regions, n_regions)
            matrix = np.corrcoef(base_corr)
            
            # Add noise
            noise = np.random.normal(0, std, (n_regions, n_regions))
            matrix = matrix * mean + noise
            
            # Ensure symmetry
            matrix = (matrix + matrix.T) / 2
            
            # Fisher z-transform range: clip to reasonable correlation range
            matrix = np.clip(matrix, -0.9, 0.9)
        
        elif connectivity_type == "SC":
            # Structural connectivity: streamline density
            # More sparse and skewed distribution
            matrix = np.random.gamma(shape=2, scale=mean, size=(n_regions, n_regions))
            
            # Add noise
            noise = np.random.normal(0, std * mean, (n_regions, n_regions))
            matrix = matrix + noise
            matrix = np.clip(matrix, 0, None)  # Non-negative
            
            # Ensure symmetry
            matrix = (matrix + matrix.T) / 2
        
        else:
            raise ValueError(f"Unknown connectivity type: {connectivity_type}")
        
        # Apply sparsity
        if sparsity > 0:
            threshold = np.percentile(np.abs(matrix), sparsity * 100)
            mask = np.abs(matrix) < threshold
            matrix[mask] = 0
        
        # Set diagonal to zero (no self-connections)
        np.fill_diagonal(matrix, 0)
        
        matrices.append(matrix)
    
    matrices = np.stack(matrices, axis=0)
    logger.info(f"Generated {connectivity_type} matrices with shape {matrices.shape}")
    
    return matrices


def generate_synthetic_cognitive_scores(
    n_subjects: int,
    score_configs: Dict[str, Dict[str, float]],
    correlations: Optional[Dict[Tuple[str, str], float]] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, list]:
    """
    Generate synthetic cognitive scores with specified distributions.
    
    Parameters
    ----------
    n_subjects : int
        Number of subjects
    score_configs : dict
        Configuration for each score type:
        {'Crystallized': {'mean': 109.52, 'std': 8.43, 'min': 85, 'max': 130}, ...}
    correlations : dict, optional
        Correlations between score types: {('Crystallized', 'Fluid'): 0.5, ...}
    seed : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Cognitive scores (n_subjects x n_scores)
    list
        Score names
    """
    if seed is not None:
        np.random.seed(seed)
    
    score_names = list(score_configs.keys())
    n_scores = len(score_names)
    
    logger.info(f"Generating synthetic cognitive scores for {n_subjects} subjects...")
    
    # Create correlation matrix
    if correlations is None:
        # Default: moderate positive correlations between cognitive measures
        corr_matrix = np.eye(n_scores)
        for i in range(n_scores):
            for j in range(i + 1, n_scores):
                corr_matrix[i, j] = corr_matrix[j, i] = 0.5
    else:
        corr_matrix = np.eye(n_scores)
        for (score1, score2), corr in correlations.items():
            i = score_names.index(score1)
            j = score_names.index(score2)
            corr_matrix[i, j] = corr_matrix[j, i] = corr
    
    # Generate correlated normal variables
    mean_vector = np.zeros(n_scores)
    scores = np.random.multivariate_normal(mean_vector, corr_matrix, size=n_subjects)
    
    # Transform to match specified distributions
    for i, score_name in enumerate(score_names):
        config = score_configs[score_name]
        
        # Standardize
        scores[:, i] = (scores[:, i] - scores[:, i].mean()) / scores[:, i].std()
        
        # Scale to target distribution
        scores[:, i] = scores[:, i] * config['std'] + config['mean']
        
        # Clip to valid range
        scores[:, i] = np.clip(scores[:, i], config['min'], config['max'])
    
    logger.info(f"Generated cognitive scores with shape {scores.shape}")
    for i, score_name in enumerate(score_names):
        logger.info(f"  {score_name}: mean={scores[:, i].mean():.2f}, std={scores[:, i].std():.2f}")
    
    return scores, score_names


def generate_synthetic_data(
    n_subjects: int,
    n_regions: int,
    config: Dict,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate complete synthetic dataset.
    
    Parameters
    ----------
    n_subjects : int
        Number of subjects
    n_regions : int
        Number of brain regions
    config : dict
        Configuration dictionary from config.yaml
    seed : int, optional
        Random seed
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'FC': Functional connectivity matrices
        - 'SC': Structural connectivity matrices
        - 'cognitive_scores': Cognitive scores array
        - 'score_names': List of cognitive score names
        - 'subject_ids': List of subject IDs
    """
    if seed is not None:
        np.random.seed(seed)
    
    logger.info("=" * 80)
    logger.info("Generating Synthetic Dataset")
    logger.info("=" * 80)
    
    # Generate subject IDs
    subject_ids = [f"sub-{i+1:04d}" for i in range(n_subjects)]
    
    # Generate functional connectivity
    fc_config = config['data']['synthetic']['fc_properties']
    fc_matrices = generate_synthetic_connectivity(
        n_subjects=n_subjects,
        n_regions=n_regions,
        connectivity_type="FC",
        mean=fc_config['mean_correlation'],
        std=fc_config['std_correlation'],
        sparsity=fc_config['sparsity'],
        seed=seed
    )
    
    # Generate structural connectivity
    sc_config = config['data']['synthetic']['sc_properties']
    sc_matrices = generate_synthetic_connectivity(
        n_subjects=n_subjects,
        n_regions=n_regions,
        connectivity_type="SC",
        mean=sc_config['mean_strength'],
        std=sc_config['std_strength'],
        sparsity=sc_config['sparsity'],
        seed=seed + 1 if seed is not None else None
    )
    
    # Generate cognitive scores
    score_configs = config['data']['synthetic']['cognitive_scores']
    cognitive_scores, score_names = generate_synthetic_cognitive_scores(
        n_subjects=n_subjects,
        score_configs=score_configs,
        seed=seed + 2 if seed is not None else None
    )
    
    logger.info("Synthetic dataset generation complete!")
    logger.info("=" * 80)
    
    return {
        'FC': fc_matrices,
        'SC': sc_matrices,
        'cognitive_scores': cognitive_scores,
        'score_names': score_names,
        'subject_ids': subject_ids
    }
