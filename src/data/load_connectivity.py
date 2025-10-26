"""
Load connectivity matrices from various formats.

Supports loading functional and structural connectivity matrices
from neuroimaging data files.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


def load_connectivity_matrix(
    file_path: str,
    matrix_type: str = "FC"
) -> np.ndarray:
    """
    Load a single connectivity matrix from file.
    
    Parameters
    ----------
    file_path : str
        Path to connectivity matrix file
    matrix_type : str
        Type of connectivity ('FC' or 'SC')
    
    Returns
    -------
    np.ndarray
        Connectivity matrix (n_regions x n_regions)
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == '.npy':
            # NumPy binary format
            matrix = np.load(file_path)
        
        elif suffix == '.txt' or suffix == '.csv':
            # Text format
            matrix = np.loadtxt(file_path, delimiter=',')
        
        elif suffix == '.nii' or suffix == '.gz':
            # NIfTI format (common for connectivity matrices)
            img = nib.load(file_path)
            matrix = img.get_fdata()
            
            # If 4D, take first volume
            if matrix.ndim == 4:
                matrix = matrix[:, :, 0, 0]
        
        elif suffix == '.pconn.nii':
            # CIFTI connectivity format (HCP)
            img = nib.load(file_path)
            matrix = img.get_fdata()
        
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Ensure 2D square matrix
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Expected square matrix, got shape {matrix.shape}")
        
        logger.debug(f"Loaded {matrix_type} matrix: {matrix.shape}")
        return matrix
    
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise


def load_all_subjects(
    data_dir: str,
    subject_ids: Optional[List[str]] = None,
    file_pattern: str = "sub-{subject_id}_connectivity.npy",
    matrix_type: str = "FC"
) -> Tuple[np.ndarray, List[str]]:
    """
    Load connectivity matrices for multiple subjects.
    
    Parameters
    ----------
    data_dir : str
        Directory containing connectivity matrices
    subject_ids : list of str, optional
        List of subject IDs to load. If None, loads all found files.
    file_pattern : str
        Filename pattern with {subject_id} placeholder
    matrix_type : str
        Type of connectivity ('FC' or 'SC')
    
    Returns
    -------
    np.ndarray
        Array of connectivity matrices (n_subjects x n_regions x n_regions)
    list of str
        List of loaded subject IDs
    """
    data_dir = Path(data_dir)
    
    # Find all connectivity files if subject_ids not provided
    if subject_ids is None:
        files = list(data_dir.glob(file_pattern.replace("{subject_id}", "*")))
        subject_ids = [f.stem.split('_')[0].replace('sub-', '') for f in files]
        logger.info(f"Found {len(subject_ids)} subject files")
    
    matrices = []
    loaded_ids = []
    
    for subject_id in subject_ids:
        file_name = file_pattern.format(subject_id=subject_id)
        file_path = data_dir / file_name
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
        
        try:
            matrix = load_connectivity_matrix(str(file_path), matrix_type)
            matrices.append(matrix)
            loaded_ids.append(subject_id)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {str(e)}")
            continue
    
    if not matrices:
        raise ValueError(f"No connectivity matrices loaded from {data_dir}")
    
    # Stack into 3D array
    matrices = np.stack(matrices, axis=0)
    logger.info(f"Loaded {len(loaded_ids)} {matrix_type} matrices with shape {matrices.shape}")
    
    return matrices, loaded_ids


def load_cognitive_scores(
    file_path: str,
    subject_ids: Optional[List[str]] = None,
    score_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load cognitive scores from CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to CSV file with cognitive scores
    subject_ids : list of str, optional
        List of subject IDs to load. If None, loads all.
    score_names : list of str, optional
        Names of cognitive scores to load. If None, loads all.
    
    Returns
    -------
    np.ndarray
        Cognitive scores array (n_subjects x n_scores)
    list of str
        List of subject IDs
    list of str
        List of score names
    """
    import pandas as pd
    
    # Load CSV
    df = pd.read_csv(file_path, index_col=0)
    
    # Filter subject IDs
    if subject_ids is not None:
        df = df.loc[df.index.isin(subject_ids)]
    
    # Filter score names
    if score_names is not None:
        df = df[score_names]
    
    subject_ids = df.index.tolist()
    score_names = df.columns.tolist()
    scores = df.values
    
    logger.info(f"Loaded cognitive scores for {len(subject_ids)} subjects, {len(score_names)} measures")
    
    return scores, subject_ids, score_names


def extract_upper_triangle(matrix: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Extract upper triangle of matrix (excluding diagonal).
    
    Parameters
    ----------
    matrix : np.ndarray
        Square connectivity matrix
    k : int
        Diagonal offset (1 excludes main diagonal)
    
    Returns
    -------
    np.ndarray
        Upper triangle values as 1D array
    """
    return matrix[np.triu_indices_from(matrix, k=k)]


def extract_lower_triangle(matrix: np.ndarray, k: int = -1) -> np.ndarray:
    """
    Extract lower triangle of matrix (excluding diagonal).
    
    Parameters
    ----------
    matrix : np.ndarray
        Square connectivity matrix
    k : int
        Diagonal offset (-1 excludes main diagonal)
    
    Returns
    -------
    np.ndarray
        Lower triangle values as 1D array
    """
    return matrix[np.tril_indices_from(matrix, k=k)]


def matrices_to_feature_vectors(
    matrices: np.ndarray,
    vectorize: bool = True,
    include_diagonal: bool = False
) -> np.ndarray:
    """
    Convert connectivity matrices to feature vectors.
    
    Parameters
    ----------
    matrices : np.ndarray
        Connectivity matrices (n_subjects x n_regions x n_regions)
    vectorize : bool
        If True, extract upper triangle. If False, keep as matrix.
    include_diagonal : bool
        Whether to include diagonal values
    
    Returns
    -------
    np.ndarray
        Feature vectors (n_subjects x n_features)
    """
    if not vectorize:
        n_subjects = matrices.shape[0]
        return matrices.reshape(n_subjects, -1)
    
    k = 0 if include_diagonal else 1
    feature_vectors = []
    
    for matrix in matrices:
        features = extract_upper_triangle(matrix, k=k)
        feature_vectors.append(features)
    
    return np.stack(feature_vectors, axis=0)
