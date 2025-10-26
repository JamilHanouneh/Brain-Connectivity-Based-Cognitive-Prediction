"""
I/O utilities for the Brain Connectivity Prediction project.

Handles configuration loading, data saving/loading, and file management.
"""

import yaml
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to config.yaml file
    
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_path : str
        Path to save config
    """
    ensure_dir(Path(output_path).parent)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_results(
    results: Union[Dict, pd.DataFrame, np.ndarray],
    output_path: str,
    format: str = "auto"
) -> None:
    """
    Save results to file.
    
    Parameters
    ----------
    results : dict, DataFrame, or ndarray
        Results to save
    output_path : str
        Path to save results
    format : str
        File format ('auto', 'json', 'csv', 'npy', 'pkl')
    """
    ensure_dir(Path(output_path).parent)
    
    # Auto-detect format
    if format == "auto":
        suffix = Path(output_path).suffix.lower()
        format_map = {
            '.json': 'json',
            '.csv': 'csv',
            '.npy': 'npy',
            '.pkl': 'pkl',
            '.pickle': 'pkl'
        }
        format = format_map.get(suffix, 'json')
    
    # Save based on format
    if format == "json":
        # Convert numpy arrays and tuple keys to lists for JSON serialization
        def convert_to_serializable(obj):
            """Recursively convert numpy types and tuple keys to JSON-compatible types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                # Convert tuple keys to strings
                new_dict = {}
                for key, value in obj.items():
                    if isinstance(key, tuple):
                        # Convert tuple to string representation
                        new_key = " vs ".join(str(k) for k in key)
                    else:
                        new_key = key
                    new_dict[str(new_key)] = convert_to_serializable(value)
                return new_dict
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_to_serializable(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2)
    
    elif format == "csv":
        if isinstance(results, pd.DataFrame):
            results.to_csv(output_path, index=True)
        elif isinstance(results, dict):
            pd.DataFrame(results).to_csv(output_path, index=True)
        else:
            raise ValueError("CSV format requires DataFrame or dict")
    
    elif format == "npy":
        if isinstance(results, np.ndarray):
            np.save(output_path, results)
        else:
            raise ValueError("NPY format requires numpy array")
    
    elif format == "pkl":
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    
    else:
        raise ValueError(f"Unknown format: {format}")


def load_results(input_path: str, format: str = "auto") -> Any:
    """
    Load results from file.
    
    Parameters
    ----------
    input_path : str
        Path to results file
    format : str
        File format ('auto', 'json', 'csv', 'npy', 'pkl')
    
    Returns
    -------
    Results object (type depends on file format)
    """
    # Auto-detect format
    if format == "auto":
        suffix = Path(input_path).suffix.lower()
        format_map = {
            '.json': 'json',
            '.csv': 'csv',
            '.npy': 'npy',
            '.pkl': 'pkl',
            '.pickle': 'pkl'
        }
        format = format_map.get(suffix, 'json')
    
    # Load based on format
    if format == "json":
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif format == "csv":
        return pd.read_csv(input_path, index_col=0)
    
    elif format == "npy":
        return np.load(input_path)
    
    elif format == "pkl":
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    
    else:
        raise ValueError(f"Unknown format: {format}")


def save_model(model: Any, output_path: str) -> None:
    """
    Save trained model to file.
    
    Parameters
    ----------
    model : Any
        Trained model object
    output_path : str
        Path to save model
    """
    ensure_dir(Path(output_path).parent)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(input_path: str) -> Any:
    """
    Load trained model from file.
    
    Parameters
    ----------
    input_path : str
        Path to model file
    
    Returns
    -------
    Loaded model object
    """
    with open(input_path, 'rb') as f:
        return pickle.load(f)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Parameters
    ----------
    directory : str or Path
        Directory path
    
    Returns
    -------
    Path
        Directory path as Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_list(
    directory: str,
    pattern: str = "*",
    recursive: bool = False
) -> list:
    """
    Get list of files matching pattern.
    
    Parameters
    ----------
    directory : str
        Directory to search
    pattern : str
        File pattern (e.g., "*.csv")
    recursive : bool
        Whether to search recursively
    
    Returns
    -------
    list
        List of matching file paths
    """
    path = Path(directory)
    
    if recursive:
        files = list(path.rglob(pattern))
    else:
        files = list(path.glob(pattern))
    
    return sorted([str(f) for f in files])
