"""
Data download utilities for Brain Connectivity Prediction.

Downloads connectivity matrices and cognitive data from public repositories.
"""

import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def download_file(
    url: str,
    output_path: str,
    chunk_size: int = 8192,
    timeout: int = 300
) -> bool:
    """
    Download file from URL with progress bar.
    
    Parameters
    ----------
    url : str
        URL to download from
    output_path : str
        Path to save downloaded file
    chunk_size : int
        Download chunk size in bytes
    timeout : int
        Request timeout in seconds
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Send GET request
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=Path(output_path).name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Successfully downloaded: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False


def download_data(
    data_dir: str = "data/raw",
    dataset_url: Optional[str] = None
) -> bool:
    """
    Download required datasets.
    
    Parameters
    ----------
    data_dir : str
        Directory to save downloaded data
    dataset_url : str, optional
        Custom dataset URL
    
    Returns
    -------
    bool
        True if successful
    """
    logger.info("Starting data download...")
    
    # Default Zenodo dataset (example - replace with actual working URL)
    if dataset_url is None:
        logger.warning("No dataset URL provided. Using synthetic data generation instead.")
        logger.info("To use real HCP data:")
        logger.info("1. Register at https://www.humanconnectome.org")
        logger.info("2. Download S1200 release connectivity matrices")
        logger.info("3. Place in data/raw/ directory")
        return True
    
    # Download dataset
    output_path = Path(data_dir) / "connectivity_data.zip"
    success = download_file(dataset_url, str(output_path))
    
    if success:
        logger.info("Data download completed successfully!")
    else:
        logger.error("Data download failed. Using synthetic data instead.")
    
    return success
