"""
Logging utilities for the Brain Connectivity Prediction project.

Provides centralized logging configuration and management.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and/or console handlers.
    
    Parameters
    ----------
    name : str
        Logger name
    log_file : str, optional
        Path to log file
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format_string : str, optional
        Custom format string
    console : bool
        Whether to log to console
    
    Returns
    -------
    logging.Logger
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if log_file is not None:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.
    
    Parameters
    ----------
    name : str
        Logger name
    
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)


class LoggerContext:
    """Context manager for temporary logger configuration."""
    
    def __init__(self, logger: logging.Logger, level: str):
        """
        Initialize logger context.
        
        Parameters
        ----------
        logger : logging.Logger
            Logger to modify
        level : str
            Temporary logging level
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = None
    
    def __enter__(self):
        """Enter context - change logging level."""
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore logging level."""
        self.logger.setLevel(self.old_level)


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger to use
    
    Returns
    -------
    function
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.info(f"Starting {func.__name__}...")
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"Completed {func.__name__} in {duration:.2f} seconds")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise
        
        return wrapper
    return decorator
