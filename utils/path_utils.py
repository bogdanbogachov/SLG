"""Utility functions for path management."""
import os
from pathlib import Path
from typing import Optional
from exceptions import FileNotFoundError
from logging_config import logger
from config import CONFIG


def ensure_dir(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Path to the directory to create
    """
    os.makedirs(path, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")


def validate_file_exists(file_path: str, error_message: Optional[str] = None) -> None:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to the file to validate
        error_message: Custom error message (optional)
        
    Raises:
        FileNotFoundError: If file does not exist
    """
    if not os.path.isfile(file_path):
        msg = error_message or f"Required file not found: {file_path}"
        raise FileNotFoundError(msg)


def validate_dir_exists(dir_path: str, error_message: Optional[str] = None) -> None:
    """
    Validate that a directory exists.
    
    Args:
        dir_path: Path to the directory to validate
        error_message: Custom error message (optional)
        
    Raises:
        FileNotFoundError: If directory does not exist
    """
    if not os.path.isdir(dir_path):
        msg = error_message or f"Required directory not found: {dir_path}"
        raise FileNotFoundError(msg)


def get_model_path(model_name: str, base_dir: str) -> str:
    """
    Get the full path to a downloaded model directory.
    
    Args:
        model_name: Name of the model directory
        base_dir: Base directory containing models
        
    Returns:
        Full path to the model directory
    """
    return os.path.join(base_dir, model_name)


def get_experiment_path(experiment_name: str, base_dir: str) -> str:
    """
    Get the full path to an experiment directory.
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for experiments
        
    Returns:
        Full path to the experiment directory
    """
    return os.path.join(base_dir, experiment_name)


def get_slg_descriptions_path() -> str:
    """Return the path to the expert descriptions file (experiment-agnostic)."""
    slg_cfg = CONFIG.get("slg", {})
    descriptions_dir = slg_cfg.get("descriptions_dir", "slg_descriptions")
    return os.path.join(descriptions_dir, "descriptions.json")


def get_slg_path(experiment_name: str, experiments_dir: str = None) -> str:
    """Return the path to the SLG expert adapter directory for an experiment."""
    if experiments_dir is None:
        experiments_dir = CONFIG["paths"]["experiments"]
    slg_subdir = CONFIG.get("slg", {}).get("slg_dir", "slg")
    return os.path.join(experiments_dir, experiment_name, slg_subdir)

