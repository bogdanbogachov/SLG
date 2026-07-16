"""Utility functions for path management."""
import hashlib
import os
import re
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


WINDOWS_FORBIDDEN_FILENAME_CHARS = r'<>:"/\|?*'


def safe_path_component(name: str, max_length: int = 180) -> str:
    """Return a deterministic Windows-safe path component."""
    original = str(name)
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", original)
    safe = re.sub(r"_+", "_", safe).strip(" ._")
    if not safe:
        safe = "untitled"

    changed = safe != original
    if changed or len(safe) > max_length:
        digest = hashlib.sha1(original.encode("utf-8")).hexdigest()[:8]
        room = max_length - len(digest) - 1
        safe = f"{safe[:room].rstrip(' ._')}_{digest}"

    return safe


def slg_expert_id_from_filename(filename: str) -> str:
    """Map a split_by_title JSON filename to the matching SLG adapter directory."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return safe_path_component(stem)


# Written by commands.slg_embeddings.save_slg_embedding_artifacts; required before SLG inference.
SLG_EMBEDDING_ARTIFACT_NAMES = (
    "chunk_embeddings_raw.npy",
    "expert_ids.json",
    "index.json",
)


def get_slg_index_dir(experiment_name: str, experiments_dir: str = None) -> str:
    """
    Return the per-experiment directory for SLG index artifacts
    (``experiments/<experiment>/<slg_index>/``), containing chunk_embeddings_raw.npy,
    expert_ids.json, and index.json.
    """
    if experiments_dir is None:
        paths_config = CONFIG["paths"]
        experiments_dir = paths_config["experiments"]
    slg_index_name = CONFIG["paths"].get("slg_index", "slg_index")
    return os.path.join(experiments_dir, experiment_name, slg_index_name)


def validate_slg_embedding_artifacts(index_dir: str) -> None:
    """
    Ensure the experiment SLG index step has produced required files under ``index_dir``.

    Raises:
        FileNotFoundError: If the directory is missing or any artifact file is absent.
    """
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(
            f"SLG index directory does not exist: {index_dir}. "
            "Run the SLG embeddings step first "
            "(e.g. commands.slg_embeddings.run_slg_embeddings). It creates this folder under "
            "experiments/<experiment>/<slg_index>/ and writes chunk_embeddings_raw.npy, "
            "expert_ids.json, and index.json."
        )
    missing = [
        name
        for name in SLG_EMBEDDING_ARTIFACT_NAMES
        if not os.path.isfile(os.path.join(index_dir, name))
    ]
    if missing:
        listed = ", ".join(missing)
        expected = ", ".join(SLG_EMBEDDING_ARTIFACT_NAMES)
        raise FileNotFoundError(
            f"Missing SLG index file(s) in {index_dir}: {listed}. "
            f"Expected all of: {expected}. "
            "Run commands.slg_embeddings.run_slg_embeddings for this experiment before inference."
        )


def get_slg_path(experiment_name: str, experiments_dir: str = None) -> str:
    """
    Get the path to SLG models for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        experiments_dir: Base directory for experiments (defaults to CONFIG['paths']['experiments'])
        
    Returns:
        Path to SLG models directory
    """
    if experiments_dir is None:
        paths_config = CONFIG['paths']
        experiments_dir = paths_config['experiments']
    slg_subdir = CONFIG.get('slg_formation', {}).get('slg_dir', 'slg')
    return os.path.join(experiments_dir, experiment_name, slg_subdir)

