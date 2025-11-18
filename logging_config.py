"""Logging configuration with configurable paths."""
import logging
import os
from typing import Optional


def setup_logger(
    name: str = "eng_llm",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logger with console and file handlers.

    Args:
        name: Name of the logger
        log_dir: Directory for log files (defaults to config or 'logs')
        log_file: Name of log file (defaults to config or 'eng_llm.log')

    Returns:
        Configured logger instance
    """
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    log.handlers.clear()

    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Lazy import CONFIG to avoid circular dependency
    # Use defaults if CONFIG is not available yet
    try:
        from config import CONFIG
        logging_config = CONFIG.get('logging', {})
        log_dir = log_dir or logging_config.get('log_dir', 'logs')
        experiment_name = CONFIG.get('experiment', 'slg')
        log_file = f'{experiment_name}.log'
    except (ImportError, KeyError, AttributeError):
        # Fallback to defaults if CONFIG is not available
        log_dir = log_dir or 'logs'
        log_file = log_file or 'slg.log'

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, log_file)
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add the handler to the logger
    log.addHandler(ch)
    log.addHandler(fh)

    return log


# Create logger instance - will be configured with config values later if needed
logger = setup_logger()
