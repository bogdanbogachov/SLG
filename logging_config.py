"""Logging configuration with configurable paths."""
import logging
import os
from typing import Optional
from config import CONFIG


def setup_logger(
    name: str = "eng_llm",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logger with console and file handlers.
    
    Args:
        name: Name of the logger
        log_dir: Directory for log files (defaults to DEFAULT_LOG_DIR)
        log_file: Name of log file (defaults to DEFAULT_LOG_FILE)
        
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
    
    # Create a file handler with configurable path
    logging_config = CONFIG['logging']
    log_dir = log_dir or logging_config['log_dir']
    log_file = log_file or logging_config['log_file']
    
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
