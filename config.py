"""Configuration management with environment variable support."""
from pydantic import BaseModel
from logging_config import logger
from exceptions import ConfigError
import yaml
import os
from typing import Dict, Any


class Config(BaseModel):
    """Configuration loader with environment variable support."""
    
    @staticmethod
    def load_config(config_file: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file with environment variable overrides.
        
        Args:
            config_file: Path to the YAML configuration file
            
        Returns:
            Dictionary containing configuration parameters
            
        Raises:
            ConfigError: If config file cannot be loaded
        """
        try:
            logger.info("Loading config file: %s", config_file)
            with open(config_file, encoding="utf-8") as f:
                parameters = yaml.safe_load(f)
                
            if parameters is None:
                raise ConfigError(f"Config file {config_file} is empty or invalid")
            
            # Require API keys from environment variables only (no fallback)
            required_api_keys = {
                'OPENAI_API_KEY': 'open_ai_api_key'
            }

            for env_var, config_key in required_api_keys.items():
                value = os.getenv(env_var)
                if not value:
                    raise ConfigError(
                        f"Required environment variable '{env_var}' is not set. "
                        f"Please export it before running the application."
                    )
                parameters[config_key] = value
                logger.info(f"Loaded {config_key} from environment variable {env_var}")
            
            return parameters
            
        except OSError as exc:
            msg = f"Config file not found: {config_file}"
            raise ConfigError(msg) from exc
        except yaml.YAMLError as exc:
            msg = f"Error parsing YAML config file {config_file}: {exc}"
            raise ConfigError(msg) from exc


CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
CONFIG = Config.load_config(CONFIG_PATH)
