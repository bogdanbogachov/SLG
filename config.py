"""Configuration management with environment variable support."""
from pydantic import BaseModel
from exceptions import ConfigError
import yaml
import os
from typing import Dict, Any

from utils.seed import set_project_seed


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
            with open(config_file, encoding="utf-8") as f:
                parameters = yaml.safe_load(f)
                
            if parameters is None:
                raise ConfigError(f"Config file {config_file} is empty or invalid")
            
            # Keys are read only from the environment and validated by the
            # stage that needs them. Dataset builds and local inference do not
            # need network credentials.
            parameters['open_ai_api_key'] = os.getenv('OPENAI_API_KEY')
            
            # Optional environment variable overrides
            optional_overrides = {
                'EXP': 'experiment'
            }
            
            for env_var, config_key in optional_overrides.items():
                value = os.getenv(env_var)
                if value:
                    parameters[config_key] = value

            parameters.setdefault('seed', 42)
            seed_env = os.getenv('SEED')
            if seed_env is not None:
                try:
                    parameters['seed'] = int(seed_env)
                except ValueError as exc:
                    raise ConfigError(
                        f"Invalid SEED environment variable: {seed_env!r}"
                    ) from exc

            return parameters
            
        except OSError as exc:
            msg = f"Config file not found: {config_file}"
            raise ConfigError(msg) from exc
        except yaml.YAMLError as exc:
            msg = f"Error parsing YAML config file {config_file}: {exc}"
            raise ConfigError(msg) from exc


CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
CONFIG = Config.load_config(CONFIG_PATH)
set_project_seed(int(CONFIG['seed']))
