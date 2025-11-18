"""Custom exceptions for the application."""


class EngLLMError(Exception):
    """Base exception for all application errors."""
    pass


class ConfigError(EngLLMError):
    """Raised when there's an issue with configuration."""
    pass


class ModelNotFoundError(EngLLMError):
    """Raised when a model file or directory is not found."""
    pass


class FileNotFoundError(EngLLMError):
    """Raised when a required file is not found."""
    pass

