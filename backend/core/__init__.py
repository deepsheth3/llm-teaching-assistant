"""
Core Package

Configuration, logging, and exception handling.
"""

from .config import Settings, get_settings, get_config
from .exceptions import (
    BaseAppException,
    ConfigurationError,
    ExternalServiceError,
    OpenAIError,
    GROBIDError,
    ArxivError,
    LeetCodeError,
    ResourceNotFoundError,
    PaperNotFoundError,
    IndexNotFoundError,
    ValidationError,
    RateLimitExceededError,
    ProcessingError,
    PDFProcessingError,
    EmbeddingError,
    LessonGenerationError,
)
from .logging import get_logger, setup_logging

__all__ = [
    # Config
    "Settings",
    "get_settings",
    "get_config",
    # Logging
    "get_logger",
    "setup_logging",
    # Exceptions
    "BaseAppException",
    "ConfigurationError",
    "ExternalServiceError",
    "OpenAIError",
    "GROBIDError",
    "ArxivError",
    "LeetCodeError",
    "ResourceNotFoundError",
    "PaperNotFoundError",
    "IndexNotFoundError",
    "ValidationError",
    "RateLimitExceededError",
    "ProcessingError",
    "PDFProcessingError",
    "EmbeddingError",
    "LessonGenerationError",
]
