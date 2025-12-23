"""
Custom Exceptions

Structured exceptions for clean error handling throughout the application.
"""

from typing import Optional, Any


class BaseAppException(Exception):
    """Base exception for all application errors."""
    
    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details
            }
        }


# ============ Configuration Errors ============

class ConfigurationError(BaseAppException):
    """Configuration-related errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message=message,
            code="CONFIGURATION_ERROR",
            status_code=500,
            details=details
        )


# ============ External Service Errors ============

class ExternalServiceError(BaseAppException):
    """Errors from external services (OpenAI, GROBID, etc.)."""
    
    def __init__(self, service: str, message: str, details: Optional[dict] = None):
        super().__init__(
            message=f"{service} error: {message}",
            code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details={"service": service, **(details or {})}
        )


class OpenAIError(ExternalServiceError):
    """OpenAI API errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__("OpenAI", message, details)


class GROBIDError(ExternalServiceError):
    """GROBID service errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__("GROBID", message, details)


class ArxivError(ExternalServiceError):
    """arXiv API errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__("arXiv", message, details)


class LeetCodeError(ExternalServiceError):
    """LeetCode API errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__("LeetCode", message, details)


# ============ Resource Errors ============

class ResourceNotFoundError(BaseAppException):
    """Resource not found errors."""
    
    def __init__(self, resource: str, identifier: str, details: Optional[dict] = None):
        super().__init__(
            message=f"{resource} not found: {identifier}",
            code="RESOURCE_NOT_FOUND",
            status_code=404,
            details={"resource": resource, "identifier": identifier, **(details or {})}
        )


class PaperNotFoundError(ResourceNotFoundError):
    """Paper not found error."""
    
    def __init__(self, query: str):
        super().__init__("Paper", query)


class IndexNotFoundError(ResourceNotFoundError):
    """FAISS index not found error."""
    
    def __init__(self, path: str):
        super().__init__("FAISS Index", path)


# ============ Validation Errors ============

class ValidationError(BaseAppException):
    """Input validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=400,
            details={"field": field, **(details or {})} if field else details
        )


# ============ Rate Limiting ============

class RateLimitExceededError(BaseAppException):
    """Rate limit exceeded error."""
    
    def __init__(self, retry_after: int = 60):
        super().__init__(
            message="Rate limit exceeded. Please try again later.",
            code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"retry_after": retry_after}
        )


# ============ Processing Errors ============

class ProcessingError(BaseAppException):
    """General processing errors."""
    
    def __init__(self, message: str, stage: str, details: Optional[dict] = None):
        super().__init__(
            message=message,
            code="PROCESSING_ERROR",
            status_code=500,
            details={"stage": stage, **(details or {})}
        )


class PDFProcessingError(ProcessingError):
    """PDF processing errors."""
    
    def __init__(self, message: str, url: Optional[str] = None):
        super().__init__(
            message=message,
            stage="pdf_processing",
            details={"url": url} if url else None
        )


class EmbeddingError(ProcessingError):
    """Embedding generation errors."""
    
    def __init__(self, message: str):
        super().__init__(message=message, stage="embedding")


class LessonGenerationError(ProcessingError):
    """Lesson generation errors."""
    
    def __init__(self, message: str, section: Optional[str] = None):
        super().__init__(
            message=message,
            stage="lesson_generation",
            details={"section": section} if section else None
        )
