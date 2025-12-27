# Complete Project Context for Claude

Generated: Fri Dec 26 20:30:29 PST 2025

## 1. Project Structure
```
```

## 2. Backend Code

### 2.1 Core

#### backend/core/__init__.py
```python
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
```

#### backend/core/config.py
```python
"""
Configuration Management v2

Centralized configuration with environment variable support,
validation, and sensible defaults.

NEW in v2:
- Relevance thresholds
- Dynamic paper fetching settings
- Pinecone (optional) settings
- Removed LeetCode settings
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # ==========================================================================
    # App Info
    # ==========================================================================
    app_name: str = "LLM Teaching Assistant"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, description="Debug mode")
    
    # ==========================================================================
    # API Settings
    # ==========================================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # ==========================================================================
    # OpenAI Settings
    # ==========================================================================
    openai_api_key: str = Field(..., description="OpenAI API key")
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"
    lesson_model: str = "gpt-4o-mini"
    
    # ==========================================================================
    # Vector Database Settings (NEW in v2)
    # ==========================================================================
    # Pinecone (optional - for production with dynamic updates)
    use_pinecone: bool = Field(default=False, description="Use Pinecone instead of FAISS")
    pinecone_api_key: Optional[str] = Field(default=None, description="Pinecone API key")
    pinecone_index_name: str = "llm-teaching-assistant"
    pinecone_environment: str = "us-east-1"
    
    # ==========================================================================
    # Relevance Thresholds (NEW in v2)
    # ==========================================================================
    high_relevance_threshold: float = Field(
        default=0.50, 
        description="Score above this = high relevance, use directly"
    )
    medium_relevance_threshold: float = Field(
        default=0.35, 
        description="Score above this = medium relevance, use but try to improve"
    )
    low_relevance_threshold: float = Field(
        default=0.20,
        description="Score below this = irrelevant"
    )
    
    # ==========================================================================
    # Dynamic Paper Fetching (NEW in v2)
    # ==========================================================================
    dynamic_fetch_enabled: bool = Field(
        default=True, 
        description="Enable fetching new papers when no good match"
    )
    max_papers_per_fetch: int = Field(
        default=10, 
        description="Max papers to fetch per query"
    )
    max_daily_fetches: int = Field(
        default=100, 
        description="Max Semantic Scholar API calls per day (cost control)"
    )
    semantic_scholar_api_key: Optional[str] = Field(
        default=None, 
        description="Optional API key for higher rate limits"
    )
    
    # ==========================================================================
    # GROBID Settings
    # ==========================================================================
    grobid_url: str = "https://kermitt2-grobid.hf.space"
    grobid_timeout: int = 120
    use_grobid: bool = True
    
    # ==========================================================================
    # File Paths
    # ==========================================================================
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    faiss_index_path: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "faiss" / "papers.index")
    urls_json_path: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "faiss" / "urls.json")
    cache_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "cache")
    
    # ==========================================================================
    # Cache Settings
    # ==========================================================================
    cache_enabled: bool = True
    cache_ttl: int = 86400  # 24 hours
    
    # ==========================================================================
    # Rate Limiting
    # ==========================================================================
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 30
    rate_limit_window: int = 60
    
    # ==========================================================================
    # Logging
    # ==========================================================================
    log_level: str = "INFO"
    log_format: str = "text"
    
    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or v == "your-api-key-here":
            raise ValueError("Valid OpenAI API key is required")
        return v
    
    @field_validator("pinecone_api_key")
    @classmethod
    def validate_pinecone_key(cls, v: Optional[str]) -> Optional[str]:
        if v and v != "your-pinecone-key":
            return v
        return None
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings


# Convenience function
def get_config() -> Settings:
    """Alias for get_settings."""
    return get_settings()
```

#### backend/core/exceptions.py
```python
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
```

#### backend/core/logging.py
```python
"""
Structured Logging

Production-ready logging with JSON output support for easy parsing.
"""

import sys
import logging
import json
from datetime import datetime
from typing import Any
from functools import lru_cache


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
        
        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter with colors."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"{color}[{timestamp}] {record.levelname:8}{self.RESET} | {record.name} | {record.getMessage()}"
        
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message


class AppLogger(logging.Logger):
    """Extended logger with structured data support."""
    
    def _log_with_data(
        self,
        level: int,
        msg: str,
        data: dict[str, Any] | None = None,
        *args,
        **kwargs
    ):
        if data:
            extra = kwargs.get("extra", {})
            extra["extra_data"] = data
            kwargs["extra"] = extra
        
        self.log(level, msg, *args, **kwargs)
    
    def debug_with_data(self, msg: str, data: dict[str, Any] | None = None, **kwargs):
        self._log_with_data(logging.DEBUG, msg, data, **kwargs)
    
    def info_with_data(self, msg: str, data: dict[str, Any] | None = None, **kwargs):
        self._log_with_data(logging.INFO, msg, data, **kwargs)
    
    def warning_with_data(self, msg: str, data: dict[str, Any] | None = None, **kwargs):
        self._log_with_data(logging.WARNING, msg, data, **kwargs)
    
    def error_with_data(self, msg: str, data: dict[str, Any] | None = None, **kwargs):
        self._log_with_data(logging.ERROR, msg, data, **kwargs)


def setup_logging(
    level: str = "INFO",
    format_type: str = "text",
    name: str = "llm_ta"
) -> AppLogger:
    """
    Setup application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: "json" for structured logs, "text" for human-readable
        name: Logger name
    
    Returns:
        Configured logger instance
    """
    # Set custom logger class
    logging.setLoggerClass(AppLogger)
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    # Set formatter
    if format_type == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())
    
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


@lru_cache()
def get_logger(name: str = "llm_ta") -> AppLogger:
    """Get a cached logger instance."""
    from core.config import get_settings
    
    settings = get_settings()
    return setup_logging(
        level=settings.log_level,
        format_type=settings.log_format,
        name=name
    )
```

### 2.2 Models

#### backend/models/__init__.py
```python
"""
Data Models Package

Pydantic models for all data structures.
"""

from .paper import (
    PaperMetadata,
    PaperSection,
    ParsedPaper,
    PaperSearchResult,
    PaperSearchRequest,
)
from .lesson import (
    LessonDifficulty,
    LessonFragment,
    FullLesson,
    LessonRequest,
    LessonResponse,
    StreamingLessonChunk,
)
from .problem import (
    ProblemDifficulty,
    LeetCodeProblem,
    ProblemRequest,
    ProblemResponse,
    ProblemCatalogEntry,
)

__all__ = [
    # Paper models
    "PaperMetadata",
    "PaperSection",
    "ParsedPaper",
    "PaperSearchResult",
    "PaperSearchRequest",
    # Lesson models
    "LessonDifficulty",
    "LessonFragment",
    "FullLesson",
    "LessonRequest",
    "LessonResponse",
    "StreamingLessonChunk",
    # Problem models
    "ProblemDifficulty",
    "LeetCodeProblem",
    "ProblemRequest",
    "ProblemResponse",
    "ProblemCatalogEntry",
]
```

#### backend/models/lesson.py
```python
"""
Lesson Data Models

Pydantic models for lesson-related data structures.
"""

from datetime import datetime
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class LessonDifficulty(str, Enum):
    """Lesson difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class LessonFragment(BaseModel):
    """A single lesson fragment from a paper section."""
    
    section_name: str = Field(..., description="Original section name")
    content: str = Field(..., description="Beginner-friendly lesson content")
    order: int = Field(..., description="Order in the full lesson")
    has_math: bool = Field(False, description="Whether section includes math")
    has_code: bool = Field(False, description="Whether section includes code")
    estimated_read_time: int = Field(0, description="Estimated read time in minutes")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.estimated_read_time == 0:
            # Average reading speed: 200 words per minute
            word_count = len(self.content.split())
            self.estimated_read_time = max(1, word_count // 200)


class FullLesson(BaseModel):
    """A complete lesson generated from a paper."""
    
    paper_id: str = Field(..., description="Source paper arXiv ID")
    paper_title: str = Field(..., description="Source paper title")
    paper_url: str = Field(..., description="Source paper URL")
    query: str = Field(..., description="Original user query")
    
    fragments: list[LessonFragment] = Field(default_factory=list)
    
    difficulty: LessonDifficulty = Field(LessonDifficulty.BEGINNER)
    total_read_time: int = Field(0, description="Total estimated read time")
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generation_time_seconds: float = Field(0, description="Time to generate")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.total_read_time == 0 and self.fragments:
            self.total_read_time = sum(f.estimated_read_time for f in self.fragments)
    
    @property
    def full_content(self) -> str:
        """Get the full lesson as a single string."""
        parts = []
        for fragment in sorted(self.fragments, key=lambda f: f.order):
            parts.append(f"## {fragment.section_name.title()}\n\n{fragment.content}")
        return "\n\n---\n\n".join(parts)
    
    @property
    def table_of_contents(self) -> list[str]:
        """Get section names as table of contents."""
        return [f.section_name.title() for f in sorted(self.fragments, key=lambda f: f.order)]


class LessonRequest(BaseModel):
    """Request to generate a lesson."""
    
    query: str = Field(..., min_length=3, max_length=500, description="What to learn about")
    difficulty: LessonDifficulty = Field(LessonDifficulty.BEGINNER)
    include_examples: bool = Field(True, description="Include examples in explanations")
    include_math: bool = Field(True, description="Include step-by-step math")
    max_sections: Optional[int] = Field(None, ge=1, le=20, description="Limit sections")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "attention mechanisms in transformers",
                "difficulty": "beginner",
                "include_examples": True,
                "include_math": True
            }
        }


class LessonResponse(BaseModel):
    """Response containing a generated lesson."""
    
    success: bool = Field(True)
    lesson: Optional[FullLesson] = None
    error: Optional[str] = None
    
    # Metadata
    cached: bool = Field(False, description="Whether result was from cache")
    processing_time_ms: int = Field(0, description="Processing time in milliseconds")


class StreamingLessonChunk(BaseModel):
    """A chunk of a streaming lesson response."""
    
    type: str = Field(..., description="Chunk type: 'metadata', 'section', 'done', 'error'")
    data: dict = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "examples": [
                {"type": "metadata", "data": {"paper_title": "Attention Is All You Need", "total_sections": 5}},
                {"type": "section", "data": {"name": "Introduction", "content": "Let's start..."}},
                {"type": "done", "data": {"total_time_seconds": 45.2}},
                {"type": "error", "data": {"message": "Failed to process"}}
            ]
        }
```

#### backend/models/paper.py
```python
"""
Paper Data Models

Pydantic models for paper-related data structures.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class PaperMetadata(BaseModel):
    """Metadata for a research paper."""
    
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    url: HttpUrl = Field(..., description="Paper URL")
    abstract: Optional[str] = Field(None, description="Paper abstract")
    authors: list[str] = Field(default_factory=list, description="Paper authors")
    categories: list[str] = Field(default_factory=list, description="arXiv categories")
    published_date: Optional[datetime] = Field(None, description="Publication date")
    
    class Config:
        json_schema_extra = {
            "example": {
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/abs/1706.03762",
                "abstract": "The dominant sequence transduction models...",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "categories": ["cs.CL", "cs.LG"],
                "published_date": "2017-06-12T00:00:00Z"
            }
        }


class PaperSection(BaseModel):
    """A section extracted from a paper."""
    
    name: str = Field(..., description="Section name/title")
    content: str = Field(..., description="Section text content")
    order: int = Field(..., description="Section order in paper")
    word_count: int = Field(0, description="Word count")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.word_count == 0:
            self.word_count = len(self.content.split())


class ParsedPaper(BaseModel):
    """A fully parsed paper with sections."""
    
    metadata: PaperMetadata
    sections: list[PaperSection] = Field(default_factory=list)
    raw_text: Optional[str] = Field(None, description="Full raw text")
    parsing_method: str = Field("grobid", description="Method used to parse (grobid/abstract)")
    parsed_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def section_names(self) -> list[str]:
        return [s.name for s in self.sections]
    
    @property
    def total_words(self) -> int:
        return sum(s.word_count for s in self.sections)


class PaperSearchResult(BaseModel):
    """Result from a paper search."""
    
    paper: PaperMetadata
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity score")
    index_position: int = Field(..., description="Position in FAISS index")
    
    class Config:
        json_schema_extra = {
            "example": {
                "paper": {
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "url": "https://arxiv.org/abs/1706.03762"
                },
                "similarity_score": 0.92,
                "index_position": 42
            }
        }


class PaperSearchRequest(BaseModel):
    """Request to search for papers."""
    
    query: str = Field(..., min_length=3, max_length=500, description="Search query")
    top_k: int = Field(1, ge=1, le=10, description="Number of results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "attention mechanisms in transformers",
                "top_k": 3
            }
        }
```

#### backend/models/problem.py
```python
"""
LeetCode Problem Models

Pydantic models for LeetCode-related data structures.
"""

from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class ProblemDifficulty(str, Enum):
    """LeetCode problem difficulty levels."""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class LeetCodeProblem(BaseModel):
    """A LeetCode problem."""
    
    title: str = Field(..., description="Problem title")
    slug: str = Field(..., description="URL slug")
    difficulty: ProblemDifficulty = Field(..., description="Difficulty level")
    statement: str = Field(..., description="Problem statement")
    url: str = Field("", description="Full LeetCode URL")
    
    # Optional metadata
    acceptance_rate: Optional[float] = Field(None, description="Acceptance rate percentage")
    topics: list[str] = Field(default_factory=list, description="Related topics")
    hints: list[str] = Field(default_factory=list, description="Problem hints")
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.url and self.slug:
            self.url = f"https://leetcode.com/problems/{self.slug}/"
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Two Sum",
                "slug": "two-sum",
                "difficulty": "Easy",
                "statement": "Given an array of integers nums and an integer target...",
                "url": "https://leetcode.com/problems/two-sum/",
                "topics": ["Array", "Hash Table"]
            }
        }


class ProblemRequest(BaseModel):
    """Request for a LeetCode problem."""
    
    difficulties: list[ProblemDifficulty] = Field(
        default=[ProblemDifficulty.MEDIUM, ProblemDifficulty.HARD],
        description="Allowed difficulties"
    )
    topics: Optional[list[str]] = Field(None, description="Filter by topics")
    exclude_premium: bool = Field(True, description="Exclude premium problems")
    
    class Config:
        json_schema_extra = {
            "example": {
                "difficulties": ["Medium", "Hard"],
                "exclude_premium": True
            }
        }


class ProblemResponse(BaseModel):
    """Response containing a LeetCode problem."""
    
    success: bool = Field(True)
    problem: Optional[LeetCodeProblem] = None
    error: Optional[str] = None
    
    # Metadata
    cached: bool = Field(False)
    processing_time_ms: int = Field(0)


class ProblemCatalogEntry(BaseModel):
    """Entry in the LeetCode problem catalog."""
    
    slug: str
    title: str
    difficulty: ProblemDifficulty
    paid_only: bool = False
    acceptance_rate: Optional[float] = None
```

### 2.3 Services

#### backend/services/__init__.py
```python
"""
Services Package v2

Business logic layer for the application.

Changes in v2:
- Removed LeetCodeService
- Added QueryService (query enhancement)
- Added ScholarService (dynamic paper fetching)
"""

from .cache_service import CacheService, get_cache_service
from .embedding_service import EmbeddingService, get_embedding_service
from .paper_service import PaperService, get_paper_service
from .lesson_service import LessonService, get_lesson_service
from .teaching_service import TeachingService, get_teaching_service
from .query_service import QueryService, get_query_service
from .scholar_service import ScholarService, get_scholar_service

__all__ = [
    # Cache
    "CacheService",
    "get_cache_service",
    # Embedding
    "EmbeddingService",
    "get_embedding_service",
    # Paper
    "PaperService",
    "get_paper_service",
    # Lesson
    "LessonService",
    "get_lesson_service",
    # Teaching (main orchestrator)
    "TeachingService",
    "get_teaching_service",
    # Query Enhancement (NEW)
    "QueryService",
    "get_query_service",
    # Scholar Service (NEW)
    "ScholarService",
    "get_scholar_service",
]
```

#### backend/services/cache_service.py
```python
"""
Cache Service

File-based and in-memory caching for improved performance.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, TypeVar, Generic
from functools import lru_cache
from pydantic import BaseModel

from core.config import get_settings
from core.logging import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


class CacheEntry(BaseModel, Generic[T]):
    """A cached entry with metadata."""
    
    key: str
    value: Any
    created_at: float
    ttl: int
    
    @property
    def is_expired(self) -> bool:
        return time.time() > (self.created_at + self.ttl)


class CacheService:
    """
    Hybrid cache service with in-memory and file-based storage.
    
    - Hot data stays in memory (LRU cache)
    - Cold data persists to disk
    - Automatic TTL expiration
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, default_ttl: int = 86400):
        settings = get_settings()
        self.cache_dir = cache_dir or settings.cache_dir
        self.default_ttl = default_ttl
        self.enabled = settings.cache_enabled
        
        # In-memory cache
        self._memory_cache: dict[str, CacheEntry] = {}
        self._max_memory_items = 100
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Cache service initialized: dir={self.cache_dir}, enabled={self.enabled}")
    
    def _make_key(self, namespace: str, identifier: str) -> str:
        """Create a cache key from namespace and identifier."""
        raw = f"{namespace}:{identifier}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{key}.json"
    
    def get(self, namespace: str, identifier: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            namespace: Cache namespace (e.g., "lessons", "papers")
            identifier: Unique identifier within namespace
            
        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None
        
        key = self._make_key(namespace, identifier)
        
        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired:
                logger.debug(f"Cache hit (memory): {namespace}:{identifier[:20]}")
                return entry.value
            else:
                del self._memory_cache[key]
        
        # Check file cache
        file_path = self._get_file_path(key)
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    entry = CacheEntry(**data)
                    
                    if not entry.is_expired:
                        # Promote to memory cache
                        self._memory_cache[key] = entry
                        self._evict_if_needed()
                        logger.debug(f"Cache hit (file): {namespace}:{identifier[:20]}")
                        return entry.value
                    else:
                        # Remove expired file
                        file_path.unlink()
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        logger.debug(f"Cache miss: {namespace}:{identifier[:20]}")
        return None
    
    def set(
        self,
        namespace: str,
        identifier: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set a value in cache.
        
        Args:
            namespace: Cache namespace
            identifier: Unique identifier
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (default: 24 hours)
            
        Returns:
            True if cached successfully
        """
        if not self.enabled:
            return False
        
        key = self._make_key(namespace, identifier)
        ttl = ttl or self.default_ttl
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=ttl
        )
        
        # Save to memory
        self._memory_cache[key] = entry
        self._evict_if_needed()
        
        # Save to file
        try:
            file_path = self._get_file_path(key)
            with open(file_path, "w") as f:
                json.dump(entry.model_dump(), f)
            logger.debug(f"Cached: {namespace}:{identifier[:20]}")
            return True
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
            return False
    
    def delete(self, namespace: str, identifier: str) -> bool:
        """Delete a cached value."""
        key = self._make_key(namespace, identifier)
        
        # Remove from memory
        if key in self._memory_cache:
            del self._memory_cache[key]
        
        # Remove from file
        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()
            return True
        
        return False
    
    def clear(self, namespace: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            namespace: If provided, only clear this namespace
            
        Returns:
            Number of entries cleared
        """
        count = 0
        
        if namespace is None:
            # Clear all
            count = len(self._memory_cache)
            self._memory_cache.clear()
            
            for file_path in self.cache_dir.glob("*.json"):
                file_path.unlink()
                count += 1
        
        logger.info(f"Cache cleared: {count} entries")
        return count
    
    def _evict_if_needed(self):
        """Evict oldest entries if memory cache is too large."""
        while len(self._memory_cache) > self._max_memory_items:
            # Remove oldest entry
            oldest_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k].created_at
            )
            del self._memory_cache[oldest_key]
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        file_count = len(list(self.cache_dir.glob("*.json")))
        
        return {
            "enabled": self.enabled,
            "memory_entries": len(self._memory_cache),
            "file_entries": file_count,
            "cache_dir": str(self.cache_dir),
            "default_ttl": self.default_ttl
        }


# Singleton instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get the global cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
```

#### backend/services/embedding_service.py
```python
"""
Embedding Service

Vector embedding generation and FAISS index management.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
import faiss
from openai import OpenAI

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import EmbeddingError, IndexNotFoundError

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings and managing FAISS index.
    
    Features:
    - Batch embedding generation
    - FAISS index creation and search
    - URL mapping management
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.embedding_model
        
        self._index: Optional[faiss.Index] = None
        self._urls: Optional[list[str]] = None
        
        logger.info(f"Embedding service initialized: model={self.model}")
    
    @property
    def index(self) -> faiss.Index:
        """Get the FAISS index, loading if necessary."""
        if self._index is None:
            self.load_index()
        return self._index
    
    @property
    def urls(self) -> list[str]:
        """Get the URL list, loading if necessary."""
        if self._urls is None:
            self.load_urls()
        return self._urls
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = np.array(response.data[0].embedding, dtype="float32")
            return embedding
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise EmbeddingError(f"Failed to create embedding: {e}")
    
    def create_embeddings_batch(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """
        Create embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            
        Returns:
            Array of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                raise EmbeddingError(f"Failed to create batch embeddings: {e}")
        
        return np.array(all_embeddings, dtype="float32")
    
    def build_index(self, embeddings: np.ndarray, urls: list[str]) -> None:
        """
        Build and save FAISS index.
        
        Args:
            embeddings: Array of embeddings
            urls: Corresponding URLs
        """
        if len(embeddings) != len(urls):
            raise ValueError("Embeddings and URLs must have same length")
        
        # Create FAISS index (L2 distance)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Save index
        self.settings.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.settings.faiss_index_path))
        
        # Save URLs
        with open(self.settings.urls_json_path, "w", encoding="utf-8") as f:
            json.dump(urls, f, ensure_ascii=False, indent=2)
        
        self._index = index
        self._urls = urls
        
        logger.info(f"Built FAISS index with {len(urls)} entries")
    
    def load_index(self) -> None:
        """Load FAISS index from disk."""
        index_path = self.settings.faiss_index_path
        
        if not index_path.exists():
            raise IndexNotFoundError(str(index_path))
        
        self._index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index: {self._index.ntotal} vectors")
    
    def load_urls(self) -> None:
        """Load URLs from disk."""
        urls_path = self.settings.urls_json_path
        
        if not urls_path.exists():
            raise IndexNotFoundError(str(urls_path))
        
        with open(urls_path, "r", encoding="utf-8") as f:
            self._urls = json.load(f)
        
        logger.info(f"Loaded {len(self._urls)} URLs")
    
    def search(self, query: str, k: int = 1) -> list[tuple[int, float, str]]:
        """
        Search for similar papers.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (index, distance, url) tuples
        """
        # Create query embedding
        query_embedding = self.create_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i in range(k):
            idx = int(indices[0][i])
            dist = float(distances[0][i])
            url = self.urls[idx] if idx < len(self.urls) else ""
            results.append((idx, dist, url))
        
        logger.debug(f"Search for '{query[:30]}...' returned {len(results)} results")
        return results
    
    def search_by_embedding(self, embedding: np.ndarray, k: int = 1) -> list[tuple[int, float, str]]:
        """
        Search using a pre-computed embedding.
        
        Args:
            embedding: Query embedding
            k: Number of results
            
        Returns:
            List of (index, distance, url) tuples
        """
        embedding = embedding.reshape(1, -1).astype("float32")
        distances, indices = self.index.search(embedding, k)
        
        results = []
        for i in range(k):
            idx = int(indices[0][i])
            dist = float(distances[0][i])
            url = self.urls[idx] if idx < len(self.urls) else ""
            results.append((idx, dist, url))
        
        return results
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "model": self.model,
            "index_loaded": self._index is not None,
            "index_size": self._index.ntotal if self._index else 0,
            "urls_loaded": self._urls is not None,
            "urls_count": len(self._urls) if self._urls else 0,
            "index_path": str(self.settings.faiss_index_path),
        }


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
```

#### backend/services/lesson_service.py
```python
"""
Lesson Generation Service

Converts research paper sections into beginner-friendly lessons.
"""

import time
import asyncio
from typing import Optional, AsyncGenerator
from openai import OpenAI, AsyncOpenAI

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import LessonGenerationError
from models.paper import ParsedPaper, PaperSection
from models.lesson import (
    LessonFragment,
    FullLesson,
    LessonRequest,
    LessonDifficulty,
    StreamingLessonChunk,
)
from services.cache_service import get_cache_service

logger = get_logger(__name__)


class LessonService:
    """
    Service for generating lessons from paper sections.
    
    Features:
    - Beginner-friendly explanations
    - Step-by-step math breakdowns
    - Smooth section transitions
    - Streaming support
    - Caching
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.cache = get_cache_service()
        
        logger.info(f"Lesson service initialized: model={self.settings.lesson_model}")
    
    def generate_lesson(
        self,
        paper: ParsedPaper,
        request: LessonRequest
    ) -> FullLesson:
        """
        Generate a full lesson from a parsed paper.
        
        Args:
            paper: Parsed paper with sections
            request: Lesson generation request
            
        Returns:
            Complete lesson with all fragments
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{paper.metadata.arxiv_id}:{request.difficulty}"
        cached = self.cache.get("lessons", cache_key)
        if cached:
            logger.info(f"Lesson cache hit: {paper.metadata.arxiv_id}")
            return FullLesson(**cached)
        
        # Generate fragments
        fragments = []
        sections = paper.sections
        
        if request.max_sections:
            sections = sections[:request.max_sections]
        
        for i, section in enumerate(sections):
            next_section = sections[i + 1] if i + 1 < len(sections) else None
            
            fragment = self._generate_fragment(
                section=section,
                next_section_name=next_section.name if next_section else None,
                request=request,
                order=i
            )
            fragments.append(fragment)
            logger.debug(f"Generated fragment {i + 1}/{len(sections)}: {section.name}")
        
        # Create full lesson
        lesson = FullLesson(
            paper_id=paper.metadata.arxiv_id,
            paper_title=paper.metadata.title,
            paper_url=str(paper.metadata.url),
            query=request.query,
            fragments=fragments,
            difficulty=request.difficulty,
            generation_time_seconds=time.time() - start_time
        )
        
        # Cache result
        self.cache.set("lessons", cache_key, lesson.model_dump(mode='json'))
        
        logger.info(
            f"Generated lesson for {paper.metadata.arxiv_id}: "
            f"{len(fragments)} sections, {lesson.total_read_time} min read time"
        )
        
        return lesson
    
    async def generate_lesson_streaming(
        self,
        paper: ParsedPaper,
        request: LessonRequest
    ) -> AsyncGenerator[StreamingLessonChunk, None]:
        """
        Generate lesson with streaming responses.
        
        Yields chunks as sections are generated.
        """
        start_time = time.time()
        
        # Send metadata first
        yield StreamingLessonChunk(
            type="metadata",
            data={
                "paper_id": paper.metadata.arxiv_id,
                "paper_title": paper.metadata.title,
                "paper_url": str(paper.metadata.url),
                "total_sections": len(paper.sections),
            }
        )
        
        sections = paper.sections
        if request.max_sections:
            sections = sections[:request.max_sections]
        
        for i, section in enumerate(sections):
            try:
                next_section = sections[i + 1] if i + 1 < len(sections) else None
                
                content = await self._generate_fragment_async(
                    section=section,
                    next_section_name=next_section.name if next_section else None,
                    request=request
                )
                
                yield StreamingLessonChunk(
                    type="section",
                    data={
                        "name": section.name,
                        "content": content,
                        "order": i,
                        "progress": (i + 1) / len(sections)
                    }
                )
            except Exception as e:
                logger.error(f"Error generating section {section.name}: {e}")
                yield StreamingLessonChunk(
                    type="error",
                    data={"message": f"Failed to generate section: {section.name}"}
                )
        
        # Send completion
        yield StreamingLessonChunk(
            type="done",
            data={
                "total_time_seconds": time.time() - start_time,
                "sections_generated": len(sections)
            }
        )
    
    def _generate_fragment(
        self,
        section: PaperSection,
        next_section_name: Optional[str],
        request: LessonRequest,
        order: int
    ) -> LessonFragment:
        """Generate a single lesson fragment."""
        prompt = self._build_prompt(section, next_section_name, request)
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.lesson_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            content = response.choices[0].message.content.strip()
        except Exception as e:
            raise LessonGenerationError(f"Failed to generate lesson: {e}", section.name)
        
        return LessonFragment(
            section_name=section.name,
            content=content,
            order=order,
            has_math=self._contains_math(content),
            has_code=self._contains_code(content)
        )
    
    async def _generate_fragment_async(
        self,
        section: PaperSection,
        next_section_name: Optional[str],
        request: LessonRequest
    ) -> str:
        """Generate a single lesson fragment asynchronously."""
        prompt = self._build_prompt(section, next_section_name, request)
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.settings.lesson_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise LessonGenerationError(f"Failed to generate lesson: {e}", section.name)
    
    def _build_prompt(
        self,
        section: PaperSection,
        next_section_name: Optional[str],
        request: LessonRequest
    ) -> str:
        """Build the prompt for lesson generation."""
        difficulty_instructions = {
            LessonDifficulty.BEGINNER: "Use simple language, avoid jargon, and explain everything from first principles.",
            LessonDifficulty.INTERMEDIATE: "Assume basic ML/CS knowledge but explain advanced concepts clearly.",
            LessonDifficulty.ADVANCED: "Be concise and technical, focusing on nuances and advanced insights."
        }
        
        prompt = f"""You are an expert teacher converting a research paper section into a {request.difficulty.value}-friendly lesson.

Section: "{section.name}"

Content:
{section.content}

Instructions:
- {difficulty_instructions[request.difficulty]}
"""
        
        if request.include_math:
            prompt += "- Break down any mathematical concepts step by step.\n"
        else:
            prompt += "- Minimize mathematical notation, focus on intuition.\n"
        
        if request.include_examples:
            prompt += "- Include concrete examples and analogies to illustrate concepts.\n"
        
        if next_section_name:
            prompt += f'\n- End with a smooth transition to the next section: "{next_section_name}".\n'
        
        prompt += "\nGenerate the lesson fragment now:"
        
        return prompt
    
    def _contains_math(self, content: str) -> bool:
        """Check if content contains mathematical notation."""
        math_indicators = ['$', '\\frac', '\\sum', '\\int', '∑', '∫', '√', 'equation']
        return any(ind in content.lower() for ind in math_indicators)
    
    def _contains_code(self, content: str) -> bool:
        """Check if content contains code."""
        code_indicators = ['```', 'def ', 'import ', 'class ', 'function']
        return any(ind in content for ind in code_indicators)
    
    def generate_single_section_lesson(
        self,
        section_name: str,
        section_text: str,
        next_section_name: Optional[str] = None,
        difficulty: LessonDifficulty = LessonDifficulty.BEGINNER
    ) -> str:
        """
        Generate a lesson for a single section (backwards compatible).
        
        Args:
            section_name: Name of the section
            section_text: Section content
            next_section_name: Next section for transition
            difficulty: Lesson difficulty
            
        Returns:
            Generated lesson text
        """
        section = PaperSection(name=section_name, content=section_text, order=0)
        request = LessonRequest(
            query="",
            difficulty=difficulty,
            include_examples=True,
            include_math=True
        )
        
        fragment = self._generate_fragment(section, next_section_name, request, 0)
        return fragment.content


# Singleton instance
_lesson_service: Optional[LessonService] = None


def get_lesson_service() -> LessonService:
    """Get the global lesson service instance."""
    global _lesson_service
    if _lesson_service is None:
        _lesson_service = LessonService()
    return _lesson_service
```

#### backend/services/paper_service.py
```python
"""
Paper Service

Paper retrieval, PDF processing, and section extraction.
"""

import re
import requests
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import (
    PaperNotFoundError,
    GROBIDError,
    ArxivError,
    PDFProcessingError,
)
from models.paper import PaperMetadata, PaperSection, ParsedPaper, PaperSearchResult
from services.embedding_service import get_embedding_service
from services.cache_service import get_cache_service

logger = get_logger(__name__)


class PaperService:
    """
    Service for paper retrieval and processing.
    
    Features:
    - Semantic search via FAISS
    - PDF parsing via GROBID
    - Fallback to abstract-only mode
    - Caching for performance
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.cache = get_cache_service()
        
        logger.info(f"Paper service initialized: grobid={self.settings.grobid_url}")
    
    def search(self, query: str, top_k: int = 1) -> list[PaperSearchResult]:
        """
        Search for papers matching a query.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of search results
        """
        results = self.embedding_service.search(query, k=top_k)
        
        search_results = []
        for idx, distance, url in results:
            # Convert L2 distance to similarity score (0-1)
            # Lower distance = higher similarity
            similarity = 1 / (1 + distance)
            
            arxiv_id = self._extract_arxiv_id(url)
            
            paper = PaperMetadata(
                arxiv_id=arxiv_id,
                title=f"Paper {arxiv_id}",  # Will be updated when fetched
                url=url
            )
            
            search_results.append(PaperSearchResult(
                paper=paper,
                similarity_score=similarity,
                index_position=idx
            ))
        
        return search_results
    
    def get_paper(self, url: str, use_grobid: bool = True) -> ParsedPaper:
        """
        Get a parsed paper from URL.
        
        Args:
            url: arXiv URL
            use_grobid: Whether to use GROBID for full parsing
            
        Returns:
            Parsed paper with sections
        """
        arxiv_id = self._extract_arxiv_id(url)
        
        # Check cache
        cache_key = f"{arxiv_id}:{'grobid' if use_grobid else 'abstract'}"
        cached = self.cache.get("papers", cache_key)
        if cached:
            logger.info(f"Paper cache hit: {arxiv_id}")
            return ParsedPaper(**cached)
        
        # Fetch metadata
        metadata = self._fetch_arxiv_metadata(arxiv_id)
        
        # Try GROBID if enabled
        sections = []
        parsing_method = "abstract"
        
        if use_grobid and self.settings.use_grobid:
            try:
                sections = self._parse_with_grobid(url)
                parsing_method = "grobid"
                logger.info(f"GROBID parsed {len(sections)} sections from {arxiv_id}")
            except Exception as e:
                logger.warning(f"GROBID failed, falling back to abstract: {e}")
        
        # Fallback: use abstract as single section
        if not sections and metadata.abstract:
            sections = [PaperSection(
                name="abstract",
                content=metadata.abstract,
                order=0
            )]
            parsing_method = "abstract"
        
        paper = ParsedPaper(
            metadata=metadata,
            sections=sections,
            parsing_method=parsing_method
        )
        
        # Cache result
        self.cache.set("papers", cache_key, paper.model_dump(mode='json'))
        
        return paper
    
    def _extract_arxiv_id(self, url: str) -> str:
        """Extract arXiv ID from URL."""
        match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)', url)
        if match:
            return match.group(1)
        
        # Try to extract from path
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        if path_parts:
            return path_parts[-1].replace(".pdf", "")
        
        return url
    
    def _fetch_arxiv_metadata(self, arxiv_id: str) -> PaperMetadata:
        """Fetch paper metadata from arXiv API."""
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
        
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ArxivError(f"Failed to fetch metadata: {e}")
        
        # Parse XML
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(response.content)
        entry = root.find('atom:entry', ns)
        
        if entry is None:
            raise PaperNotFoundError(arxiv_id)
        
        title = entry.find('atom:title', ns)
        summary = entry.find('atom:summary', ns)
        authors = entry.findall('atom:author/atom:name', ns)
        
        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=title.text.strip() if title is not None else f"Paper {arxiv_id}",
            url=f"https://arxiv.org/abs/{arxiv_id}",
            abstract=summary.text.strip() if summary is not None else None,
            authors=[a.text for a in authors if a.text]
        )
    
    def _parse_with_grobid(self, url: str) -> list[PaperSection]:
        """Parse paper using GROBID service."""
        # Convert to PDF URL
        if '/abs/' in url:
            arxiv_id = self._extract_arxiv_id(url)
            pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
        else:
            pdf_url = url
        
        # Download PDF
        logger.debug(f"Downloading PDF: {pdf_url}")
        try:
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
            pdf_bytes = response.content
        except requests.RequestException as e:
            raise PDFProcessingError(f"Failed to download PDF: {e}", url)
        
        # Send to GROBID
        logger.debug(f"Sending to GROBID: {self.settings.grobid_url}")
        try:
            files = {'input': ('paper.pdf', BytesIO(pdf_bytes), 'application/pdf')}
            response = requests.post(
                f'{self.settings.grobid_url}/api/processFulltextDocument',
                files=files,
                timeout=self.settings.grobid_timeout
            )
            response.raise_for_status()
            tei_xml = response.text
        except requests.RequestException as e:
            raise GROBIDError(f"GROBID processing failed: {e}")
        
        # Parse TEI XML
        return self._parse_tei_xml(tei_xml)
    
    def _parse_tei_xml(self, tei_xml: str) -> list[PaperSection]:
        """Parse TEI XML to extract sections."""
        TEI_NS = 'http://www.tei-c.org/ns/1.0'
        ET.register_namespace('tei', TEI_NS)
        
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as e:
            raise PDFProcessingError(f"Failed to parse TEI XML: {e}")
        
        sections = []
        order = 0
        
        for div in root.findall(f'.//{{{TEI_NS}}}div'):
            # Skip body-level divs
            if div.attrib.get('type') == 'body':
                continue
            
            # Get section name
            section_name = (
                div.attrib.get('type')
                or div.attrib.get('subtype')
                or next(
                    (h.text for h in div.findall(f'./{{{TEI_NS}}}head') if h.text),
                    None
                )
            )
            
            if not section_name:
                continue
            
            # Extract text content
            text_parts = []
            for element in div.iter():
                if element.text and element.tag != f'{{{TEI_NS}}}head':
                    text_parts.append(element.text.strip())
                if element.tail:
                    text_parts.append(element.tail.strip())
            
            content = ' '.join(part for part in text_parts if part)
            
            if content:
                sections.append(PaperSection(
                    name=section_name.lower(),
                    content=content,
                    order=order
                ))
                order += 1
        
        return sections
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "grobid_url": self.settings.grobid_url,
            "grobid_enabled": self.settings.use_grobid,
            "cache_stats": self.cache.get_stats()
        }


# Singleton instance
_paper_service: Optional[PaperService] = None


def get_paper_service() -> PaperService:
    """Get the global paper service instance."""
    global _paper_service
    if _paper_service is None:
        _paper_service = PaperService()
    return _paper_service
```

#### backend/services/query_service.py
```python
"""
Query Enhancement Service

Uses LLM to enhance user queries for better retrieval:
- Expand with related terms
- Detect user intent (explain, compare, simplify)
- Infer difficulty level
"""

import json
from typing import Optional
from pydantic import BaseModel
from openai import OpenAI

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


class EnhancedQuery(BaseModel):
    """Enhanced query with metadata."""
    original: str
    enhanced: str
    intent: str  # "explain", "compare", "summarize", "simplify", "deep_dive"
    detected_difficulty: str  # "beginner", "intermediate", "advanced"
    key_concepts: list[str]
    is_comparison: bool = False


class QueryService:
    """
    Enhance user queries for better retrieval.
    
    Features:
    - Query expansion with related terms
    - Intent detection (explain vs compare vs simplify)
    - Difficulty inference from phrasing
    - Key concept extraction
    """
    
    SYSTEM_PROMPT = """You are a query enhancement system for an academic paper search engine.

Given a user query about machine learning, AI, or computer science, analyze it and output JSON with:

{
    "enhanced": "expanded query with related technical terms for better search",
    "intent": "one of: explain, compare, summarize, simplify, deep_dive",
    "detected_difficulty": "one of: beginner, intermediate, advanced",
    "key_concepts": ["list", "of", "3-5", "key", "concepts"],
    "is_comparison": true/false
}

Intent Detection Rules:
- "ELI5", "simply", "basics", "beginner", "intro" → intent: "simplify", difficulty: "beginner"
- "Compare X vs Y", "difference between" → intent: "compare", is_comparison: true
- "Deep dive", "in-depth", "technical details" → intent: "deep_dive", difficulty: "advanced"
- "How does X work", "What is X" → intent: "explain"
- "Summarize", "overview", "brief" → intent: "summarize"

Query Enhancement Rules:
- Add related technical terms that would appear in academic papers
- Include synonyms and related concepts
- Keep the enhanced query concise (under 15 words)

Examples:
- "ELI5 attention" → enhanced: "attention mechanism transformer neural network basics introduction"
- "BERT vs GPT" → enhanced: "BERT GPT language model comparison pretraining architecture"
- "How do transformers work" → enhanced: "transformer architecture self-attention encoder decoder mechanism"
"""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
    
    def enhance_query(self, query: str) -> EnhancedQuery:
        """
        Enhance a user query for better retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            EnhancedQuery with expanded terms and metadata
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            
            enhanced = EnhancedQuery(
                original=query,
                enhanced=result.get("enhanced", query),
                intent=result.get("intent", "explain"),
                detected_difficulty=result.get("detected_difficulty", "beginner"),
                key_concepts=result.get("key_concepts", []),
                is_comparison=result.get("is_comparison", False)
            )
            
            logger.info(
                f"Enhanced query: '{query[:30]}...' → intent={enhanced.intent}, "
                f"difficulty={enhanced.detected_difficulty}"
            )
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            # Return basic enhancement on failure
            return EnhancedQuery(
                original=query,
                enhanced=query,
                intent="explain",
                detected_difficulty="beginner",
                key_concepts=query.split()[:5]
            )
    
    def quick_intent_detection(self, query: str) -> tuple[str, str]:
        """
        Fast intent detection without LLM call.
        
        Returns:
            (intent, difficulty)
        """
        query_lower = query.lower()
        
        # Simplify indicators
        if any(word in query_lower for word in ["eli5", "simple", "basics", "beginner", "intro"]):
            return "simplify", "beginner"
        
        # Comparison indicators
        if any(word in query_lower for word in [" vs ", " versus ", "compare", "difference"]):
            return "compare", "intermediate"
        
        # Deep dive indicators
        if any(word in query_lower for word in ["deep dive", "in-depth", "technical", "advanced"]):
            return "deep_dive", "advanced"
        
        # Summary indicators
        if any(word in query_lower for word in ["summarize", "overview", "brief", "tldr"]):
            return "summarize", "beginner"
        
        # Default
        return "explain", "beginner"


# Singleton instance
_query_service: Optional[QueryService] = None


def get_query_service() -> QueryService:
    """Get singleton Query service instance."""
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service
```

#### backend/services/scholar_service.py
```python
"""
Semantic Scholar API Service

Fetches academic papers dynamically when:
- User query doesn't match existing papers well
- Need to expand knowledge base

API Docs: https://api.semanticscholar.org/
Rate Limit: 100 requests per 5 minutes (free tier)
"""

import httpx
from typing import Optional
from datetime import date

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


class ScholarService:
    """
    Fetch academic papers from Semantic Scholar API.
    
    Features:
    - Free API (no scraping needed)
    - Structured data with abstracts
    - ArXiv integration
    - Rate limiting built-in
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self):
        self.settings = get_settings()
        self.client = httpx.Client(timeout=30.0)
        self._daily_fetch_count = 0
        self._last_reset_date = date.today()
    
    def _check_daily_limit(self) -> bool:
        """Check and reset daily limit if needed."""
        today = date.today()
        if today > self._last_reset_date:
            self._daily_fetch_count = 0
            self._last_reset_date = today
        
        return self._daily_fetch_count < self.settings.max_daily_fetches
    
    def search_papers(
        self,
        query: str,
        limit: int = 10,
        year_min: Optional[int] = None
    ) -> list[dict]:
        """
        Search for papers matching a query.
        
        Args:
            query: Search query
            limit: Max results (max 100)
            year_min: Minimum publication year
        
        Returns:
            List of paper objects with abstracts
        """
        if not self._check_daily_limit():
            logger.warning(f"Daily fetch limit reached ({self.settings.max_daily_fetches})")
            return []
        
        fields = [
            "paperId",
            "title",
            "abstract",
            "url",
            "year",
            "authors",
            "citationCount",
            "externalIds",
            "openAccessPdf"
        ]
        
        try:
            params = {
                "query": query,
                "limit": min(limit, 100),
                "fields": ",".join(fields)
            }
            
            if year_min:
                params["year"] = f"{year_min}-"
            
            # Add API key if available (higher rate limits)
            headers = {}
            if self.settings.semantic_scholar_api_key:
                headers["x-api-key"] = self.settings.semantic_scholar_api_key
            
            response = self.client.get(
                f"{self.BASE_URL}/paper/search",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            papers = data.get("data", [])
            
            # Filter: must have abstract for embedding
            papers_with_abstracts = [
                p for p in papers
                if p.get("abstract") and len(p.get("abstract", "")) > 100
            ]
            
            self._daily_fetch_count += 1
            logger.info(
                f"Fetched {len(papers_with_abstracts)}/{len(papers)} papers "
                f"for query: '{query[:50]}...' (daily: {self._daily_fetch_count}/{self.settings.max_daily_fetches})"
            )
            
            return papers_with_abstracts
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Semantic Scholar rate limit hit, backing off")
            else:
                logger.error(f"Semantic Scholar API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Semantic Scholar error: {e}")
            return []
    
    def get_paper_by_id(self, paper_id: str) -> Optional[dict]:
        """Get a specific paper by Semantic Scholar ID."""
        try:
            response = self.client.get(
                f"{self.BASE_URL}/paper/{paper_id}",
                params={
                    "fields": "paperId,title,abstract,url,year,authors,externalIds,openAccessPdf"
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return None
    
    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[dict]:
        """Get a paper by ArXiv ID."""
        try:
            response = self.client.get(
                f"{self.BASE_URL}/paper/arXiv:{arxiv_id}",
                params={
                    "fields": "paperId,title,abstract,url,year,authors,externalIds"
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return None
    
    @staticmethod
    def extract_arxiv_id(paper: dict) -> Optional[str]:
        """Extract ArXiv ID from paper data."""
        external_ids = paper.get("externalIds", {})
        return external_ids.get("ArXiv")
    
    @staticmethod
    def get_arxiv_url(paper: dict) -> Optional[str]:
        """Get ArXiv URL if available."""
        arxiv_id = ScholarService.extract_arxiv_id(paper)
        if arxiv_id:
            return f"https://arxiv.org/abs/{arxiv_id}"
        return paper.get("url")
    
    @staticmethod
    def get_pdf_url(paper: dict) -> Optional[str]:
        """Get PDF URL if available."""
        # Try open access PDF first
        open_access = paper.get("openAccessPdf", {})
        if open_access and open_access.get("url"):
            return open_access["url"]
        
        # Try ArXiv PDF
        arxiv_id = ScholarService.extract_arxiv_id(paper)
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        return None
    
    def get_daily_stats(self) -> dict:
        """Get daily usage statistics."""
        return {
            "fetches_today": self._daily_fetch_count,
            "max_daily": self.settings.max_daily_fetches,
            "remaining": self.settings.max_daily_fetches - self._daily_fetch_count
        }


# Singleton instance
_scholar_service: Optional[ScholarService] = None


def get_scholar_service() -> ScholarService:
    """Get singleton Scholar service instance."""
    global _scholar_service
    if _scholar_service is None:
        _scholar_service = ScholarService()
    return _scholar_service
```

#### backend/services/teaching_service.py
```python
"""
Teaching Service v2

Main orchestration service with:
- Relevance threshold checking
- Dynamic paper fetching
- Query enhancement integration
- Removed LeetCode dependency
"""

import time
from typing import Optional, AsyncGenerator

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import PaperNotFoundError
from models.paper import PaperSearchResult, ParsedPaper
from models.lesson import (
    LessonRequest,
    LessonResponse,
    FullLesson,
    StreamingLessonChunk,
)
from services.paper_service import get_paper_service
from services.lesson_service import get_lesson_service
from services.cache_service import get_cache_service
from services.query_service import get_query_service, EnhancedQuery
from services.scholar_service import get_scholar_service
from services.embedding_service import get_embedding_service

logger = get_logger(__name__)


class NoRelevantPapersError(Exception):
    """Raised when no relevant papers found for a query."""
    pass


class TeachingService:
    """
    Main teaching service v2 that orchestrates all functionality.
    
    New in v2:
    - Relevance threshold checking (0.50/0.35/0.20)
    - Dynamic paper fetching from Semantic Scholar
    - Query enhancement for better search
    - Removed LeetCode integration
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.paper_service = get_paper_service()
        self.lesson_service = get_lesson_service()
        self.cache = get_cache_service()
        self.query_service = get_query_service()
        self.scholar_service = get_scholar_service()
        self.embedding_service = get_embedding_service()
        
        logger.info(
            f"Teaching service v2 initialized: "
            f"thresholds=({self.settings.high_relevance_threshold}/"
            f"{self.settings.medium_relevance_threshold}), "
            f"dynamic_fetch={self.settings.dynamic_fetch_enabled}"
        )
    
    def teach(self, request: LessonRequest, use_enhancement: bool = True) -> LessonResponse:
        """
        Main teaching endpoint v2.
        
        Flow:
        1. Enhance query (detect intent, difficulty)
        2. Search for relevant paper
        3. Check relevance threshold
        4. Dynamic fetch if needed
        5. Generate lesson
        
        Args:
            request: Lesson request with query and preferences
            use_enhancement: Whether to use LLM query enhancement
            
        Returns:
            Complete lesson response
        """
        start_time = time.time()
        
        try:
            query = request.query
            logger.info(f"Teaching request v2: {query[:50]}...")
            
            # Step 1: Enhance query
            if use_enhancement:
                enhanced = self.query_service.enhance_query(query)
                search_query = enhanced.enhanced
                detected_difficulty = enhanced.detected_difficulty
                logger.info(f"Enhanced: '{query[:30]}' → '{search_query[:50]}' (intent={enhanced.intent})")
            else:
                search_query = query
                detected_difficulty = request.difficulty.value if request.difficulty else "beginner"
            
            # Step 2: Search for papers
            search_results = self.paper_service.search(search_query, top_k=3)
            
            if not search_results:
                raise PaperNotFoundError(query)
            
            best_result = search_results[0]
            best_score = best_result.similarity_score
            logger.info(f"Initial search: {best_result.paper.arxiv_id} (score: {best_score:.3f})")
            
            # Step 3: Handle relevance threshold
            final_result = self._handle_relevance(
                query=query,
                search_query=search_query,
                search_results=search_results,
                best_score=best_score
            )
            
            if final_result is None:
                raise NoRelevantPapersError(
                    f"No relevant papers found for: {query}. "
                    f"Best match score: {best_score:.2f} (threshold: {self.settings.medium_relevance_threshold})"
                )
            
            # Step 4: Get full paper
            paper = self.paper_service.get_paper(
                str(final_result.paper.url),
                use_grobid=self.settings.use_grobid
            )
            
            # Step 5: Generate lesson
            lesson = self.lesson_service.generate_lesson(paper, request)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return LessonResponse(
                success=True,
                lesson=lesson,
                cached=False,
                processing_time_ms=processing_time
            )
            
        except NoRelevantPapersError as e:
            logger.warning(f"No relevant papers: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            return LessonResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Teaching failed: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            
            return LessonResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time
            )
    
    def _handle_relevance(
        self,
        query: str,
        search_query: str,
        search_results: list[PaperSearchResult],
        best_score: float
    ) -> Optional[PaperSearchResult]:
        """
        Handle relevance-based routing.
        
        - High relevance (>= 0.50): Use directly
        - Medium relevance (0.35-0.50): Use but try to improve
        - Low relevance (< 0.35): Must find new papers
        
        Returns:
            Best PaperSearchResult or None if no good match
        """
        settings = self.settings
        
        # High relevance - use directly
        if best_score >= settings.high_relevance_threshold:
            logger.info(f"✓ High relevance ({best_score:.2f} >= {settings.high_relevance_threshold})")
            return search_results[0]
        
        # Medium relevance - try to improve silently
        if best_score >= settings.medium_relevance_threshold:
            logger.info(f"~ Medium relevance ({best_score:.2f}), attempting to improve")
            
            if settings.dynamic_fetch_enabled:
                new_papers = self._fetch_and_add_papers(query)
                
                if new_papers:
                    # Re-search with new papers
                    new_results = self.paper_service.search(search_query, top_k=3)
                    if new_results and new_results[0].similarity_score > best_score:
                        logger.info(f"✓ Found better match: {new_results[0].similarity_score:.2f}")
                        return new_results[0]
            
            # Use original if no improvement
            logger.info(f"Using original medium-relevance match")
            return search_results[0]
        
        # Low relevance - must fetch new papers
        logger.info(f"✗ Low relevance ({best_score:.2f} < {settings.medium_relevance_threshold})")
        
        if not settings.dynamic_fetch_enabled:
            logger.warning("Dynamic fetch disabled, using low-relevance match")
            return search_results[0]
        
        new_papers = self._fetch_and_add_papers(query)
        
        if new_papers:
            # Re-search
            new_results = self.paper_service.search(search_query, top_k=3)
            
            if new_results:
                new_score = new_results[0].similarity_score
                if new_score >= settings.medium_relevance_threshold:
                    logger.info(f"✓ Found relevant paper after fetch: {new_score:.2f}")
                    return new_results[0]
                else:
                    logger.warning(f"Still low relevance after fetch: {new_score:.2f}")
                    return new_results[0]  # Return best available
        
        # No improvement possible
        logger.warning("No new papers found, returning None")
        return None
    
    def _fetch_and_add_papers(self, query: str) -> list[dict]:
        """
        Fetch papers from Semantic Scholar and add to index.
        
        Returns:
            List of newly added papers
        """
        logger.info(f"Fetching papers from Semantic Scholar for: {query[:30]}...")
        
        # Fetch from Semantic Scholar
        papers = self.scholar_service.search_papers(
            query=query,
            limit=self.settings.max_papers_per_fetch
        )
        
        if not papers:
            logger.info("No papers found from Semantic Scholar")
            return []
        
        # Prepare for indexing
        new_papers = []
        
        for paper in papers:
            paper_id = paper.get("paperId")
            abstract = paper.get("abstract", "")
            title = paper.get("title", "")
            
            if not abstract or not paper_id:
                continue
            
            try:
                # Create embedding
                embedding = self.embedding_service.create_embedding(abstract)
                
                # Get URL
                url = self.scholar_service.get_arxiv_url(paper)
                if not url:
                    url = f"https://www.semanticscholar.org/paper/{paper_id}"
                
                new_papers.append({
                    "id": f"semantic_{paper_id}",
                    "embedding": embedding,
                    "url": url,
                    "title": title,
                    "abstract": abstract[:1000],
                    "source": "semantic_scholar",
                    "year": paper.get("year"),
                    "citation_count": paper.get("citationCount", 0)
                })
                
            except Exception as e:
                logger.warning(f"Failed to process paper {paper_id}: {e}")
        
        # Add to embedding service index
        if new_papers:
            added = self._add_papers_to_index(new_papers)
            logger.info(f"Dynamically added {added} papers for: {query[:30]}...")
        
        return new_papers
    
    def _add_papers_to_index(self, papers: list[dict]) -> int:
        """Add papers to the FAISS index."""
        import numpy as np
        import json
        import faiss
        
        try:
            # Get current index and URLs
            index = self.embedding_service.index
            urls = list(self.embedding_service.urls)
            
            # Stack new embeddings
            embeddings = np.vstack([p["embedding"] for p in papers]).astype("float32")
            
            # Add to index
            index.add(embeddings)
            
            # Add URLs
            for paper in papers:
                urls.append(paper["url"])
            
            # Update service state
            self.embedding_service._urls = urls
            
            # Save to disk
            faiss.write_index(index, str(self.settings.faiss_index_path))
            with open(self.settings.urls_json_path, "w") as f:
                json.dump(urls, f)
            
            logger.info(f"Index updated: {index.ntotal} total vectors")
            return len(papers)
            
        except Exception as e:
            logger.error(f"Failed to add papers to index: {e}")
            return 0
    
    async def teach_streaming(
        self,
        request: LessonRequest
    ) -> AsyncGenerator[StreamingLessonChunk, None]:
        """
        Streaming teaching endpoint - yields chunks as they're generated.
        """
        try:
            query = request.query
            
            # Enhance query
            enhanced = self.query_service.enhance_query(query)
            search_query = enhanced.enhanced
            
            # Search for paper
            search_results = self.paper_service.search(search_query, top_k=1)
            
            if not search_results:
                yield StreamingLessonChunk(
                    type="error",
                    data={"message": f"No papers found for: {query}"}
                )
                return
            
            best_result = search_results[0]
            
            # Get paper
            paper = self.paper_service.get_paper(
                str(best_result.paper.url),
                use_grobid=self.settings.use_grobid
            )
            
            # Stream lesson generation
            async for chunk in self.lesson_service.generate_lesson_streaming(paper, request):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming teaching failed: {e}")
            yield StreamingLessonChunk(
                type="error",
                data={"message": str(e)}
            )
    
    def search_papers(self, query: str, top_k: int = 5) -> list[PaperSearchResult]:
        """
        Search for papers without generating lessons.
        """
        # Enhance query first
        enhanced = self.query_service.enhance_query(query)
        return self.paper_service.search(enhanced.enhanced, top_k=top_k)
    
    def get_paper_details(self, url: str) -> ParsedPaper:
        """
        Get full paper details.
        """
        return self.paper_service.get_paper(url)
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "paper_service": self.paper_service.get_stats(),
            "scholar_service": self.scholar_service.get_daily_stats(),
            "cache": self.cache.get_stats(),
            "thresholds": {
                "high": self.settings.high_relevance_threshold,
                "medium": self.settings.medium_relevance_threshold,
                "low": self.settings.low_relevance_threshold
            },
            "dynamic_fetch_enabled": self.settings.dynamic_fetch_enabled,
            "index_size": self.embedding_service.index.ntotal if self.embedding_service._index else 0
        }


# Singleton instance
_teaching_service: Optional[TeachingService] = None


def get_teaching_service() -> TeachingService:
    """Get the global teaching service instance."""
    global _teaching_service
    if _teaching_service is None:
        _teaching_service = TeachingService()
    return _teaching_service
```

### 2.4 API

#### backend/api/main.py
```python
"""
FastAPI Application v2

Main entry point for the LLM Teaching Assistant API.

Changes in v2:
- Removed LeetCode routes
- Updated description
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import BaseAppException
from api.routes import teach, health

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Dynamic fetch: {settings.dynamic_fetch_enabled}")
    logger.info(f"Thresholds: high={settings.high_relevance_threshold}, medium={settings.medium_relevance_threshold}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        🎓 **LLM Teaching Assistant API v2**
        
        An AI-powered teaching assistant that:
        - Retrieves and explains research papers from arXiv
        - Converts academic content into beginner-friendly lessons
        - **Dynamically fetches new papers** when no good match exists
        - **Enhances queries** for better search results
        
        ## What's New in v2
        
        - **Relevance Thresholds**: Only uses papers above quality threshold
        - **Dynamic Fetching**: Searches Semantic Scholar when needed
        - **Query Enhancement**: LLM improves search queries
        - **Intent Detection**: Adapts to explain/compare/simplify requests
        
        ## Quick Start
        
        ```python
        import requests
        
        # Generate a lesson
        response = requests.post(
            "http://localhost:8000/api/v1/teach",
            json={"query": "Explain attention mechanisms"}
        )
        lesson = response.json()
        ```
        
        ## Endpoints
        
        - `POST /api/v1/teach` - Generate a lesson
        - `POST /api/v1/teach/stream` - Stream lesson generation
        - `GET /api/v1/stats` - Service statistics
        - `GET /health` - Health check
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Exception handler
    @app.exception_handler(BaseAppException)
    async def app_exception_handler(request: Request, exc: BaseAppException):
        logger.error(f"Application error: {exc.code} - {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    # Generic exception handler
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred"
                }
            }
        )
    
    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(teach.router, prefix=settings.api_prefix, tags=["Teaching"])
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
```

#### backend/api/routes/__init__.py
```python
"""
API Routes Package v2

Changes:
- Removed leetcode routes
"""

from . import health
from . import teach

__all__ = ["health", "teach"]
```

#### backend/api/routes/health.py
```python
"""
Health Check Routes

Endpoints for monitoring application health.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from core.config import get_settings
from services.teaching_service import get_teaching_service

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    service: str


class DetailedHealthResponse(BaseModel):
    """Detailed health check with service stats."""
    status: str
    version: str
    service: str
    stats: dict


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        Basic health status
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        service=settings.app_name
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Detailed health check with service statistics.
    
    Returns:
        Detailed health status with stats
    """
    settings = get_settings()
    
    try:
        teaching_service = get_teaching_service()
        stats = teaching_service.get_stats()
        status = "healthy"
    except Exception as e:
        stats = {"error": str(e)}
        status = "degraded"
    
    return DetailedHealthResponse(
        status=status,
        version=settings.app_version,
        service=settings.app_name,
        stats=stats
    )


@router.get("/")
async def root():
    """Root endpoint with API information."""
    settings = get_settings()
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }
```

#### backend/api/routes/teach.py
```python
"""
Teaching API Routes v2

Endpoints:
- POST /teach - Generate lesson (with query enhancement)
- POST /teach/stream - Stream lesson generation
- GET /stats - Service statistics
- POST /search - Search papers

Changes in v2:
- Added /stats endpoint
- Added /search endpoint
- Query enhancement by default
"""

import json
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.logging import get_logger
from core.exceptions import PaperNotFoundError
from models.lesson import LessonRequest, LessonResponse, LessonDifficulty
from services.teaching_service import get_teaching_service

logger = get_logger(__name__)

router = APIRouter()


class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    """Search response model."""
    success: bool
    query: str
    results: list[dict]


@router.post("/teach", response_model=LessonResponse)
async def generate_lesson(request: LessonRequest):
    """
    Generate a lesson for a query.
    
    v2 Features:
    - Automatically enhances query for better search
    - Checks relevance threshold (0.50/0.35)
    - Dynamically fetches papers if no good match
    
    Args:
        request: LessonRequest with query and preferences
        
    Returns:
        LessonResponse with generated lesson or error
    """
    logger.info(f"POST /teach: {request.query[:50]}...")
    
    try:
        teaching_service = get_teaching_service()
        response = teaching_service.teach(request)
        return response
        
    except Exception as e:
        logger.error(f"Lesson generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/teach/stream")
async def stream_lesson(request: LessonRequest):
    """
    Stream lesson generation via Server-Sent Events.
    
    Yields chunks as sections are generated.
    """
    logger.info(f"POST /teach/stream: {request.query[:50]}...")
    
    teaching_service = get_teaching_service()
    
    async def event_generator():
        async for chunk in teaching_service.teach_streaming(request):
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@router.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """
    Search for papers without generating lessons.
    
    Useful for:
    - Previewing available papers
    - Checking relevance scores
    - Exploring the index
    """
    logger.info(f"POST /search: {request.query[:50]}...")
    
    try:
        teaching_service = get_teaching_service()
        results = teaching_service.search_papers(request.query, top_k=request.top_k)
        
        return SearchResponse(
            success=True,
            query=request.query,
            results=[
                {
                    "arxiv_id": r.paper.arxiv_id,
                    "title": r.paper.title,
                    "url": str(r.paper.url),
                    "similarity_score": r.similarity_score
                }
                for r in results
            ]
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return SearchResponse(
            success=False,
            query=request.query,
            results=[]
        )


@router.get("/stats")
async def get_stats():
    """
    Get service statistics.
    
    Returns:
    - Index size (number of papers)
    - Daily fetch usage
    - Threshold settings
    - Cache stats
    """
    try:
        teaching_service = get_teaching_service()
        stats = teaching_service.get_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
```

### 2.5 Backend Config Files

#### backend/requirements.txt
```
# Core
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# AI/ML
openai>=1.10.0
faiss-cpu>=1.7.4
numpy>=1.24.0

# HTTP
requests>=2.31.0
httpx>=0.26.0
aiohttp>=3.9.0

# Parsing
beautifulsoup4>=4.12.0
lxml>=5.1.0

# Utilities
python-dotenv>=1.0.0
python-multipart>=0.0.6

# Development
pytest>=7.4.0
pytest-asyncio>=0.23.0
black>=24.0.0
isort>=5.13.0
mypy>=1.8.0

# Optional: LangGraph (if you want to keep agent functionality)
# langgraph>=0.0.20
# langchain>=0.1.0
# langmem>=0.0.30
```

#### backend/.env.example
```
# =============================================================================
# LLM Teaching Assistant v2 - Environment Configuration
# =============================================================================
# Copy this file to .env and fill in your values

# =============================================================================
# REQUIRED
# =============================================================================

# OpenAI API Key (get from https://platform.openai.com)
OPENAI_API_KEY=sk-your-key-here

# =============================================================================
# RELEVANCE THRESHOLDS (NEW in v2)
# =============================================================================

# Score above this = high relevance, use directly
HIGH_RELEVANCE_THRESHOLD=0.50

# Score above this = medium relevance, use but try to improve  
MEDIUM_RELEVANCE_THRESHOLD=0.35

# Score below this = irrelevant
LOW_RELEVANCE_THRESHOLD=0.20

# =============================================================================
# DYNAMIC PAPER FETCHING (NEW in v2)
# =============================================================================

# Enable fetching new papers when no good match found
DYNAMIC_FETCH_ENABLED=true

# Max papers to fetch per query from Semantic Scholar
MAX_PAPERS_PER_FETCH=10

# Max Semantic Scholar API calls per day (cost control)
MAX_DAILY_FETCHES=100

# Optional: Semantic Scholar API key (for higher rate limits)
# Get from: https://www.semanticscholar.org/product/api
SEMANTIC_SCHOLAR_API_KEY=

# =============================================================================
# PINECONE (OPTIONAL - for production)
# =============================================================================

# Use Pinecone instead of FAISS (for persistent dynamic updates)
USE_PINECONE=false

# Pinecone settings (required if USE_PINECONE=true)
# Get from https://www.pinecone.io/
PINECONE_API_KEY=
PINECONE_INDEX_NAME=llm-teaching-assistant
PINECONE_ENVIRONMENT=us-east-1

# =============================================================================
# API Settings
# =============================================================================

API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# =============================================================================
# Model Settings
# =============================================================================

EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o
LESSON_MODEL=gpt-4o-mini

# =============================================================================
# GROBID (PDF Parsing)
# =============================================================================

GROBID_URL=https://kermitt2-grobid.hf.space
GROBID_TIMEOUT=120
USE_GROBID=true

# =============================================================================
# Cache Settings
# =============================================================================

CACHE_ENABLED=true
CACHE_TTL=86400

# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL=INFO
LOG_FORMAT=text
```

## 3. Frontend Code

### 3.1 Entry Points

#### frontend/src/main.tsx
```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/globals.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

#### frontend/src/App.tsx
```tsx
import { useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import { ThemeProvider } from '@/hooks/useTheme'
import { Header, Hero, LessonDisplay, ProblemDisplay, LoadingOverlay } from '@/components'
import { generateLesson, getRandomProblem, Lesson, Problem, LessonRequest } from '@/lib/api'

type ViewState = 
  | { type: 'home' }
  | { type: 'loading'; message: string }
  | { type: 'lesson'; lesson: Lesson }
  | { type: 'problem'; problem: Problem }
  | { type: 'error'; message: string }

export default function App() {
  const [viewState, setViewState] = useState<ViewState>({ type: 'home' })
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (query: string, mode: 'learn' | 'code') => {
    setIsLoading(true)

    try {
      if (mode === 'learn') {
        setViewState({ type: 'loading', message: 'Searching for relevant papers...' })
        
        const request: LessonRequest = {
          query,
          difficulty: 'beginner',
          include_examples: true,
          include_math: true,
          max_sections: 5,
        }

        const response = await generateLesson(request)

        if (response.success && response.lesson) {
          setViewState({ type: 'lesson', lesson: response.lesson })
        } else {
          setViewState({ 
            type: 'error', 
            message: response.error || 'Failed to generate lesson' 
          })
        }
      } else {
        setViewState({ type: 'loading', message: 'Finding a coding challenge...' })
        
        const response = await getRandomProblem()

        if (response.success && response.problem) {
          setViewState({ type: 'problem', problem: response.problem })
        } else {
          setViewState({ 
            type: 'error', 
            message: response.error || 'Failed to fetch problem' 
          })
        }
      }
    } catch (error) {
      console.error('Error:', error)
      setViewState({ 
        type: 'error', 
        message: 'Something went wrong. Please try again.' 
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleClose = () => {
    setViewState({ type: 'home' })
  }

  const handleNewProblem = async () => {
    setIsLoading(true)
    try {
      const response = await getRandomProblem()
      if (response.success && response.problem) {
        setViewState({ type: 'problem', problem: response.problem })
      }
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <ThemeProvider>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-950 transition-colors">
        <Header />
        
        <main>
          <Hero onSubmit={handleSubmit} isLoading={isLoading} />
        </main>

        <AnimatePresence>
          {viewState.type === 'loading' && (
            <LoadingOverlay message={viewState.message} />
          )}

          {viewState.type === 'lesson' && (
            <LessonDisplay 
              lesson={viewState.lesson} 
              onClose={handleClose} 
            />
          )}

          {viewState.type === 'problem' && (
            <ProblemDisplay
              problem={viewState.problem}
              onClose={handleClose}
              onNewProblem={handleNewProblem}
              isLoading={isLoading}
            />
          )}
        </AnimatePresence>

        <AnimatePresence>
          {viewState.type === 'error' && (
            <div className="fixed bottom-4 right-4 z-50">
              <div className="bg-red-500 text-white px-4 py-3 rounded-xl shadow-lg flex items-center gap-3">
                <span>{viewState.message}</span>
                <button
                  onClick={handleClose}
                  className="text-white/80 hover:text-white"
                >
                  ✕
                </button>
              </div>
            </div>
          )}
        </AnimatePresence>
      </div>
    </ThemeProvider>
  )
}
```

### 3.2 Components

#### frontend/src/components/Button.tsx
```tsx
import { forwardRef, ButtonHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'outline'
  size?: 'sm' | 'md' | 'lg'
  isLoading?: boolean
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', isLoading, children, disabled, ...props }, ref) => {
    const baseStyles = 'inline-flex items-center justify-center font-medium rounded-xl transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed'
    
    const variants = {
      primary: 'bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-600 hover:to-primary-700 text-white shadow-lg shadow-primary-500/25 hover:shadow-xl hover:shadow-primary-500/30 focus:ring-primary-500',
      secondary: 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-gray-500',
      ghost: 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300 focus:ring-gray-500',
      outline: 'border-2 border-primary-500 text-primary-500 hover:bg-primary-50 dark:hover:bg-primary-950 focus:ring-primary-500',
    }
    
    const sizes = {
      sm: 'px-3 py-1.5 text-sm',
      md: 'px-4 py-2 text-sm',
      lg: 'px-6 py-3 text-base',
    }

    return (
      <button
        ref={ref}
        className={cn(baseStyles, variants[variant], sizes[size], className)}
        disabled={disabled || isLoading}
        {...props}
      >
        {isLoading && (
          <svg
            className="animate-spin -ml-1 mr-2 h-4 w-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        )}
        {children}
      </button>
    )
  }
)

Button.displayName = 'Button'

export default Button
```

#### frontend/src/components/Card.tsx
```tsx
import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface CardProps {
  children: ReactNode
  className?: string
  hover?: boolean
  glass?: boolean
}

export function Card({ children, className, hover = false, glass = false }: CardProps) {
  return (
    <div
      className={cn(
        'rounded-2xl border',
        glass 
          ? 'glass glass-border' 
          : 'bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-800',
        hover && 'transition-all duration-300 hover:shadow-xl hover:shadow-primary-500/5 hover:-translate-y-1',
        className
      )}
    >
      {children}
    </div>
  )
}

export function CardHeader({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('px-6 py-4 border-b border-gray-200 dark:border-gray-800', className)}>
      {children}
    </div>
  )
}

export function CardContent({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('px-6 py-4', className)}>
      {children}
    </div>
  )
}
```

#### frontend/src/components/Header.tsx
```tsx
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Sun, Moon, Menu, X, GraduationCap } from 'lucide-react'
import { useTheme } from '@/hooks/useTheme'

interface NavItem {
  label: string
  href: string
}

const navItems: NavItem[] = [
  { label: 'Learn', href: '#learn' },
  { label: 'Practice', href: '#practice' },
  { label: 'Search', href: '#search' },
]

export default function Header() {
  const { setTheme, resolvedTheme } = useTheme()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const toggleTheme = () => {
    setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')
  }

  return (
    <header className="fixed top-0 left-0 right-0 z-50">
      <div className="glass glass-border mx-4 mt-4 rounded-2xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <motion.a
              href="/"
              className="flex items-center space-x-2"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center">
                <GraduationCap className="w-6 h-6 text-white" />
              </div>
              <span className="font-bold text-xl text-gradient hidden sm:block">
                LearnAI
              </span>
            </motion.a>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center space-x-1">
              {navItems.map((item) => (
                <a
                  key={item.label}
                  href={item.href}
                  className="px-4 py-2 rounded-lg text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                >
                  {item.label}
                </a>
              ))}
            </nav>

            {/* Right side */}
            <div className="flex items-center space-x-2">
              {/* Theme toggle */}
              <motion.button
                onClick={toggleTheme}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <AnimatePresence mode="wait">
                  {resolvedTheme === 'dark' ? (
                    <motion.div
                      key="sun"
                      initial={{ rotate: -90, opacity: 0 }}
                      animate={{ rotate: 0, opacity: 1 }}
                      exit={{ rotate: 90, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <Sun className="w-5 h-5 text-yellow-500" />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="moon"
                      initial={{ rotate: 90, opacity: 0 }}
                      animate={{ rotate: 0, opacity: 1 }}
                      exit={{ rotate: -90, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <Moon className="w-5 h-5 text-gray-600" />
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.button>

              {/* Mobile menu button */}
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="md:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
              >
                {mobileMenuOpen ? (
                  <X className="w-5 h-5" />
                ) : (
                  <Menu className="w-5 h-5" />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile menu */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden border-t border-gray-200 dark:border-gray-700"
            >
              <div className="px-4 py-3 space-y-1">
                {navItems.map((item) => (
                  <a
                    key={item.label}
                    href={item.href}
                    className="block px-4 py-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    {item.label}
                  </a>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </header>
  )
}
```

#### frontend/src/components/Hero.tsx
```tsx
import { useState } from 'react'
import { motion } from 'framer-motion'
import { Sparkles, ArrowRight, BookOpen, Code } from 'lucide-react'
import Button from './Button'
import { Textarea } from './Input'

interface HeroProps {
  onSubmit: (query: string, mode: 'learn' | 'code') => void
  isLoading: boolean
}

export default function Hero({ onSubmit, isLoading }: HeroProps) {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<'learn' | 'code'>('learn')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      onSubmit(query, mode)
    }
  }

  const suggestions = [
    'Explain attention mechanisms in transformers',
    'How does BERT pre-training work?',
    'What is LoRA fine-tuning?',
    'Explain the GPT architecture',
  ]

  return (
    <section className="relative min-h-screen flex items-center justify-center px-4 pt-24 pb-12">
      {/* Background decoration */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent-500/10 rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-4xl mx-auto text-center">
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-50 dark:bg-primary-950 text-primary-600 dark:text-primary-400 text-sm font-medium mb-6"
        >
          <Sparkles className="w-4 h-4" />
          <span>AI-Powered Learning</span>
        </motion.div>

        {/* Heading */}
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight mb-6"
        >
          Learn AI Research
          <br />
          <span className="text-gradient">The Easy Way</span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-lg sm:text-xl text-gray-600 dark:text-gray-400 mb-8 max-w-2xl mx-auto"
        >
          Transform complex research papers into beginner-friendly lessons.
          Practice coding with curated LeetCode problems.
        </motion.p>

        {/* Mode switcher */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="flex justify-center gap-2 mb-6"
        >
          <button
            onClick={() => setMode('learn')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
              mode === 'learn'
                ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/25'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <BookOpen className="w-4 h-4" />
            Learn
          </button>
          <button
            onClick={() => setMode('code')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
              mode === 'code'
                ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/25'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <Code className="w-4 h-4" />
            Practice
          </button>
        </motion.div>

        {/* Input form */}
        <motion.form
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          onSubmit={handleSubmit}
          className="relative max-w-2xl mx-auto"
        >
          <div className="relative">
            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                mode === 'learn'
                  ? 'What would you like to learn about? (e.g., "Explain transformers")'
                  : 'Describe what you want to practice...'
              }
              rows={3}
              className="pr-24 text-lg"
            />
            <div className="absolute right-2 bottom-2">
              <Button
                type="submit"
                isLoading={isLoading}
                disabled={!query.trim()}
                className="rounded-xl"
              >
                {isLoading ? (
                  'Generating...'
                ) : (
                  <>
                    Go
                    <ArrowRight className="w-4 h-4 ml-1" />
                  </>
                )}
              </Button>
            </div>
          </div>
        </motion.form>

        {/* Suggestions */}
        {mode === 'learn' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="mt-6 flex flex-wrap justify-center gap-2"
          >
            <span className="text-sm text-gray-500 dark:text-gray-400 mr-2">
              Try:
            </span>
            {suggestions.map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => setQuery(suggestion)}
                className="text-sm px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                {suggestion}
              </button>
            ))}
          </motion.div>
        )}
      </div>
    </section>
  )
}
```

#### frontend/src/components/Input.tsx
```tsx
import { forwardRef, InputHTMLAttributes, TextareaHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={cn(
          'w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700',
          'bg-white dark:bg-gray-800',
          'text-gray-900 dark:text-gray-100',
          'placeholder:text-gray-400 dark:placeholder:text-gray-500',
          'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
          'transition-all duration-200',
          className
        )}
        {...props}
      />
    )
  }
)

Input.displayName = 'Input'

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        ref={ref}
        className={cn(
          'w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700',
          'bg-white dark:bg-gray-800',
          'text-gray-900 dark:text-gray-100',
          'placeholder:text-gray-400 dark:placeholder:text-gray-500',
          'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
          'transition-all duration-200 resize-none',
          className
        )}
        {...props}
      />
    )
  }
)

Textarea.displayName = 'Textarea'
```

#### frontend/src/components/LessonDisplay.tsx
```tsx
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { 
  BookOpen, 
  Clock, 
  ExternalLink, 
  ChevronDown, 
  ChevronUp,
  Copy,
  Check,
  X
} from 'lucide-react'
import { Card, CardContent, CardHeader } from './Card'
import Button from './Button'
import { Lesson, LessonFragment } from '@/lib/api'
import { formatReadTime } from '@/lib/utils'

interface LessonDisplayProps {
  lesson: Lesson
  onClose: () => void
}

export default function LessonDisplay({ lesson, onClose }: LessonDisplayProps) {
  const [expandedSections, setExpandedSections] = useState<Set<number>>(
    new Set(lesson.fragments.map((_, i) => i))
  )
  const [copied, setCopied] = useState(false)

  const toggleSection = (index: number) => {
    const newExpanded = new Set(expandedSections)
    if (newExpanded.has(index)) {
      newExpanded.delete(index)
    } else {
      newExpanded.add(index)
    }
    setExpandedSections(newExpanded)
  }

  const copyLesson = async () => {
    const fullContent = lesson.fragments
      .map((f) => `## ${f.section_name}\n\n${f.content}`)
      .join('\n\n---\n\n')
    
    await navigator.clipboard.writeText(fullContent)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 overflow-y-auto bg-black/50 backdrop-blur-sm"
    >
      <div className="min-h-screen px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 50, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 50, scale: 0.95 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className="max-w-4xl mx-auto"
        >
          <Card className="shadow-2xl">
            {/* Header */}
            <CardHeader className="relative">
              <div className="flex items-start justify-between">
                <div className="flex-1 pr-8">
                  <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-2">
                    <BookOpen className="w-4 h-4" />
                    <span>Lesson from research paper</span>
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                    {lesson.paper_title}
                  </h2>
                  <div className="flex flex-wrap items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      <span>{formatReadTime(lesson.total_read_time)}</span>
                    </div>
                    <a
                      href={lesson.paper_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 hover:text-primary-500 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>View Paper</span>
                    </a>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={copyLesson}
                    className="text-gray-500"
                  >
                    {copied ? (
                      <Check className="w-4 h-4 text-green-500" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onClose}
                    className="text-gray-500"
                  >
                    <X className="w-5 h-5" />
                  </Button>
                </div>
              </div>
            </CardHeader>

            {/* Table of Contents */}
            <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-800">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">
                Table of Contents
              </h3>
              <div className="flex flex-wrap gap-2">
                {lesson.fragments.map((fragment, index) => (
                  <button
                    key={index}
                    onClick={() => {
                      setExpandedSections(new Set([index]))
                      document.getElementById(`section-${index}`)?.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start',
                      })
                    }}
                    className="text-sm px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-primary-100 dark:hover:bg-primary-900 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
                  >
                    {fragment.section_name}
                  </button>
                ))}
              </div>
            </div>

            {/* Content */}
            <CardContent className="space-y-4">
              {lesson.fragments.map((fragment, index) => (
                <LessonSection
                  key={index}
                  fragment={fragment}
                  index={index}
                  isExpanded={expandedSections.has(index)}
                  onToggle={() => toggleSection(index)}
                />
              ))}
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  )
}

interface LessonSectionProps {
  fragment: LessonFragment
  index: number
  isExpanded: boolean
  onToggle: () => void
}

function LessonSection({ fragment, index, isExpanded, onToggle }: LessonSectionProps) {
  return (
    <motion.div
      id={`section-${index}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className="border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden"
    >
      <button
        onClick={onToggle}
        className="w-full px-4 py-3 flex items-center justify-between bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="w-6 h-6 rounded-full bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400 text-sm font-medium flex items-center justify-center">
            {index + 1}
          </span>
          <h3 className="font-semibold text-gray-900 dark:text-white capitalize">
            {fragment.section_name}
          </h3>
        </div>
        <div className="flex items-center gap-2 text-gray-500">
          <span className="text-sm">{formatReadTime(fragment.estimated_read_time)}</span>
          {isExpanded ? (
            <ChevronUp className="w-5 h-5" />
          ) : (
            <ChevronDown className="w-5 h-5" />
          )}
        </div>
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="p-4 prose-custom">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {fragment.content}
              </ReactMarkdown>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
```

#### frontend/src/components/Loading.tsx
```tsx
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function LoadingSpinner({ size = 'md', className }: LoadingSpinnerProps) {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
  }

  return (
    <div className={cn('relative', sizes[size], className)}>
      <motion.div
        className="absolute inset-0 rounded-full border-2 border-primary-200 dark:border-primary-800"
      />
      <motion.div
        className="absolute inset-0 rounded-full border-2 border-transparent border-t-primary-500"
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
      />
    </div>
  )
}

interface LoadingOverlayProps {
  message?: string
}

export function LoadingOverlay({ message = 'Loading...' }: LoadingOverlayProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm"
    >
      <div className="text-center">
        <LoadingSpinner size="lg" className="mx-auto mb-4" />
        <p className="text-gray-600 dark:text-gray-400">{message}</p>
      </div>
    </motion.div>
  )
}
```

#### frontend/src/components/ProblemDisplay.tsx
```tsx
import { motion } from 'framer-motion'
import { 
  Code, 
  ExternalLink, 
  X,
  Tag,
  RefreshCw
} from 'lucide-react'
import { Card, CardContent, CardHeader } from './Card'
import Button from './Button'
import { Problem } from '@/lib/api'
import { cn } from '@/lib/utils'

interface ProblemDisplayProps {
  problem: Problem
  onClose: () => void
  onNewProblem: () => void
  isLoading: boolean
}

export default function ProblemDisplay({ 
  problem, 
  onClose, 
  onNewProblem,
  isLoading 
}: ProblemDisplayProps) {
  const difficultyColors: Record<string, string> = {
    Easy: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
    Medium: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300',
    Hard: 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300',
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 overflow-y-auto bg-black/50 backdrop-blur-sm"
    >
      <div className="min-h-screen px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 50, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 50, scale: 0.95 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className="max-w-4xl mx-auto"
        >
          <Card className="shadow-2xl">
            {/* Header */}
            <CardHeader className="relative">
              <div className="flex items-start justify-between">
                <div className="flex-1 pr-8">
                  <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-2">
                    <Code className="w-4 h-4" />
                    <span>Coding Challenge</span>
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">
                    {problem.title}
                  </h2>
                  <div className="flex flex-wrap items-center gap-3">
                    <span className={cn(
                      'px-3 py-1 rounded-full text-sm font-medium',
                      difficultyColors[problem.difficulty] || difficultyColors['Medium']
                    )}>
                      {problem.difficulty}
                    </span>
                    <a
                      href={problem.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 text-sm text-primary-500 hover:text-primary-600 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Solve on LeetCode</span>
                    </a>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={onNewProblem}
                    isLoading={isLoading}
                  >
                    <RefreshCw className="w-4 h-4 mr-1" />
                    New Problem
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onClose}
                    className="text-gray-500"
                  >
                    <X className="w-5 h-5" />
                  </Button>
                </div>
              </div>
            </CardHeader>

            {/* Topics */}
            {problem.topics && problem.topics.length > 0 && (
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-800">
                <div className="flex items-center gap-2 mb-2">
                  <Tag className="w-4 h-4 text-gray-500" />
                  <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Related Topics
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {problem.topics.map((topic) => (
                    <span
                      key={topic}
                      className="text-sm px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"
                    >
                      {topic}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Problem Statement */}
            <CardContent>
              <div className="prose-custom">
                <h3 className="text-lg font-semibold mb-4">Problem Statement</h3>
                <div className="bg-gray-50 dark:bg-gray-800/50 rounded-xl p-4 font-mono text-sm whitespace-pre-wrap">
                  {problem.statement}
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  )
}
```

### 3.3 Lib & Utils

#### frontend/src/lib/api.ts
```typescript
const API_BASE = (import.meta as any).env?.VITE_API_URL 
  ? `${(import.meta as any).env.VITE_API_URL}/api/v1`
  : '/api/v1'

export interface LessonRequest {
  query: string
  difficulty?: 'beginner' | 'intermediate' | 'advanced'
  include_examples?: boolean
  include_math?: boolean
  max_sections?: number
}

export interface LessonFragment {
  section_name: string
  content: string
  order: number
  estimated_read_time: number
}

export interface Lesson {
  paper_id: string
  paper_title: string
  paper_url: string
  query: string
  fragments: LessonFragment[]
  total_read_time: number
  generation_time_seconds: number
}

export interface LessonResponse {
  success: boolean
  lesson?: Lesson
  error?: string
  processing_time_ms: number
}

export interface Problem {
  title: string
  slug: string
  difficulty: 'Easy' | 'Medium' | 'Hard'
  statement: string
  url: string
  topics: string[]
}

export interface ProblemResponse {
  success: boolean
  problem?: Problem
  error?: string
  processing_time_ms: number
}

export interface StreamChunk {
  type: 'metadata' | 'section' | 'done' | 'error'
  data: Record<string, unknown>
}

export async function generateLesson(request: LessonRequest): Promise<LessonResponse> {
  const response = await fetch(`${API_BASE}/teach`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  return response.json()
}

export async function getRandomProblem(
  difficulties: string[] = ['Medium', 'Hard']
): Promise<ProblemResponse> {
  const response = await fetch(`${API_BASE}/leetcode/random`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ difficulties, exclude_premium: true }),
  })
  return response.json()
}

export async function searchPapers(query: string, topK: number = 5) {
  const response = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  })
  return response.json()
}
```

#### frontend/src/lib/utils.ts
```typescript
import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatReadTime(minutes: number): string {
  if (minutes < 1) return 'Less than 1 min read'
  if (minutes === 1) return '1 min read'
  return `${minutes} min read`
}
```

### 3.4 Styles

### 3.5 Frontend Config Files

#### frontend/package.json
```json
{
  "name": "llm-teaching-assistant-ui",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint src --ext ts,tsx"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.21.0",
    "framer-motion": "^10.18.0",
    "lucide-react": "^0.303.0",
    "react-markdown": "^9.0.1",
    "remark-gfm": "^4.0.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.45",
    "@types/react-dom": "^18.2.18",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "tailwindcss": "^3.4.0",
    "typescript": "^5.3.3",
    "vite": "^5.0.10"
  }
}
```

#### frontend/vite.config.ts
```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

#### frontend/tailwind.config.js
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
          950: '#082f49',
        },
        accent: {
          50: '#fdf4ff',
          100: '#fae8ff',
          200: '#f5d0fe',
          300: '#f0abfc',
          400: '#e879f9',
          500: '#d946ef',
          600: '#c026d3',
          700: '#a21caf',
          800: '#86198f',
          900: '#701a75',
          950: '#4a044e',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'gradient': 'gradient 8s linear infinite',
        'float': 'float 6s ease-in-out infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 2s linear infinite',
      },
      keyframes: {
        gradient: {
          '0%, 100%': {
            'background-size': '200% 200%',
            'background-position': 'left center'
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'right center'
          },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'hero-pattern': 'url("data:image/svg+xml,%3Csvg width=\'60\' height=\'60\' viewBox=\'0 0 60 60\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cg fill=\'none\' fill-rule=\'evenodd\'%3E%3Cg fill=\'%239C92AC\' fill-opacity=\'0.05\'%3E%3Cpath d=\'M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z\'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")',
      },
    },
  },
  plugins: [],
}
```

#### frontend/tsconfig.json
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

## 4. Root Config Files

#### README.md
```markdown
# 🎓 LLM Teaching Assistant

<div align="center">

![Hero](https://img.shields.io/badge/AI-Powered_Learning-blue?style=for-the-badge&logo=openai&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Transform dense research papers into lessons you'll actually understand.**

[Live Demo](https://your-app.railway.app) · [Report Bug](https://github.com/ganeshasrinivasd/llm-teaching-assistant/issues) · [Request Feature](https://github.com/ganeshasrinivasd/llm-teaching-assistant/issues)

</div>

---

## 🤔 The Problem

Ever tried reading a machine learning research paper?

```
"We propose a novel attention mechanism utilizing scaled dot-product 
attention with multi-head projections across the latent space..."
```

**Translation:** 😵‍💫

Research papers are written by experts, for experts. But what if you're:
- A student trying to learn ML
- A developer wanting to understand new techniques
- A curious mind exploring AI

You're stuck with two bad options:
1. **Read the paper** → Get lost in jargon, math, and assumptions
2. **Ask ChatGPT** → Get a generic summary that misses the nuances

---

## 💡 The Solution

What if an AI could:
1. **Find** the most relevant paper for what you want to learn
2. **Read** the entire paper (not just summarize the abstract)
3. **Teach** you section by section, like a patient tutor

That's exactly what this does.

```
You: "Teach me about attention mechanisms"

AI: *finds the Transformer paper*
    *reads all 15 pages*
    *generates a personalized lesson*
    
    "Let's start with WHY attention matters. Imagine you're 
    translating 'The cat sat on the mat' to French. When 
    translating 'cat', which English words should you focus on?
    
    This is attention - letting the model CHOOSE what to look at..."
```

---

## 🧠 Why Not Just Use ChatGPT?

Great question. Here's the difference:

### ChatGPT Approach
```
You: "Explain transformers"
ChatGPT: *searches its training data*
         *gives you a general explanation*
         *might be outdated or incomplete*
```

### Our Approach
```
You: "Explain transformers"
Us:  1. Search 231 curated ML papers using semantic similarity
     2. Find the ACTUAL paper that best matches your query
     3. Download the PDF
     4. Parse it into structured sections using GROBID
     5. Generate lessons from the REAL content
     6. Cite the source so you can verify
```

### Technical Comparison

| Aspect | ChatGPT | LLM Teaching Assistant |
|--------|---------|------------------------|
| **Source** | Training data (static) | Live papers (dynamic) |
| **Accuracy** | May hallucinate | Grounded in real papers |
| **Depth** | Surface-level | Section-by-section deep dive |
| **Citation** | None | Links to original paper |
| **Recency** | Knowledge cutoff | Always current papers |
| **Customization** | Generic | Adapts to your level |

### Non-Technical Explanation

Think of it like this:

**ChatGPT** = A friend who read a lot of books and tells you what they remember

**Us** = A librarian who:
- Finds the exact book you need
- Reads it cover to cover
- Explains each chapter in simple terms
- Shows you where to find the original

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                    │
│                         (React + TypeScript)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │    Hero     │  │   Lesson    │  │   Problem   │  │   Theme     │     │
│  │   Input     │  │   Display   │  │   Display   │  │   Toggle    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │ HTTP/REST
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              BACKEND                                     │
│                           (FastAPI + Python)                             │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      API Layer (/api/v1)                          │   │
│  │  ┌─────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │ /health │  │   /teach    │  │ /teach/stream│  │ /leetcode  │  │   │
│  │  └─────────┘  └─────────────┘  └──────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Service Layer                                 │   │
│  │  ┌──────────────┐  ┌───────────────┐  ┌────────────────────┐     │   │
│  │  │   Teaching   │  │    Paper      │  │      Lesson        │     │   │
│  │  │   Service    │──│   Service     │──│      Service       │     │   │
│  │  │ (orchestrate)│  │ (fetch+parse) │  │ (generate lessons) │     │   │
│  │  └──────────────┘  └───────────────┘  └────────────────────┘     │   │
│  │         │                  │                     │                │   │
│  │  ┌──────────────┐  ┌───────────────┐  ┌────────────────────┐     │   │
│  │  │   LeetCode   │  │   Embedding   │  │      Cache         │     │   │
│  │  │   Service    │  │   Service     │  │      Service       │     │   │
│  │  └──────────────┘  └───────────────┘  └────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌───────────────┐          ┌─────────────────┐          ┌─────────────────┐
│   FAISS       │          │     GROBID      │          │    OpenAI       │
│   Vector DB   │          │  (PDF Parser)   │          │     API         │
│               │          │                 │          │                 │
│ 231 papers    │          │ Extracts        │          │ • Embeddings    │
│ indexed by    │          │ sections from   │          │ • GPT-4o-mini   │
│ semantic      │          │ academic PDFs   │          │   for lessons   │
│ similarity    │          │                 │          │                 │
└───────────────┘          └─────────────────┘          └─────────────────┘
        ▲                            ▲
        │                            │
┌───────────────┐          ┌─────────────────┐
│    arXiv      │          │    LeetCode     │
│    Papers     │          │      API        │
│               │          │                 │
│ Source of     │          │ Coding problems │
│ ML research   │          │ for practice    │
└───────────────┘          └─────────────────┘
```

---

## 🔄 How It Works (Flow)

```
                                    ┌─────────────────┐
                                    │   User Query    │
                                    │ "Explain BERT"  │
                                    └────────┬────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │   1. EMBED THE QUERY     │
                              │   OpenAI text-embedding  │
                              │   → 1536-dim vector      │
                              └──────────────┬───────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │   2. SEMANTIC SEARCH     │
                              │   FAISS finds closest    │
                              │   paper from 231 indexed │
                              │   → arxiv.org/abs/xxx    │
                              └──────────────┬───────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │   3. FETCH & PARSE PDF   │
                              │   Download from arXiv    │
                              │   GROBID extracts:       │
                              │   • Introduction         │
                              │   • Methods              │
                              │   • Results              │
                              │   • 20+ sections         │
                              └──────────────┬───────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │   4. GENERATE LESSONS    │
                              │   For each section:      │
                              │   GPT-4o-mini creates    │
                              │   beginner-friendly      │
                              │   explanation            │
                              └──────────────┬───────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │   5. RETURN LESSON       │
                              │   Complete course with:  │
                              │   • Table of contents    │
                              │   • Section-by-section   │
                              │   • Source citation      │
                              │   • Estimated read time  │
                              └──────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose | Why This? |
|------------|---------|-----------|
| **FastAPI** | REST API | Async, fast, auto-docs, Python type hints |
| **FAISS** | Vector search | Facebook's library, blazing fast similarity search |
| **GROBID** | PDF parsing | Best-in-class academic PDF parser, extracts structure |
| **OpenAI** | Embeddings + LLM | text-embedding-3-small + GPT-4o-mini |
| **Pydantic** | Data validation | Type safety, automatic serialization |

### Frontend
| Technology | Purpose | Why This? |
|------------|---------|-----------|
| **React 18** | UI framework | Component-based, huge ecosystem |
| **TypeScript** | Type safety | Catch errors at compile time |
| **Tailwind CSS** | Styling | Utility-first, rapid development |
| **Framer Motion** | Animations | Smooth, declarative animations |
| **Vite** | Build tool | Lightning fast HMR |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| **Railway** | Hosting (backend + frontend) |
| **GROBID Cloud** | PDF parsing service |
| **GitHub** | Version control |

---

## 📁 Project Structure

```
llm-teaching-assistant/
│
├── backend/                          # Python FastAPI backend
│   ├── api/
│   │   ├── main.py                   # FastAPI app entry
│   │   └── routes/
│   │       ├── teach.py              # /teach endpoints
│   │       ├── leetcode.py           # /leetcode endpoints
│   │       └── health.py             # Health checks
│   │
│   ├── services/
│   │   ├── teaching_service.py       # Main orchestration
│   │   ├── paper_service.py          # Paper fetching + GROBID
│   │   ├── lesson_service.py         # GPT lesson generation
│   │   ├── embedding_service.py      # FAISS + OpenAI embeddings
│   │   ├── leetcode_service.py       # LeetCode integration
│   │   └── cache_service.py          # Caching layer
│   │
│   ├── models/                       # Pydantic data models
│   ├── core/                         # Config, logging, exceptions
│   └── requirements.txt
│
├── frontend/                         # React TypeScript frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Hero.tsx              # Main input section
│   │   │   ├── LessonDisplay.tsx     # Lesson modal
│   │   │   ├── ProblemDisplay.tsx    # LeetCode modal
│   │   │   └── Header.tsx            # Navigation
│   │   ├── lib/
│   │   │   └── api.ts                # API client
│   │   └── App.tsx                   # Main app
│   │
│   └── package.json
│
└── README.md                         # You are here
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenAI API key

### 1. Clone & Setup Backend

```bash
git clone https://github.com/ganeshasrinivasd/llm-teaching-assistant.git
cd llm-teaching-assistant/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Initialize the paper index
python scripts/setup_index.py

# Run the server
uvicorn api.main:app --reload
```

### 2. Setup Frontend

```bash
cd ../frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### 3. Open App

Visit **http://localhost:3000** 🎉

---

## 📖 API Reference

### Generate Lesson
```http
POST /api/v1/teach
Content-Type: application/json

{
  "query": "Explain attention mechanisms",
  "difficulty": "beginner",
  "max_sections": 5
}
```

### Get Coding Problem
```http
POST /api/v1/leetcode/random
Content-Type: application/json

{
  "difficulties": ["Medium", "Hard"]
}
```

### Health Check
```http
GET /health
```

Full API docs available at `/docs` when running locally.

---

## 🎯 Features

- [x] Semantic paper search
- [x] PDF parsing with GROBID
- [x] Section-by-section lessons
- [x] LeetCode integration
- [x] Dark/Light mode
- [x] Mobile responsive
- [ ] Streaming responses (coming soon)
- [ ] User accounts
- [ ] Save lesson history
- [ ] Multiple difficulty levels

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LLMSys-PaperList](https://github.com/AmberLJC/LLMSys-PaperList) for the curated paper collection
- [GROBID](https://github.com/kermitt2/grobid) for PDF parsing
- [OpenAI](https://openai.com) for embeddings and language models
- [LeetCode](https://leetcode.com) for coding problems

---

<div align="center">

**Built with ❤️ for learners everywhere**

[⬆ Back to top](#-llm-teaching-assistant)

</div>
```

## 6. Project Summary

### Current v2 Features
- ✅ Relevance thresholds (0.50/0.35/0.20)
- ✅ Query enhancement (intent detection)
- ✅ Dynamic paper fetching (Semantic Scholar)
- ✅ LeetCode removed
- ✅ FAISS vector search
- ✅ GROBID PDF parsing
- ✅ GPT-4o-mini lesson generation

### TODO for Next Phase
- 🔜 Migrate FAISS → Pinecone (persistent dynamic updates)
- 🔜 Multi-paper comparison lessons
- 🔜 Concept Map UI
- 🔜 User feedback system

---

*Context generated for Claude to understand the complete codebase*
