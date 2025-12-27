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
    ClarifyRequest,
    ClarifyResponse,
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
    "ClarifyRequest",
    "ClarifyResponse",
    # Problem models
    "ProblemDifficulty",
    "LeetCodeProblem",
    "ProblemRequest",
    "ProblemResponse",
    "ProblemCatalogEntry",
]
