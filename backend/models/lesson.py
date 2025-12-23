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
