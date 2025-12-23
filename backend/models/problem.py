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
