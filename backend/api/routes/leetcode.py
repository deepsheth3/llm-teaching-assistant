"""
LeetCode Routes

Endpoints for coding practice.
"""

from typing import Optional, List
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from core.logging import get_logger
from models.problem import ProblemRequest, ProblemResponse, ProblemDifficulty
from services.teaching_service import get_teaching_service

router = APIRouter()
logger = get_logger(__name__)


class CodingProblemRequest(BaseModel):
    """Request for a coding problem."""
    difficulties: List[ProblemDifficulty] = Field(
        default=[ProblemDifficulty.MEDIUM, ProblemDifficulty.HARD],
        description="Allowed difficulty levels"
    )
    exclude_premium: bool = Field(True, description="Exclude premium problems")
    
    class Config:
        json_schema_extra = {
            "example": {
                "difficulties": ["Medium", "Hard"],
                "exclude_premium": True
            }
        }


@router.post("/leetcode/random", response_model=ProblemResponse)
async def get_random_problem(request: Optional[CodingProblemRequest] = None):
    """
    Get a random LeetCode problem for practice.
    
    By default, returns Medium or Hard problems (non-premium).
    
    Returns:
        Random problem with statement and metadata
    """
    request = request or CodingProblemRequest()
    logger.info(f"LeetCode request: difficulties={request.difficulties}")
    
    teaching_service = get_teaching_service()
    
    problem_request = ProblemRequest(
        difficulties=request.difficulties,
        exclude_premium=request.exclude_premium
    )
    
    return teaching_service.get_coding_problem(problem_request)


@router.get("/leetcode/problem/{slug}", response_model=ProblemResponse)
async def get_problem_by_slug(slug: str):
    """
    Get a specific LeetCode problem by slug.
    
    Args:
        slug: Problem URL slug (e.g., "two-sum")
        
    Returns:
        The requested problem
    """
    logger.info(f"LeetCode slug request: {slug}")
    
    teaching_service = get_teaching_service()
    
    try:
        problem = teaching_service.leetcode_service.get_problem_by_slug(slug)
        return ProblemResponse(success=True, problem=problem)
    except Exception as e:
        return ProblemResponse(success=False, error=str(e))
