"""
Teaching Routes

Endpoints for lesson generation and paper search.
"""

import json
from typing import Optional
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.logging import get_logger
from models.lesson import LessonRequest, LessonResponse, LessonDifficulty
from models.paper import PaperSearchRequest
from services.teaching_service import get_teaching_service

router = APIRouter()
logger = get_logger(__name__)


class TeachRequest(BaseModel):
    """Request to learn about a topic."""
    query: str = Field(..., min_length=3, max_length=500, description="What do you want to learn?")
    difficulty: LessonDifficulty = Field(LessonDifficulty.BEGINNER, description="Lesson difficulty")
    include_examples: bool = Field(True, description="Include examples")
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


class SearchRequest(BaseModel):
    """Request to search for papers."""
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(5, ge=1, le=20)


@router.post("/teach", response_model=LessonResponse)
async def teach(request: TeachRequest):
    """
    Generate a lesson about a topic.
    
    This endpoint:
    1. Searches for the most relevant research paper
    2. Parses the paper into sections
    3. Generates beginner-friendly explanations for each section
    
    Returns:
        Complete lesson with all sections
    """
    logger.info(f"Teach request: {request.query[:50]}...")
    
    teaching_service = get_teaching_service()
    
    lesson_request = LessonRequest(
        query=request.query,
        difficulty=request.difficulty,
        include_examples=request.include_examples,
        include_math=request.include_math,
        max_sections=request.max_sections
    )
    
    return teaching_service.teach(lesson_request)


@router.post("/teach/stream")
async def teach_streaming(request: TeachRequest):
    """
    Generate a lesson with streaming responses.
    
    Returns Server-Sent Events (SSE) as sections are generated.
    
    Event types:
    - `metadata`: Paper information
    - `section`: Generated lesson section
    - `done`: Generation complete
    - `error`: Error occurred
    """
    logger.info(f"Streaming teach request: {request.query[:50]}...")
    
    teaching_service = get_teaching_service()
    
    lesson_request = LessonRequest(
        query=request.query,
        difficulty=request.difficulty,
        include_examples=request.include_examples,
        include_math=request.include_math,
        max_sections=request.max_sections
    )
    
    async def event_generator():
        async for chunk in teaching_service.teach_streaming(lesson_request):
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/search")
async def search_papers(request: SearchRequest):
    """
    Search for relevant papers.
    
    Returns a list of papers matching the query, ranked by relevance.
    """
    logger.info(f"Search request: {request.query[:50]}...")
    
    teaching_service = get_teaching_service()
    results = teaching_service.search_papers(request.query, top_k=request.top_k)
    
    return {
        "query": request.query,
        "results": [r.model_dump() for r in results],
        "count": len(results)
    }


@router.get("/paper")
async def get_paper(
    url: str = Query(..., description="arXiv paper URL")
):
    """
    Get detailed information about a specific paper.
    
    Returns parsed paper with sections.
    """
    logger.info(f"Paper request: {url}")
    
    teaching_service = get_teaching_service()
    paper = teaching_service.get_paper_details(url)
    
    return paper.model_dump()
