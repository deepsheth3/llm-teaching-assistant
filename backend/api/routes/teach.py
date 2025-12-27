"""
Teaching API Routes v2

Endpoints:
- POST /teach - Generate lesson (with query enhancement)
- POST /teach/stream - Stream lesson generation
- POST /clarify - NEW: Generate 3 clearer versions of user's query
- GET /stats - Service statistics
- POST /search - Search papers
"""

import json
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.logging import get_logger
from core.exceptions import PaperNotFoundError
from models.lesson import (
    LessonRequest, 
    LessonResponse, 
    LessonDifficulty,
    ClarifyRequest,
    ClarifyResponse
)
from services.teaching_service import get_teaching_service
from services.query_service import get_query_service

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


# ============ NEW: Clarify Endpoint ============

@router.post("/clarify", response_model=ClarifyResponse)
async def clarify_query(request: ClarifyRequest):
    """
    Generate 3 clearer versions of user's query.
    
    User picks one, then calls /teach with selected prompt.
    
    Example:
        Input: "GPT, WHAT IS IT"
        Output: [
            "What is GPT and how does it work?",
            "Explain the GPT architecture and its key components",
            "What is Generative Pre-trained Transformer and why is it important?"
        ]
    """
    logger.info(f"POST /clarify: {request.query}")
    
    try:
        query_service = get_query_service()
        prompts = query_service.clarify_query(request.query)
        
        return ClarifyResponse(
            original=request.query,
            prompts=prompts
        )
        
    except Exception as e:
        logger.error(f"Clarification failed: {e}")
        # Fallback: return original query 3 times
        return ClarifyResponse(
            original=request.query,
            prompts=[request.query, request.query, request.query]
        )


# ============ Existing Endpoints ============

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
