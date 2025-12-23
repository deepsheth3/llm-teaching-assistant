"""
Teaching Service

Main orchestration service that combines paper retrieval, 
lesson generation, and LeetCode functionality.
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
from models.problem import ProblemRequest, ProblemResponse, LeetCodeProblem
from services.paper_service import get_paper_service
from services.lesson_service import get_lesson_service
from services.leetcode_service import get_leetcode_service
from services.cache_service import get_cache_service

logger = get_logger(__name__)


class TeachingService:
    """
    Main teaching service that orchestrates all functionality.
    
    This is the primary entry point for:
    - Teaching about research topics
    - Generating lessons from papers
    - Providing coding practice
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.paper_service = get_paper_service()
        self.lesson_service = get_lesson_service()
        self.leetcode_service = get_leetcode_service()
        self.cache = get_cache_service()
        
        logger.info("Teaching service initialized")
    
    def teach(self, request: LessonRequest) -> LessonResponse:
        """
        Main teaching endpoint - finds relevant paper and generates lesson.
        
        Args:
            request: Lesson request with query and preferences
            
        Returns:
            Complete lesson response
        """
        start_time = time.time()
        
        try:
            # Search for relevant paper
            logger.info(f"Teaching request: {request.query[:50]}...")
            search_results = self.paper_service.search(request.query, top_k=1)
            
            if not search_results:
                raise PaperNotFoundError(request.query)
            
            best_result = search_results[0]
            logger.info(f"Found paper: {best_result.paper.arxiv_id} (score: {best_result.similarity_score:.2f})")
            
            # Get full paper
            paper = self.paper_service.get_paper(
                str(best_result.paper.url),
                use_grobid=self.settings.use_grobid
            )
            
            # Generate lesson
            lesson = self.lesson_service.generate_lesson(paper, request)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return LessonResponse(
                success=True,
                lesson=lesson,
                cached=False,
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
    
    async def teach_streaming(
        self,
        request: LessonRequest
    ) -> AsyncGenerator[StreamingLessonChunk, None]:
        """
        Streaming teaching endpoint - yields chunks as they're generated.
        
        Args:
            request: Lesson request
            
        Yields:
            Streaming lesson chunks
        """
        try:
            # Search for paper
            search_results = self.paper_service.search(request.query, top_k=1)
            
            if not search_results:
                yield StreamingLessonChunk(
                    type="error",
                    data={"message": f"No papers found for: {request.query}"}
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
    
    def get_coding_problem(self, request: Optional[ProblemRequest] = None) -> ProblemResponse:
        """
        Get a coding problem for practice.
        
        Args:
            request: Problem request criteria
            
        Returns:
            Problem response
        """
        start_time = time.time()
        
        try:
            problem = self.leetcode_service.get_random_problem(request)
            processing_time = int((time.time() - start_time) * 1000)
            
            return ProblemResponse(
                success=True,
                problem=problem,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to get coding problem: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            
            return ProblemResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time
            )
    
    def search_papers(self, query: str, top_k: int = 5) -> list[PaperSearchResult]:
        """
        Search for papers without generating lessons.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of search results
        """
        return self.paper_service.search(query, top_k=top_k)
    
    def get_paper_details(self, url: str) -> ParsedPaper:
        """
        Get full paper details.
        
        Args:
            url: Paper URL
            
        Returns:
            Parsed paper
        """
        return self.paper_service.get_paper(url)
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "paper_service": self.paper_service.get_stats(),
            "leetcode_service": self.leetcode_service.get_stats(),
            "cache": self.cache.get_stats()
        }


# Singleton instance
_teaching_service: Optional[TeachingService] = None


def get_teaching_service() -> TeachingService:
    """Get the global teaching service instance."""
    global _teaching_service
    if _teaching_service is None:
        _teaching_service = TeachingService()
    return _teaching_service
