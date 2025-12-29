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

            arxiv_id = self.scholar_service.get_arxiv_id(paper)
            if self.paper_service.paper_exists(arxiv_id=arxiv_id,semantic_scholar_id=paper_id):
                logger.info(f"Paper already exists in index: {arxiv_id} / {paper_id}")
                continue
            
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
