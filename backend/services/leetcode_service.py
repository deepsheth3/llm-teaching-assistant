"""
LeetCode Service

Fetches coding problems from LeetCode for interview practice.
"""

import random
import requests
import bs4
from typing import Optional

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import LeetCodeError
from models.problem import (
    LeetCodeProblem,
    ProblemDifficulty,
    ProblemRequest,
    ProblemCatalogEntry,
)
from services.cache_service import get_cache_service

logger = get_logger(__name__)


# LeetCode API constants
LEETCODE_CATALOG_URL = "https://leetcode.com/api/problems/algorithms/"
LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"
LEETCODE_GRAPHQL_QUERY = """
query questionData($titleSlug: String!) {
    question(titleSlug: $titleSlug) {
        content
        hints
        topicTags { name }
    }
}
"""


class LeetCodeService:
    """
    Service for fetching LeetCode problems.
    
    Features:
    - Catalog fetching with caching
    - Random problem selection by difficulty
    - Problem statement retrieval
    - Topic filtering
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = get_cache_service()
        self._catalog: Optional[list[ProblemCatalogEntry]] = None
        
        logger.info("LeetCode service initialized")
    
    def get_random_problem(self, request: Optional[ProblemRequest] = None) -> LeetCodeProblem:
        """
        Get a random LeetCode problem.
        
        Args:
            request: Problem selection criteria
            
        Returns:
            Random problem matching criteria
        """
        request = request or ProblemRequest()
        
        # Get catalog
        catalog = self._get_catalog()
        
        # Filter problems
        filtered = self._filter_problems(catalog, request)
        
        if not filtered:
            raise LeetCodeError("No problems match the specified criteria")
        
        # Select random problem
        selected = random.choice(filtered)
        
        # Fetch full problem
        return self._fetch_problem(selected)
    
    def get_problem_by_slug(self, slug: str) -> LeetCodeProblem:
        """
        Get a specific problem by slug.
        
        Args:
            slug: Problem URL slug (e.g., "two-sum")
            
        Returns:
            The problem
        """
        # Check cache
        cached = self.cache.get("leetcode_problems", slug)
        if cached:
            logger.debug(f"LeetCode cache hit: {slug}")
            return LeetCodeProblem(**cached)
        
        # Find in catalog
        catalog = self._get_catalog()
        entry = next((p for p in catalog if p.slug == slug), None)
        
        if not entry:
            raise LeetCodeError(f"Problem not found: {slug}")
        
        return self._fetch_problem(entry)
    
    def _get_catalog(self) -> list[ProblemCatalogEntry]:
        """Get the problem catalog, with caching."""
        if self._catalog is not None:
            return self._catalog
        
        # Check cache
        cached = self.cache.get("leetcode", "catalog")
        if cached:
            self._catalog = [ProblemCatalogEntry(**p) for p in cached]
            logger.debug(f"Loaded catalog from cache: {len(self._catalog)} problems")
            return self._catalog
        
        # Fetch from API
        self._catalog = self._fetch_catalog()
        
        # Cache for 24 hours
        self.cache.set(
            "leetcode",
            "catalog",
            [p.model_dump() for p in self._catalog],
            ttl=86400
        )
        
        return self._catalog
    
    def _fetch_catalog(self) -> list[ProblemCatalogEntry]:
        """Fetch the problem catalog from LeetCode API."""
        logger.info("Fetching LeetCode catalog...")
        
        try:
            response = requests.get(LEETCODE_CATALOG_URL, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise LeetCodeError(f"Failed to fetch catalog: {e}")
        
        difficulty_map = {1: ProblemDifficulty.EASY, 2: ProblemDifficulty.MEDIUM, 3: ProblemDifficulty.HARD}
        
        catalog = []
        for problem in data.get("stat_status_pairs", []):
            stat = problem.get("stat", {})
            diff = problem.get("difficulty", {})
            
            catalog.append(ProblemCatalogEntry(
                slug=stat.get("question__title_slug", ""),
                title=stat.get("question__title", ""),
                difficulty=difficulty_map.get(diff.get("level", 2), ProblemDifficulty.MEDIUM),
                paid_only=problem.get("paid_only", False),
                acceptance_rate=stat.get("total_acs", 0) / max(stat.get("total_submitted", 1), 1) * 100
            ))
        
        logger.info(f"Fetched {len(catalog)} problems")
        return catalog
    
    def _filter_problems(
        self,
        catalog: list[ProblemCatalogEntry],
        request: ProblemRequest
    ) -> list[ProblemCatalogEntry]:
        """Filter problems based on request criteria."""
        filtered = []
        
        for problem in catalog:
            # Filter by premium
            if request.exclude_premium and problem.paid_only:
                continue
            
            # Filter by difficulty
            if problem.difficulty not in request.difficulties:
                continue
            
            filtered.append(problem)
        
        return filtered
    
    def _fetch_problem(self, entry: ProblemCatalogEntry) -> LeetCodeProblem:
        """Fetch full problem details."""
        # Check cache
        cached = self.cache.get("leetcode_problems", entry.slug)
        if cached:
            return LeetCodeProblem(**cached)
        
        # Fetch via GraphQL
        try:
            response = requests.post(
                LEETCODE_GRAPHQL_URL,
                json={
                    "query": LEETCODE_GRAPHQL_QUERY,
                    "variables": {"titleSlug": entry.slug}
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise LeetCodeError(f"Failed to fetch problem {entry.slug}: {e}")
        
        question = data.get("data", {}).get("question", {})
        
        if not question:
            raise LeetCodeError(f"Problem not found: {entry.slug}")
        
        # Parse HTML content
        html_content = question.get("content", "")
        statement = self._parse_html(html_content)
        
        # Extract topics
        topics = [tag.get("name", "") for tag in question.get("topicTags", [])]
        
        # Extract hints
        hints = question.get("hints", [])
        
        problem = LeetCodeProblem(
            title=entry.title,
            slug=entry.slug,
            difficulty=entry.difficulty,
            statement=statement,
            topics=topics,
            hints=hints,
            acceptance_rate=entry.acceptance_rate
        )
        
        # Cache problem
        self.cache.set("leetcode_problems", entry.slug, problem.model_dump())
        
        return problem
    
    def _parse_html(self, html: str) -> str:
        """Parse HTML content to clean text."""
        if not html:
            return ""
        
        soup = bs4.BeautifulSoup(html, "html.parser")
        return soup.get_text("\n").strip()
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        catalog_size = len(self._catalog) if self._catalog else 0
        
        return {
            "catalog_loaded": self._catalog is not None,
            "catalog_size": catalog_size,
            "cache_stats": self.cache.get_stats()
        }


# Singleton instance
_leetcode_service: Optional[LeetCodeService] = None


def get_leetcode_service() -> LeetCodeService:
    """Get the global LeetCode service instance."""
    global _leetcode_service
    if _leetcode_service is None:
        _leetcode_service = LeetCodeService()
    return _leetcode_service
