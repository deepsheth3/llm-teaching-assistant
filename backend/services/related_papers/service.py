"""
Related papers orchestration service.

Implements semantic read-through caching:
- Try Pinecone first
- Fallback to Semantic Scholar
- Store new papers when fetched
"""

from typing import List, Dict

from services.related_papers.retriever import retrieve_similar_papers
from services.related_papers.paper_sources import fetch_related_papers
from services.related_papers.storage import upsert_papers_to_pinecone


def get_related_papers(query: str, limit: int = 5) -> List[Dict]:
    """
    Get related papers using semantic cache.

    Args:
        query: User query
        limit: Number of papers

    Returns:
        List of paper metadata
    """
    # 1️⃣ Try semantic cache
    cached_papers = retrieve_similar_papers(query)

    if cached_papers:
        return cached_papers[:limit]

    # 2️⃣ Fallback to external API
    papers = fetch_related_papers(query, limit=limit)

    # 3️⃣ Store for future reuse
    if papers:
        upsert_papers_to_pinecone(papers)

    return papers
