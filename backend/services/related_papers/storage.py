"""
Pinecone-backed storage for related papers.
"""

from typing import List, Dict

from services.embedding_service import get_embedding_service
from services.cache_service import get_cache_service
from services.related_papers.pinecone_client import get_pinecone_index
from services.related_papers.embedding_utils import paper_to_embedding_text


def upsert_papers_to_pinecone(papers: List[Dict]) -> None:
    """
    Embed papers and upsert them into Pinecone.

    Args:
        papers: List of paper metadata dicts
    """
    if not papers:
        return

    embedding_service = get_embedding_service()
    cache = get_cache_service()
    index = get_pinecone_index()

    vectors = []

    for paper in papers:
        # Build semantic text
        text = paper_to_embedding_text(
            paper["title"],
            paper["abstract"],
        )

        # Generate embedding
        embedding = embedding_service.create_embedding(text)

        # Store full metadata in cache (cheap + fast)
        cache.set(
            namespace="papers",
            identifier=paper["paper_id"],
            value=paper,
        )

        # Prepare Pinecone vector
        vectors.append({
            "id": paper["paper_id"],
            "values": embedding.tolist(),
            "metadata": {
                "title": paper["title"],
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "source": paper.get("source"),
            },
        })

    # Upsert all vectors at once
    index.upsert(vectors=vectors)
