"""
Semantic retrieval logic for related papers.
"""

from typing import List, Dict

from services.embedding_service import get_embedding_service
from services.cache_service import get_cache_service
from services.related_papers.pinecone_client import get_pinecone_index

# Cosine similarity threshold
SIMILARITY_THRESHOLD = 0.80
TOP_K = 5


def retrieve_similar_papers(query: str) -> List[Dict]:
    """
    Retrieve semantically similar papers for a query.

    Args:
        query: User query or paper description

    Returns:
        List of paper metadata dicts (from cache)
    """
    embedding_service = get_embedding_service()
    cache = get_cache_service()
    index = get_pinecone_index()

    # Embed query
    query_embedding = embedding_service.create_embedding(query)

    # Query Pinecone
    response = index.query(
        vector=query_embedding.tolist(),
        top_k=TOP_K,
        include_metadata=False,
    )

    papers: List[Dict] = []

    for match in response.get("matches", []):
        score = match["score"]  # cosine similarity
        paper_id = match["id"]

        # Threshold check
        if score < SIMILARITY_THRESHOLD:
            continue

        paper = cache.get(namespace="papers", identifier=paper_id)
        if paper:
            papers.append(paper)

    return papers
