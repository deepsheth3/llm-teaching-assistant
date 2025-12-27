import requests
from typing import List, Dict
# from core.logging import get_logger

# logger = get_logger(__name__)

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,authors,year,venue,url,paperId"


def fetch_related_papers(query: str, limit: int = 20) -> List[Dict]:
    """
    Fetch related research papers from Semantic Scholar.

    Args:
        query: Natural language query or paper title
        limit: Number of papers to fetch

    Returns:
        List of paper metadata dictionaries
    """
    try:
        response = requests.get(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            params={
                "query": query,
                "limit": limit,
                "fields": FIELDS,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        papers = []
        for paper in data.get("data", []):
            # Skip papers without abstracts (not useful for embeddings)
            if not paper.get("abstract"):
                continue

            papers.append({
                "paper_id": paper.get("paperId"),
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "authors": [a.get("name") for a in paper.get("authors", [])],
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "url": paper.get("url"),
                "source": "semantic_scholar",
            })
            if len(papers) == 10:
                break

        print(f"Fetched {len(papers)} related papers for query='{query[:40]}'")
        return papers

    except Exception as e:
        print(f"Semantic Scholar fetch failed: {e}")
        return []
    

if __name__ == '__main__':
    papers = fetch_related_papers("transformer attention mechanisms", limit=20)
    for paper in papers:
        print(paper['title'])
        print(paper['abstract'])
        print(paper['url'])
        print('\n')