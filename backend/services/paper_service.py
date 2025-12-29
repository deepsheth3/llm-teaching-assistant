"""
Paper Service

Paper retrieval, PDF processing, and section extraction.
"""

import re
import requests
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import (
    PaperNotFoundError,
    GROBIDError,
    ArxivError,
    PDFProcessingError,
)
from models.paper import PaperMetadata, PaperSection, ParsedPaper, PaperSearchResult
from services.embedding_service import get_embedding_service
from services.cache_service import get_cache_service

logger = get_logger(__name__)


class PaperService:
    """
    Service for paper retrieval and processing.
    
    Features:
    - Semantic search via FAISS
    - PDF parsing via GROBID
    - Fallback to abstract-only mode
    - Caching for performance
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.cache = get_cache_service()

        self._existing_paper_ids = set()
        self._load_existing_papers()
        
        logger.info(f"Paper service initialized: grobid={self.settings.grobid_url}")

    def _load_existing_papers(self):
        """Load all existing paper IDs from the embedding service into a set for fast lookup."""
        try:
            existing_urls = self.embedding_service.urls
            for url in existing_urls:
                arxiv_id = self._extract_arxiv_id(url)
                if arxiv_id:
                    self._existing_paper_ids.add(arxiv_id)

                if "semantic_" in url:
                    semantic_id = url.split("semantic_")[-1].split("/")[0]
                    if semantic_id:
                        self._existing_paper_ids.add(semantic_id)

            logger.info(f"Loaded {len(self._existing_paper_ids)} existing paper IDs for tracking.")

        except Exception as e:
            logger.error(f"Error loading existing papers: {e}")
    
    def search(self, query: str, top_k: int = 1) -> list[PaperSearchResult]:
        """
        Search for papers matching a query.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of search results
        """
        results = self.embedding_service.search(query, k=top_k)
        
        search_results = []
        for idx, distance, url in results:
            # Convert L2 distance to similarity score (0-1)
            # Lower distance = higher similarity
            similarity = 1 / (1 + distance)
            
            arxiv_id = self._extract_arxiv_id(url)
            
            paper = PaperMetadata(
                arxiv_id=arxiv_id,
                title=f"Paper {arxiv_id}",  # Will be updated when fetched
                url=url
            )
            
            search_results.append(PaperSearchResult(
                paper=paper,
                similarity_score=similarity,
                index_position=idx
            ))
        
        return search_results
    
    def get_paper(self, url: str, use_grobid: bool = True) -> ParsedPaper:
        """
        Get a parsed paper from URL.
        
        Args:
            url: arXiv URL
            use_grobid: Whether to use GROBID for full parsing
            
        Returns:
            Parsed paper with sections
        """
        arxiv_id = self._extract_arxiv_id(url)
        
        # Check cache
        cache_key = f"{arxiv_id}:{'grobid' if use_grobid else 'abstract'}"
        cached = self.cache.get("papers", cache_key)
        if cached:
            logger.info(f"Paper cache hit: {arxiv_id}")
            return ParsedPaper(**cached)
        
        # Fetch metadata
        metadata = self._fetch_arxiv_metadata(arxiv_id)
        
        # Try GROBID if enabled
        sections = []
        parsing_method = "abstract"
        
        if use_grobid and self.settings.use_grobid:
            try:
                sections = self._parse_with_grobid(url)
                parsing_method = "grobid"
                logger.info(f"GROBID parsed {len(sections)} sections from {arxiv_id}")
            except Exception as e:
                logger.warning(f"GROBID failed, falling back to abstract: {e}")
        
        # Fallback: use abstract as single section
        if not sections and metadata.abstract:
            sections = [PaperSection(
                name="abstract",
                content=metadata.abstract,
                order=0
            )]
            parsing_method = "abstract"
        
        paper = ParsedPaper(
            metadata=metadata,
            sections=sections,
            parsing_method=parsing_method
        )
        
        # Cache result
        self.cache.set("papers", cache_key, paper.model_dump(mode='json'))
        
        return paper
    
    def _extract_arxiv_id(self, url: str) -> str:
        """Extract arXiv ID from URL."""
        match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)', url)
        if match:
            return match.group(1)
        
        # Try to extract from path
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        if path_parts:
            return path_parts[-1].replace(".pdf", "")
        
        return url
    
    def _fetch_arxiv_metadata(self, arxiv_id: str) -> PaperMetadata:
        """Fetch paper metadata from arXiv API."""
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
        
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ArxivError(f"Failed to fetch metadata: {e}")
        
        # Parse XML
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(response.content)
        entry = root.find('atom:entry', ns)
        
        if entry is None:
            raise PaperNotFoundError(arxiv_id)
        
        title = entry.find('atom:title', ns)
        summary = entry.find('atom:summary', ns)
        authors = entry.findall('atom:author/atom:name', ns)
        
        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=title.text.strip() if title is not None else f"Paper {arxiv_id}",
            url=f"https://arxiv.org/abs/{arxiv_id}",
            abstract=summary.text.strip() if summary is not None else None,
            authors=[a.text for a in authors if a.text]
        )
    
    def _parse_with_grobid(self, url: str) -> list[PaperSection]:
        """Parse paper using GROBID service."""
        # Convert to PDF URL
        if '/abs/' in url:
            arxiv_id = self._extract_arxiv_id(url)
            pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
        else:
            pdf_url = url
        
        # Download PDF
        logger.debug(f"Downloading PDF: {pdf_url}")
        try:
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
            pdf_bytes = response.content
        except requests.RequestException as e:
            raise PDFProcessingError(f"Failed to download PDF: {e}", url)
        
        # Send to GROBID
        logger.debug(f"Sending to GROBID: {self.settings.grobid_url}")
        try:
            files = {'input': ('paper.pdf', BytesIO(pdf_bytes), 'application/pdf')}
            response = requests.post(
                f'{self.settings.grobid_url}/api/processFulltextDocument',
                files=files,
                timeout=self.settings.grobid_timeout
            )
            response.raise_for_status()
            tei_xml = response.text
        except requests.RequestException as e:
            raise GROBIDError(f"GROBID processing failed: {e}")
        
        # Parse TEI XML
        return self._parse_tei_xml(tei_xml)
    
    def _parse_tei_xml(self, tei_xml: str) -> list[PaperSection]:
        """Parse TEI XML to extract sections."""
        TEI_NS = 'http://www.tei-c.org/ns/1.0'
        ET.register_namespace('tei', TEI_NS)
        
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as e:
            raise PDFProcessingError(f"Failed to parse TEI XML: {e}")
        
        sections = []
        order = 0
        
        for div in root.findall(f'.//{{{TEI_NS}}}div'):
            # Skip body-level divs
            if div.attrib.get('type') == 'body':
                continue
            
            # Get section name
            section_name = (
                div.attrib.get('type')
                or div.attrib.get('subtype')
                or next(
                    (h.text for h in div.findall(f'./{{{TEI_NS}}}head') if h.text),
                    None
                )
            )
            
            if not section_name:
                continue
            
            # Extract text content
            text_parts = []
            for element in div.iter():
                if element.text and element.tag != f'{{{TEI_NS}}}head':
                    text_parts.append(element.text.strip())
                if element.tail:
                    text_parts.append(element.tail.strip())
            
            content = ' '.join(part for part in text_parts if part)
            
            if content:
                sections.append(PaperSection(
                    name=section_name.lower(),
                    content=content,
                    order=order
                ))
                order += 1
        
        return sections
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "grobid_url": self.settings.grobid_url,
            "grobid_enabled": self.settings.use_grobid,
            "cache_stats": self.cache.get_stats()
        }
    
    def paper_exists(self, arxiv_id: str = None, semantic_scholar_id: str = None) -> bool:
        """Check if paper exists - checks both IDs in same set"""
        if arxiv_id and arxiv_id in self._existing_paper_ids:
            return True
        
        if semantic_scholar_id and semantic_scholar_id in self._existing_paper_ids:
            return True
        
        return False

    def add_paper_to_tracking(self, arxiv_id: str = None, semantic_scholar_id: str = None):
        """Add either or both IDs to the set"""
        if arxiv_id:
            self._existing_paper_ids.add(arxiv_id)
        if semantic_scholar_id:
            self._existing_paper_ids.add(semantic_scholar_id)


# Singleton instance
_paper_service: Optional[PaperService] = None


def get_paper_service() -> PaperService:
    """Get the global paper service instance."""
    global _paper_service
    if _paper_service is None:
        _paper_service = PaperService()
    return _paper_service
