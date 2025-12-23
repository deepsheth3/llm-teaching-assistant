"""
Paper Data Models

Pydantic models for paper-related data structures.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class PaperMetadata(BaseModel):
    """Metadata for a research paper."""
    
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    url: HttpUrl = Field(..., description="Paper URL")
    abstract: Optional[str] = Field(None, description="Paper abstract")
    authors: list[str] = Field(default_factory=list, description="Paper authors")
    categories: list[str] = Field(default_factory=list, description="arXiv categories")
    published_date: Optional[datetime] = Field(None, description="Publication date")
    
    class Config:
        json_schema_extra = {
            "example": {
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/abs/1706.03762",
                "abstract": "The dominant sequence transduction models...",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "categories": ["cs.CL", "cs.LG"],
                "published_date": "2017-06-12T00:00:00Z"
            }
        }


class PaperSection(BaseModel):
    """A section extracted from a paper."""
    
    name: str = Field(..., description="Section name/title")
    content: str = Field(..., description="Section text content")
    order: int = Field(..., description="Section order in paper")
    word_count: int = Field(0, description="Word count")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.word_count == 0:
            self.word_count = len(self.content.split())


class ParsedPaper(BaseModel):
    """A fully parsed paper with sections."""
    
    metadata: PaperMetadata
    sections: list[PaperSection] = Field(default_factory=list)
    raw_text: Optional[str] = Field(None, description="Full raw text")
    parsing_method: str = Field("grobid", description="Method used to parse (grobid/abstract)")
    parsed_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def section_names(self) -> list[str]:
        return [s.name for s in self.sections]
    
    @property
    def total_words(self) -> int:
        return sum(s.word_count for s in self.sections)


class PaperSearchResult(BaseModel):
    """Result from a paper search."""
    
    paper: PaperMetadata
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity score")
    index_position: int = Field(..., description="Position in FAISS index")
    
    class Config:
        json_schema_extra = {
            "example": {
                "paper": {
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "url": "https://arxiv.org/abs/1706.03762"
                },
                "similarity_score": 0.92,
                "index_position": 42
            }
        }


class PaperSearchRequest(BaseModel):
    """Request to search for papers."""
    
    query: str = Field(..., min_length=3, max_length=500, description="Search query")
    top_k: int = Field(1, ge=1, le=10, description="Number of results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "attention mechanisms in transformers",
                "top_k": 3
            }
        }
