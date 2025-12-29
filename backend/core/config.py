"""
Configuration Management v2

Centralized configuration with environment variable support,
validation, and sensible defaults.

NEW in v2:
- Relevance thresholds
- Dynamic paper fetching settings
- Pinecone (optional) settings
- Removed LeetCode settings
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # ==========================================================================
    # App Info
    # ==========================================================================
    app_name: str = "LLM Teaching Assistant"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, description="Debug mode")
    
    # ==========================================================================
    # API Settings
    # ==========================================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # ==========================================================================
    # OpenAI Settings
    # ==========================================================================
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key (required for production, optional for tests)")
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"
    lesson_model: str = "gpt-4o-mini"
    
    # ==========================================================================
    # Vector Database Settings (NEW in v2)
    # ==========================================================================
    # Pinecone (optional - for production with dynamic updates)
    use_pinecone: bool = Field(default=False, description="Use Pinecone instead of FAISS")
    pinecone_api_key: Optional[str] = Field(default=None, description="Pinecone API key")
    pinecone_index_name: str = "llm-teaching-assistant"
    pinecone_environment: str = "us-east-1"
    
    # ==========================================================================
    # Relevance Thresholds (NEW in v2)
    # ==========================================================================
    high_relevance_threshold: float = Field(
        default=0.50, 
        description="Score above this = high relevance, use directly"
    )
    medium_relevance_threshold: float = Field(
        default=0.35, 
        description="Score above this = medium relevance, use but try to improve"
    )
    low_relevance_threshold: float = Field(
        default=0.20,
        description="Score below this = irrelevant"
    )
    
    # ==========================================================================
    # Dynamic Paper Fetching (NEW in v2)
    # ==========================================================================
    dynamic_fetch_enabled: bool = Field(
        default=True, 
        description="Enable fetching new papers when no good match"
    )
    max_papers_per_fetch: int = Field(
        default=10, 
        description="Max papers to fetch per query"
    )
    max_daily_fetches: int = Field(
        default=100, 
        description="Max Semantic Scholar API calls per day (cost control)"
    )
    semantic_scholar_api_key: Optional[str] = Field(
        default=None, 
        description="Optional API key for higher rate limits"
    )
    
    # ==========================================================================
    # GROBID Settings
    # ==========================================================================
    grobid_url: str = "https://kermitt2-grobid.hf.space"
    grobid_timeout: int = 120
    use_grobid: bool = True
    
    # ==========================================================================
    # File Paths
    # ==========================================================================
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    faiss_index_path: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "faiss" / "papers.index")
    urls_json_path: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "faiss" / "urls.json")
    cache_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "cache")
    
    # ==========================================================================
    # Cache Settings
    # ==========================================================================
    cache_enabled: bool = True
    cache_ttl: int = 86400  # 24 hours
    
    # ==========================================================================
    # Rate Limiting
    # ==========================================================================
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 30
    rate_limit_window: int = 60
    
    # ==========================================================================
    # Logging
    # ==========================================================================
    log_level: str = "INFO"
    log_format: str = "text"
    
    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        # Allow empty API key during tests
        if v and v == "your-api-key-here":
            raise ValueError("Valid OpenAI API key is required (or leave empty for tests)")
        return v
    
    @field_validator("pinecone_api_key")
    @classmethod
    def validate_pinecone_key(cls, v: Optional[str]) -> Optional[str]:
        if v and v != "your-pinecone-key":
            return v
        return None
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings


# Convenience function
def get_config() -> Settings:
    """Alias for get_settings."""
    return get_settings()
