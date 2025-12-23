"""
Configuration Management

Centralized configuration with environment variable support,
validation, and sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # App Info
    app_name: str = "LLM Teaching Assistant"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, description="Debug mode")
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # OpenAI Settings
    openai_api_key: str = Field(..., description="OpenAI API key")
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"
    lesson_model: str = "gpt-4o-mini"  # Cheaper model for lesson generation
    
    # GROBID Settings
    grobid_url: str = "https://kermitt2-grobid.hf.space"
    grobid_timeout: int = 120  # seconds
    use_grobid: bool = True  # Can disable for abstract-only mode
    
    # File Paths
    data_dir: Path = Path("data")
    faiss_index_path: Path = Path("data/faiss/papers.index")
    urls_json_path: Path = Path("data/faiss/urls.json")
    cache_dir: Path = Path("data/cache")
    
    # Cache Settings
    cache_enabled: bool = True
    cache_ttl: int = 86400  # 24 hours in seconds
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 30  # requests per minute
    rate_limit_window: int = 60  # seconds
    
    # LeetCode Settings
    leetcode_difficulties: list[str] = ["Medium", "Hard"]
    leetcode_allow_premium: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"
    
    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or v == "your-api-key-here":
            raise ValueError("Valid OpenAI API key is required")
        return v
    
    @field_validator("data_dir", "cache_dir", mode="after")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v
    
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
