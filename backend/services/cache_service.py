"""
Cache Service

File-based and in-memory caching for improved performance.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, TypeVar, Generic
from functools import lru_cache
from pydantic import BaseModel

from core.config import get_settings
from core.logging import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


class CacheEntry(BaseModel, Generic[T]):
    """A cached entry with metadata."""
    
    key: str
    value: Any
    created_at: float
    ttl: int
    
    @property
    def is_expired(self) -> bool:
        return time.time() > (self.created_at + self.ttl)


class CacheService:
    """
    Hybrid cache service with in-memory and file-based storage.
    
    - Hot data stays in memory (LRU cache)
    - Cold data persists to disk
    - Automatic TTL expiration
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, default_ttl: int = 86400):
        settings = get_settings()
        self.cache_dir = cache_dir or settings.cache_dir
        self.default_ttl = default_ttl
        self.enabled = settings.cache_enabled
        
        # In-memory cache
        self._memory_cache: dict[str, CacheEntry] = {}
        self._max_memory_items = 100
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Cache service initialized: dir={self.cache_dir}, enabled={self.enabled}")
    
    def _make_key(self, namespace: str, identifier: str) -> str:
        """Create a cache key from namespace and identifier."""
        raw = f"{namespace}:{identifier}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{key}.json"
    
    def get(self, namespace: str, identifier: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            namespace: Cache namespace (e.g., "lessons", "papers")
            identifier: Unique identifier within namespace
            
        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None
        
        key = self._make_key(namespace, identifier)
        
        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired:
                logger.debug(f"Cache hit (memory): {namespace}:{identifier[:20]}")
                return entry.value
            else:
                del self._memory_cache[key]
        
        # Check file cache
        file_path = self._get_file_path(key)
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    entry = CacheEntry(**data)
                    
                    if not entry.is_expired:
                        # Promote to memory cache
                        self._memory_cache[key] = entry
                        self._evict_if_needed()
                        logger.debug(f"Cache hit (file): {namespace}:{identifier[:20]}")
                        return entry.value
                    else:
                        # Remove expired file
                        file_path.unlink()
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        logger.debug(f"Cache miss: {namespace}:{identifier[:20]}")
        return None
    
    def set(
        self,
        namespace: str,
        identifier: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set a value in cache.
        
        Args:
            namespace: Cache namespace
            identifier: Unique identifier
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (default: 24 hours)
            
        Returns:
            True if cached successfully
        """
        if not self.enabled:
            return False
        
        key = self._make_key(namespace, identifier)
        ttl = ttl or self.default_ttl
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=ttl
        )
        
        # Save to memory
        self._memory_cache[key] = entry
        self._evict_if_needed()
        
        # Save to file
        try:
            file_path = self._get_file_path(key)
            with open(file_path, "w") as f:
                json.dump(entry.model_dump(), f)
            logger.debug(f"Cached: {namespace}:{identifier[:20]}")
            return True
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
            return False
    
    def delete(self, namespace: str, identifier: str) -> bool:
        """Delete a cached value."""
        key = self._make_key(namespace, identifier)
        
        # Remove from memory
        if key in self._memory_cache:
            del self._memory_cache[key]
        
        # Remove from file
        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()
            return True
        
        return False
    
    def clear(self, namespace: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            namespace: If provided, only clear this namespace
            
        Returns:
            Number of entries cleared
        """
        count = 0
        
        if namespace is None:
            # Clear all
            count = len(self._memory_cache)
            self._memory_cache.clear()
            
            for file_path in self.cache_dir.glob("*.json"):
                file_path.unlink()
                count += 1
        
        logger.info(f"Cache cleared: {count} entries")
        return count
    
    def _evict_if_needed(self):
        """Evict oldest entries if memory cache is too large."""
        while len(self._memory_cache) > self._max_memory_items:
            # Remove oldest entry
            oldest_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k].created_at
            )
            del self._memory_cache[oldest_key]
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        file_count = len(list(self.cache_dir.glob("*.json")))
        
        return {
            "enabled": self.enabled,
            "memory_entries": len(self._memory_cache),
            "file_entries": file_count,
            "cache_dir": str(self.cache_dir),
            "default_ttl": self.default_ttl
        }


# Singleton instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get the global cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
