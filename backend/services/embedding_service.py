"""
Embedding Service

Vector embedding generation and FAISS index management.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
import faiss
from openai import OpenAI

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import EmbeddingError, IndexNotFoundError

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings and managing FAISS index.
    
    Features:
    - Batch embedding generation
    - FAISS index creation and search
    - URL mapping management
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.embedding_model
        
        self._index: Optional[faiss.Index] = None
        self._urls: Optional[list[str]] = None
        
        logger.info(f"Embedding service initialized: model={self.model}")
    
    @property
    def index(self) -> faiss.Index:
        """Get the FAISS index, loading if necessary."""
        if self._index is None:
            self.load_index()
        return self._index
    
    @property
    def urls(self) -> list[str]:
        """Get the URL list, loading if necessary."""
        if self._urls is None:
            self.load_urls()
        return self._urls
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = np.array(response.data[0].embedding, dtype="float32")
            return embedding
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise EmbeddingError(f"Failed to create embedding: {e}")
    
    def create_embeddings_batch(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """
        Create embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            
        Returns:
            Array of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                raise EmbeddingError(f"Failed to create batch embeddings: {e}")
        
        return np.array(all_embeddings, dtype="float32")
    
    def build_index(self, embeddings: np.ndarray, urls: list[str]) -> None:
        """
        Build and save FAISS index.
        
        Args:
            embeddings: Array of embeddings
            urls: Corresponding URLs
        """
        if len(embeddings) != len(urls):
            raise ValueError("Embeddings and URLs must have same length")
        
        # Create FAISS index (L2 distance)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Save index
        self.settings.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.settings.faiss_index_path))
        
        # Save URLs
        with open(self.settings.urls_json_path, "w", encoding="utf-8") as f:
            json.dump(urls, f, ensure_ascii=False, indent=2)
        
        self._index = index
        self._urls = urls
        
        logger.info(f"Built FAISS index with {len(urls)} entries")
    
    def load_index(self) -> None:
        """Load FAISS index from disk."""
        index_path = self.settings.faiss_index_path
        
        if not index_path.exists():
            raise IndexNotFoundError(str(index_path))
        
        self._index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index: {self._index.ntotal} vectors")
    
    def load_urls(self) -> None:
        """Load URLs from disk."""
        urls_path = self.settings.urls_json_path
        
        if not urls_path.exists():
            raise IndexNotFoundError(str(urls_path))
        
        with open(urls_path, "r", encoding="utf-8") as f:
            self._urls = json.load(f)
        
        logger.info(f"Loaded {len(self._urls)} URLs")
    
    def search(self, query: str, k: int = 1) -> list[tuple[int, float, str]]:
        """
        Search for similar papers.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (index, distance, url) tuples
        """
        # Create query embedding
        query_embedding = self.create_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i in range(k):
            idx = int(indices[0][i])
            dist = float(distances[0][i])
            url = self.urls[idx] if idx < len(self.urls) else ""
            results.append((idx, dist, url))
        
        logger.debug(f"Search for '{query[:30]}...' returned {len(results)} results")
        return results
    
    def search_by_embedding(self, embedding: np.ndarray, k: int = 1) -> list[tuple[int, float, str]]:
        """
        Search using a pre-computed embedding.
        
        Args:
            embedding: Query embedding
            k: Number of results
            
        Returns:
            List of (index, distance, url) tuples
        """
        embedding = embedding.reshape(1, -1).astype("float32")
        distances, indices = self.index.search(embedding, k)
        
        results = []
        for i in range(k):
            idx = int(indices[0][i])
            dist = float(distances[0][i])
            url = self.urls[idx] if idx < len(self.urls) else ""
            results.append((idx, dist, url))
        
        return results
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "model": self.model,
            "index_loaded": self._index is not None,
            "index_size": self._index.ntotal if self._index else 0,
            "urls_loaded": self._urls is not None,
            "urls_count": len(self._urls) if self._urls else 0,
            "index_path": str(self.settings.faiss_index_path),
        }


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
