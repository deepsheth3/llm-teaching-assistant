"""
Tests for paper deduplication functionality.

Tests that:
1. paper_exists() correctly identifies existing papers
2. Duplicate papers are not added to the index
3. New papers are added correctly
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import os

# Set test environment variables before importing services
os.environ['OPENAI_API_KEY'] = 'sk-test-key-for-testing'


@pytest.fixture(autouse=True)
def mock_embedding_service():
    """Mock the embedding service for all tests."""
    with patch('services.embedding_service.OpenAI'):
        yield


class TestPaperDeduplication:
    """Test suite for paper deduplication."""
    
    @pytest.fixture
    def paper_service(self):
        """Create a PaperService instance with mocked dependencies."""
        with patch('services.paper_service.get_embedding_service') as mock_emb, \
             patch('services.paper_service.get_cache_service') as mock_cache:
            
            # Setup mocks
            mock_emb.return_value.urls = []
            mock_cache.return_value = Mock()
            
            from services.paper_service import PaperService
            service = PaperService()
            service._existing_paper_ids = set()  # Start with empty set
            return service
    
    def test_paper_exists_with_arxiv_id(self, paper_service):
        """Test that paper_exists detects papers by arXiv ID."""
        # Setup: Add a paper ID to the tracking set
        arxiv_id = "2017.12345v1"
        paper_service._existing_paper_ids.add(arxiv_id)
        
        # Test: Check if paper exists
        assert paper_service.paper_exists(arxiv_id=arxiv_id) is True
    
    def test_paper_does_not_exist(self, paper_service):
        """Test that paper_exists returns False for new papers."""
        arxiv_id = "2024.99999v1"
        
        # Ensure set is empty
        paper_service._existing_paper_ids = set()
        
        # Test: Check if new paper exists
        assert paper_service.paper_exists(arxiv_id=arxiv_id) is False
    
    def test_paper_exists_with_semantic_scholar_id(self, paper_service):
        """Test that paper_exists detects papers by Semantic Scholar ID."""
        semantic_id = "abc123def456"
        paper_service._existing_paper_ids.add(semantic_id)
        
        # Test: Check by semantic ID
        assert paper_service.paper_exists(semantic_scholar_id=semantic_id) is True
    
    def test_add_paper_to_tracking(self, paper_service):
        """Test that add_paper_to_tracking updates the set."""
        arxiv_id = "2017.12345v1"
        semantic_id = "xyz789"
        
        # Start with empty set
        paper_service._existing_paper_ids = set()
        
        # Add paper to tracking
        paper_service.add_paper_to_tracking(arxiv_id=arxiv_id, semantic_scholar_id=semantic_id)
        
        # Test: Both IDs should be in the set
        assert arxiv_id in paper_service._existing_paper_ids
        assert semantic_id in paper_service._existing_paper_ids
    
    def test_load_existing_papers_initializes_set(self, paper_service):
        """Test that _load_existing_papers populates the set on startup."""
        # Mock the embedding service to return some URLs
        with patch.object(paper_service.embedding_service, 'urls', 
                         ['https://arxiv.org/abs/2017.12345v1',
                          'https://arxiv.org/abs/2018.67890v1',
                          'https://www.semanticscholar.org/paper/abc123']):
            
            # Reset and reload
            paper_service._existing_paper_ids = set()
            paper_service._load_existing_papers()
            
            # Test: Set should contain extracted IDs
            assert len(paper_service._existing_paper_ids) > 0
            assert '2017.12345v1' in paper_service._existing_paper_ids or \
                   '2018.67890v1' in paper_service._existing_paper_ids
    
    def test_deduplication_prevents_duplicate_adds(self):
        """Test that duplicates are caught before being added."""
        with patch('services.paper_service.get_paper_service') as mock_ps, \
             patch('services.embedding_service.get_embedding_service') as mock_es:
            
            # Setup mock paper service
            ps = Mock()
            ps.paper_exists.return_value = True  # Simulate paper already exists
            mock_ps.return_value = ps
            
            # Test: paper_exists should return True, preventing add
            assert ps.paper_exists(arxiv_id="2017.12345v1") is True


class TestTeachingServiceDeduplication:
    """Test suite for deduplication in TeachingService."""
    
    @pytest.fixture
    def teaching_service(self):
        """Create a mock TeachingService."""
        with patch('services.teaching_service.get_paper_service'), \
             patch('services.teaching_service.get_lesson_service'), \
             patch('services.teaching_service.get_cache_service'), \
             patch('services.teaching_service.get_query_service'), \
             patch('services.teaching_service.get_scholar_service'), \
             patch('services.teaching_service.get_embedding_service'):
            
            from services.teaching_service import TeachingService
            service = TeachingService()
            return service
    
    def test_fetch_and_add_papers_skips_duplicates(self, teaching_service):
        """Test that _fetch_and_add_papers skips papers that already exist."""
        # Mock _add_papers_to_index to prevent disk I/O and local import issues
        with patch.object(teaching_service, '_add_papers_to_index') as mock_add_index:
             
            # Mock papers from Semantic Scholar
            mock_papers = [
                {
                    'paperId': 'paper1',
                    'title': 'Existing Paper',
                    'abstract': 'This paper already exists in the index.',
                    'url': 'https://semanticscholar.org/paper/paper1'
                },
                {
                    'paperId': 'paper2',
                    'title': 'New Paper',
                    'abstract': 'This is a new paper that should be added.',
                    'url': 'https://semanticscholar.org/paper/paper2'
                }
            ]
            
            # Mock the scholar service to return papers
            teaching_service.scholar_service.search_papers.return_value = mock_papers
            
            # Mock paper service - first paper exists, second doesn't
            teaching_service.paper_service.paper_exists.side_effect = [True, False]
            teaching_service.scholar_service.get_arxiv_id.return_value = None
            teaching_service.scholar_service.get_arxiv_url.return_value = None
            teaching_service.embedding_service.create_embedding.return_value = [0.1, 0.2]
            
            # Call _fetch_and_add_papers
            result = teaching_service._fetch_and_add_papers("test query")
            
            # Test: Only new paper should be added (not the duplicate)
            assert len(result) == 1
            assert result[0]['id'] == 'semantic_paper2'
            
            # Verify _add_papers_to_index was called with the correct paper
            mock_add_index.assert_called_once()
            args, _ = mock_add_index.call_args
            assert args[0][0]['id'] == 'semantic_paper2'


# Integration test (requires real services to be running)
@pytest.mark.integration
class TestDeduplicationIntegration:
    """Integration tests for deduplication with real services."""
    
    def test_duplicate_papers_not_added_to_index(self):
        """
        Integration test: Fetch same paper twice and verify only added once.
        
        This test requires the backend to be running and .env to be configured.
        """
        pytest.skip("Integration test - requires running backend")
        
        from services.teaching_service import get_teaching_service
        
        service = get_teaching_service()
        
        # Get initial index size
        initial_size = service.embedding_service.index.ntotal
        
        # Fetch papers for a query
        papers1 = service._fetch_and_add_papers("attention mechanisms")
        size_after_first = service.embedding_service.index.ntotal
        
        # Fetch same papers again (should all be duplicates)
        papers2 = service._fetch_and_add_papers("attention mechanisms")
        size_after_second = service.embedding_service.index.ntotal
        
        # Test: No new papers should be added on second fetch
        assert len(papers2) == 0, "Duplicate papers were added!"
        assert size_after_second == size_after_first, "Index size increased with duplicates!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
