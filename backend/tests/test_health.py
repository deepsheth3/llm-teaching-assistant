import os
import pytest
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport

# Set test environment variables BEFORE importing app
os.environ['OPENAI_API_KEY'] = 'sk-test-key-for-testing'

# Mock the OpenAI client before importing app
with patch('services.embedding_service.OpenAI'):
    from api.main import app

@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
