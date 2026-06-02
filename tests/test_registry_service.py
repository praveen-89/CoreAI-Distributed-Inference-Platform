"""
CoreAI Distributed Inference Platform - Phase 6 Registry Tests
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.services.registry_service import get_available_models


@pytest.fixture
def mock_redis_client():
    client = AsyncMock()
    # Assume 2 workers are active, one serving gpt-2 and another serving custom-model
    client.get_active_models = AsyncMock(return_value={"gpt-2", "custom-model"})
    return client


class TestRegistryService:
    @pytest.mark.asyncio
    async def test_get_available_models_success(self, mock_redis_client):
        models = await get_available_models(mock_redis_client)
        
        assert len(models) == 2
        # Should be sorted
        assert models[0].id == "custom-model"
        assert models[1].id == "gpt-2"
        assert models[0].object == "model"
        
    @pytest.mark.asyncio
    async def test_get_available_models_fallback_on_error(self, mock_redis_client):
        mock_redis_client.get_active_models.side_effect = Exception("Redis connection lost")
        
        models = await get_available_models(mock_redis_client)
        
        # Service should suppress the exception and return empty list
        assert models == []
