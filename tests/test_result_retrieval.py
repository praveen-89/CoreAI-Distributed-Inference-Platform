"""
CoreAI Distributed Inference Platform - Phase 5 Result Retrieval Tests
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.services.result_service import TimeoutException, wait_for_inference_result


@pytest.fixture
def mock_redis_client():
    client = AsyncMock()
    client.get_result = AsyncMock(return_value=None)
    client.delete_result = AsyncMock()
    client.get_task_status = AsyncMock(return_value=None)
    
    mock_pubsub = AsyncMock()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.unsubscribe = AsyncMock()
    mock_pubsub.close = AsyncMock()
    
    client.subscribe_result_channel = MagicMock(return_value=mock_pubsub)
    return client, mock_pubsub


class TestResultService:
    @pytest.mark.asyncio
    async def test_wait_for_inference_result_cached_immediately(self, mock_redis_client):
        client, pubsub = mock_redis_client
        # Simulate result already in cache
        client.get_result.return_value = json.dumps({"choices": [{"message": {"content": "hello"}}]})
        
        result = await wait_for_inference_result(client, "req-1", timeout=1.0)
        
        assert result["choices"][0]["message"]["content"] == "hello"
        client.delete_result.assert_called_once_with("req-1")
        pubsub.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_wait_for_inference_result_pubsub_success(self, mock_redis_client):
        client, pubsub = mock_redis_client
        
        # Simulate pubsub yielding a DONE message
        async def mock_listen():
            yield {"type": "message", "data": b"DONE"}
        
        pubsub.listen = mock_listen
        
        # The first two get_result calls (before/after subscribe) return None.
        # The third call (after pubsub DONE) returns the result.
        client.get_result.side_effect = [
            None,  # before subscribe
            None,  # after subscribe
            json.dumps({"choices": [{"message": {"content": "from pubsub"}}]}),  # after message
        ]
        
        result = await wait_for_inference_result(client, "req-2", timeout=1.0)
        
        assert result["choices"][0]["message"]["content"] == "from pubsub"
        pubsub.subscribe.assert_called_once_with("channel:result:req-2")
        pubsub.unsubscribe.assert_called_once_with("channel:result:req-2")

    @pytest.mark.asyncio
    async def test_wait_for_inference_result_timeout(self, mock_redis_client):
        client, pubsub = mock_redis_client
        
        # Simulate pubsub hanging
        async def mock_listen():
            await asyncio.sleep(2.0)
            yield {"type": "message", "data": "DONE"}
        
        pubsub.listen = mock_listen
        
        with pytest.raises(TimeoutException):
            await wait_for_inference_result(client, "req-3", timeout=0.1)
            
        pubsub.unsubscribe.assert_called_once()
        pubsub.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_inference_result_worker_failure(self, mock_redis_client):
        client, pubsub = mock_redis_client
        
        async def mock_listen():
            yield {"type": "message", "data": "DONE"}
        pubsub.listen = mock_listen
        
        # pubsub says DONE, but get_result returns None because worker crashed
        client.get_result.side_effect = [None, None, None]
        client.get_task_status.return_value = {"status": "FAILED", "error": "OOM"}
        
        with pytest.raises(RuntimeError, match="Inference failed: OOM"):
            await wait_for_inference_result(client, "req-4", timeout=1.0)
