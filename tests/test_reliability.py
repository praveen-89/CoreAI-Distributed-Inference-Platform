"""
CoreAI Distributed Inference Platform - Phase 9 Reliability Tests
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from shared.reliability import reap_stale_tasks


@pytest.fixture
def mock_redis_client():
    client = AsyncMock()
    # Mock the underlying redis aioredis client
    client.redis = AsyncMock()
    return client


class TestReliabilityReaper:
    @pytest.mark.asyncio
    async def test_reap_stale_tasks_worker_alive(self, mock_redis_client):
        redis_mock = mock_redis_client.redis
        redis_mock.keys.return_value = ["processing:gpt-2:worker-1"]
        # Worker is alive
        redis_mock.exists.return_value = True

        await reap_stale_tasks(mock_redis_client)

        redis_mock.keys.assert_called_once_with("processing:*:*")
        redis_mock.exists.assert_called_once_with("worker:worker-1")
        # Ensure we didn't try to pop or delete
        redis_mock.rpoplpush.assert_not_called()
        redis_mock.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_reap_stale_tasks_worker_dead(self, mock_redis_client):
        redis_mock = mock_redis_client.redis
        redis_mock.keys.return_value = ["processing:gpt-2:worker-1"]
        # Worker is dead
        redis_mock.exists.return_value = False

        # Simulate 2 tasks in the processing queue, then it becomes empty
        redis_mock.rpoplpush.side_effect = [
            b'{"task_id": "req-1"}',
            b'{"task_id": "req-2"}',
            None,
        ]

        await reap_stale_tasks(mock_redis_client)

        redis_mock.keys.assert_called_once_with("processing:*:*")
        redis_mock.exists.assert_called_once_with("worker:worker-1")
        
        # Verify tasks were moved back to the main queue
        assert redis_mock.rpoplpush.call_count == 3
        redis_mock.rpoplpush.assert_any_call("processing:gpt-2:worker-1", "queue:gpt-2")
        
        # Verify the processing queue was cleaned up
        redis_mock.delete.assert_called_once_with("processing:gpt-2:worker-1")
