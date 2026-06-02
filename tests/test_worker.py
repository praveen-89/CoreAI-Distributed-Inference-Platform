"""
CoreAI Distributed Inference Platform - Phase 4 Worker Tests
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from worker.main import WorkerDaemon
from worker.model_runner import ModelRunner


@pytest.fixture
def mock_settings():
    return MagicMock(
        worker_id="worker-test",
        model_id="gpt2",
        device="cpu",
        redis_host="localhost",
        redis_port=6379,
        redis_password="",
        redis_db=0,
    )


@pytest.fixture
def mock_redis_client():
    client = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.dequeue_task = AsyncMock(return_value=None)
    client.set_task_status = AsyncMock()
    client.set_result = AsyncMock()
    client.publish_result_ready = AsyncMock()
    return client


@pytest.fixture
def mock_model_runner():
    runner = MagicMock(spec=ModelRunner)
    runner.load = MagicMock()
    runner.generate = MagicMock(
        return_value={
            "choice": {
                "index": 0,
                "message": {"role": "assistant", "content": "mocked response"},
                "finish_reason": "stop",
            },
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
    )
    return runner


class TestWorkerDaemon:
    @pytest.mark.asyncio
    @patch("worker.main.RedisClient")
    @patch("worker.main.ModelRunner")
    @patch("worker.main.get_settings")
    async def test_process_task_success(
        self,
        mock_get_settings,
        mock_model_runner_cls,
        mock_redis_client_cls,
        mock_settings,
        mock_redis_client,
        mock_model_runner,
    ):
        mock_get_settings.return_value = mock_settings
        mock_redis_client_cls.return_value = mock_redis_client
        mock_model_runner_cls.return_value = mock_model_runner

        daemon = WorkerDaemon()
        daemon.runner = mock_model_runner

        task = {
            "request_id": "req-123",
            "model": "gpt2",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 50,
            "temperature": 0.7,
        }

        await daemon.process_task(task)

        # Verify state updates
        mock_redis_client.set_task_status.assert_any_call("req-123", "PROCESSING")
        
        # Verify model generation called
        mock_model_runner.generate.assert_called_once_with(
            [{"role": "user", "content": "hello"}], max_tokens=50, temperature=0.7
        )

        # Verify result saved
        assert mock_redis_client.set_result.call_count == 1
        call_args = mock_redis_client.set_result.call_args[0]
        assert call_args[0] == "req-123"
        
        result_json = json.loads(call_args[1])
        assert result_json["id"] == "req-123"
        assert result_json["choices"][0]["message"]["content"] == "mocked response"

        # Verify final status update and pubsub
        mock_redis_client.set_task_status.assert_any_call("req-123", "COMPLETED", result=call_args[1])
        mock_redis_client.publish_result_ready.assert_called_once_with("req-123")

    @pytest.mark.asyncio
    @patch("worker.main.RedisClient")
    @patch("worker.main.ModelRunner")
    @patch("worker.main.get_settings")
    async def test_process_task_failure(
        self,
        mock_get_settings,
        mock_model_runner_cls,
        mock_redis_client_cls,
        mock_settings,
        mock_redis_client,
        mock_model_runner,
    ):
        mock_get_settings.return_value = mock_settings
        mock_redis_client_cls.return_value = mock_redis_client
        mock_model_runner_cls.return_value = mock_model_runner

        daemon = WorkerDaemon()
        
        # Model runner throws exception
        mock_model_runner.generate.side_effect = RuntimeError("GPU out of memory")
        daemon.runner = mock_model_runner

        task = {
            "request_id": "req-999",
            "messages": [],
        }

        await daemon.process_task(task)

        # Verify failure state recorded
        mock_redis_client.set_task_status.assert_any_call(
            "req-999", "FAILED", extra={"error": "GPU out of memory"}
        )
