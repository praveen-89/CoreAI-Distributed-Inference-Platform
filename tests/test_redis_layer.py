"""
CoreAI Distributed Inference Platform – Phase 3 Redis Layer Tests

Tests cover:
  • RedisClient helper methods (mocked redis connection)
  • Queue service: envelope building, enqueue, backpressure
  • Async task state: create, retrieve
  • Gateway still works in degraded mode (no Redis)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.schemas import ChatCompletionRequest, ChatMessage, Role
from gateway.services.queue_service import (
    build_task_envelope,
    create_async_task,
    enqueue_inference_task,
    generate_request_id,
    get_task_result,
)
from shared.redis_client import RedisClient

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


def _make_payload(model: str = "gpt-2", content: str = "Hello") -> ChatCompletionRequest:
    """Build a minimal valid ChatCompletionRequest for testing."""
    return ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role=Role.USER, content=content)],
    )


@pytest.fixture()
def mock_redis_client() -> RedisClient:
    """Return a RedisClient whose underlying redis object is fully mocked."""
    client = RedisClient.__new__(RedisClient)
    mock_redis = AsyncMock()
    client._redis = mock_redis
    client._pool = MagicMock()
    return client


# ---------------------------------------------------------------------------
#  RedisClient Unit Tests
# ---------------------------------------------------------------------------


class TestRedisClient:
    """Tests for ``shared.redis_client.RedisClient``."""

    @pytest.mark.asyncio
    async def test_enqueue_task_calls_lpush(self, mock_redis_client: RedisClient) -> None:
        mock_redis_client._redis.lpush = AsyncMock(return_value=1)

        result = await mock_redis_client.enqueue_task("gpt-2", {"request_id": "r1"})

        mock_redis_client._redis.lpush.assert_called_once()
        assert result == 1

    @pytest.mark.asyncio
    async def test_get_queue_depth(self, mock_redis_client: RedisClient) -> None:
        mock_redis_client._redis.llen = AsyncMock(return_value=42)

        depth = await mock_redis_client.get_queue_depth("gpt-2")

        assert depth == 42
        mock_redis_client._redis.llen.assert_called_once_with("queue:gpt-2")

    @pytest.mark.asyncio
    async def test_set_and_get_task_status(self, mock_redis_client: RedisClient) -> None:
        mock_redis_client._redis.hset = AsyncMock()
        mock_redis_client._redis.hgetall = AsyncMock(
            return_value={"status": "PENDING", "model": "gpt-2"}
        )

        await mock_redis_client.set_task_status("r1", "PENDING", extra={"model": "gpt-2"})
        data = await mock_redis_client.get_task_status("r1")

        assert data is not None
        assert data["status"] == "PENDING"

    @pytest.mark.asyncio
    async def test_get_task_status_returns_none_for_missing(
        self, mock_redis_client: RedisClient
    ) -> None:
        mock_redis_client._redis.hgetall = AsyncMock(return_value={})
        result = await mock_redis_client.get_task_status("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_result(self, mock_redis_client: RedisClient) -> None:
        mock_redis_client._redis.set = AsyncMock()
        await mock_redis_client.set_result("r1", '{"output": "hello"}', ttl=60)
        mock_redis_client._redis.set.assert_called_once_with(
            "result:r1", '{"output": "hello"}', ex=60
        )

    @pytest.mark.asyncio
    async def test_ping_returns_false_on_failure(self, mock_redis_client: RedisClient) -> None:
        mock_redis_client._redis.ping = AsyncMock(side_effect=ConnectionError("fail"))
        result = await mock_redis_client.ping()
        assert result is False

    @pytest.mark.asyncio
    async def test_ping_returns_true(self, mock_redis_client: RedisClient) -> None:
        mock_redis_client._redis.ping = AsyncMock(return_value=True)
        result = await mock_redis_client.ping()
        assert result is True


# ---------------------------------------------------------------------------
#  Queue Service Unit Tests
# ---------------------------------------------------------------------------


class TestQueueService:
    """Tests for ``gateway.services.queue_service``."""

    def test_generate_request_id_format(self) -> None:
        rid = generate_request_id()
        assert rid.startswith("req-")
        assert len(rid) > 10  # uuid portion is non-trivial

    def test_build_task_envelope_structure(self) -> None:
        payload = _make_payload()
        envelope = build_task_envelope("req-123", payload)

        assert envelope["request_id"] == "req-123"
        assert envelope["model"] == "gpt-2"
        assert len(envelope["messages"]) == 1
        assert envelope["messages"][0]["role"] == "user"
        assert envelope["temperature"] == 1.0
        assert "created_at" in envelope

    @pytest.mark.asyncio
    async def test_enqueue_inference_task_success(
        self, mock_redis_client: RedisClient
    ) -> None:
        mock_redis_client._redis.llen = AsyncMock(return_value=0)
        mock_redis_client._redis.lpush = AsyncMock(return_value=1)

        request_id, accepted = await enqueue_inference_task(
            mock_redis_client, _make_payload()
        )

        assert accepted is True
        assert request_id.startswith("req-")

    @pytest.mark.asyncio
    async def test_enqueue_inference_task_backpressure(
        self, mock_redis_client: RedisClient
    ) -> None:
        # Simulate queue at capacity
        mock_redis_client._redis.llen = AsyncMock(return_value=5000)

        request_id, accepted = await enqueue_inference_task(
            mock_redis_client, _make_payload(), max_queue_depth=5000
        )

        assert accepted is False
        assert request_id == ""

    @pytest.mark.asyncio
    async def test_create_async_task_registers_state(
        self, mock_redis_client: RedisClient
    ) -> None:
        mock_redis_client._redis.llen = AsyncMock(return_value=0)
        mock_redis_client._redis.lpush = AsyncMock(return_value=1)
        mock_redis_client._redis.hset = AsyncMock()

        request_id, accepted = await create_async_task(
            mock_redis_client, _make_payload()
        )

        assert accepted is True
        # Verify hset was called to register task state
        mock_redis_client._redis.hset.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_task_result_returns_data(
        self, mock_redis_client: RedisClient
    ) -> None:
        mock_redis_client._redis.hgetall = AsyncMock(
            return_value={
                "status": "COMPLETED",
                "model": "gpt-2",
                "result": json.dumps({"choices": []}),
            }
        )

        result = await get_task_result(mock_redis_client, "req-123")

        assert result is not None
        assert result["status"] == "COMPLETED"
        assert result["result"] == {"choices": []}

    @pytest.mark.asyncio
    async def test_get_task_result_returns_none_for_missing(
        self, mock_redis_client: RedisClient
    ) -> None:
        mock_redis_client._redis.hgetall = AsyncMock(return_value={})

        result = await get_task_result(mock_redis_client, "nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
#  Gateway Degraded Mode (no Redis)
# ---------------------------------------------------------------------------


class TestGatewayDegradedMode:
    """Verify the gateway still responds when Redis is unavailable."""

    def test_health_without_redis(self) -> None:
        """Health endpoint should return healthy even without Redis."""
        from fastapi.testclient import TestClient

        # Patch the module-level _redis_client to None
        with patch("gateway.main._redis_client", None):
            from gateway.main import app
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["redis_connected"] is False

    def test_completions_stub_without_redis(self) -> None:
        """Chat completions should return stub when Redis is down."""
        from fastapi.testclient import TestClient

        with patch("gateway.main._redis_client", None):
            from gateway.main import app
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-coreai-development-key-2026"},
                json={
                    "model": "gpt-2",
                    "messages": [{"role": "user", "content": "test"}],
                },
            )
            assert response.status_code == 200
            assert "[STUB]" in response.json()["choices"][0]["message"]["content"]
