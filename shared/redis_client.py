"""
CoreAI Distributed Inference Platform - Shared Redis Client

Provides a centralised, async Redis connection pool and a high-level
helper class for all Redis operations used across the platform (gateway,
workers, monitoring).

Design decisions:
    • Uses ``redis.asyncio`` for non-blocking I/O compatible with FastAPI's
      async event loop.
    • A single ``ConnectionPool`` is shared across all callers within a
      process to avoid socket exhaustion.
    • All public methods are thin wrappers around raw Redis commands,
      keeping serialisation (JSON) explicit at the call site for
      transparency and debuggability.

Key patterns:
    queue:<model>                    – Redis List used as the task queue.
    processing:<model>:<worker_id>  – Redis List tracking in-flight tasks
                                       for crash recovery (Phase 9).
    task:<request_id>               – Redis Hash holding async task metadata.
    result:<request_id>             – Redis String caching inference output.
    channel:result:<request_id>     – Redis Pub/Sub channel for sync result
                                       notification.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger("coreai.redis")


class RedisClient:
    """Async Redis helper wrapping connection pool lifecycle and
    domain-specific queue / hash / pubsub operations.

    Usage::

        client = RedisClient(host="localhost", port=6379, password="secret")
        await client.connect()
        # ... use client ...
        await client.disconnect()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str = "",
        db: int = 0,
    ) -> None:
        self._host = host
        self._port = port
        self._password = password
        self._db = db
        self._pool: aioredis.ConnectionPool | None = None
        self._redis: aioredis.Redis | None = None

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the connection pool and verify connectivity with PING."""
        self._pool = aioredis.ConnectionPool(
            host=self._host,
            port=self._port,
            password=self._password or None,
            db=self._db,
            decode_responses=True,
            max_connections=50,
        )
        self._redis = aioredis.Redis(connection_pool=self._pool)
        # Validate connectivity early so startup fails fast.
        await self._redis.ping()
        logger.info("Redis connection established → %s:%s/%s", self._host, self._port, self._db)

    async def disconnect(self) -> None:
        """Gracefully tear down the connection pool."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        logger.info("Redis connection closed.")

    @property
    def redis(self) -> aioredis.Redis:
        """Return the underlying ``aioredis.Redis`` instance.

        Raises ``RuntimeError`` if called before ``connect()``.
        """
        if self._redis is None:
            raise RuntimeError("RedisClient is not connected. Call connect() first.")
        return self._redis

    async def ping(self) -> bool:
        """Return ``True`` if Redis responds to PING."""
        try:
            return await self.redis.ping()
        except Exception:
            return False

    # ── Queue Operations ────────────────────────────────────────────────

    async def enqueue_task(self, model: str, task_payload: dict[str, Any]) -> int:
        """Push a serialised task onto the model-specific queue.

        Args:
            model: Model identifier (e.g. ``"gpt-2"``).  Determines
                the Redis List key ``queue:<model>``.
            task_payload: Dictionary containing the full task envelope
                (request_id, messages, params, etc.).

        Returns:
            The new length of the queue after the push.
        """
        queue_key = f"queue:{model}"
        serialised = json.dumps(task_payload)
        length = await self.redis.lpush(queue_key, serialised)
        logger.debug("Enqueued task to '%s' (queue depth: %d)", queue_key, length)
        return length

    async def get_queue_depth(self, model: str) -> int:
        """Return the number of pending tasks in a model queue."""
        return await self.redis.llen(f"queue:{model}")

    async def dequeue_task(
        self, model: str, worker_id: str, timeout: int = 1
    ) -> tuple[dict[str, Any], str] | tuple[None, None]:
        """Block and reliably pop a task from the queue into a processing queue.
        
        Args:
            model: Model identifier.
            worker_id: The ID of the worker pulling this task.
            timeout: Seconds to block waiting for an item.
            
        Returns:
            A tuple of (parsed JSON task dict, raw string payload),
            or (None, None) if the timeout expired.
        """
        queue_key = f"queue:{model}"
        processing_key = f"processing:{model}:{worker_id}"
        
        # BRPOPLPUSH blocks until an item is available, pops it, and pushes it to processing_key atomically.
        # Note: aioredis >= 2.0 provides bzpopmin/etc, but for lists brpoplpush is standard.
        payload_str = await self.redis.brpoplpush(queue_key, processing_key, timeout=timeout)
        if payload_str:
            return json.loads(payload_str), payload_str
        return None, None

    async def acknowledge_task(self, model: str, worker_id: str, raw_payload: str) -> None:
        """Remove a completed task from the worker's processing queue."""
        processing_key = f"processing:{model}:{worker_id}"
        await self.redis.lrem(processing_key, 1, raw_payload)

    # ── Async Task State (Hash) ─────────────────────────────────────────

    async def set_task_status(
        self,
        request_id: str,
        status: str,
        *,
        result: str | None = None,
        extra: dict[str, str] | None = None,
    ) -> None:
        """Create or update the task metadata hash ``task:<request_id>``.

        Args:
            request_id: Unique task identifier.
            status: Current lifecycle state (PENDING, PROCESSING, etc.).
            result: Optional JSON-serialised inference result.
            extra: Additional key-value pairs to store.
        """
        key = f"task:{request_id}"
        mapping: dict[str, str] = {"status": status}
        if result is not None:
            mapping["result"] = result
        if extra:
            mapping.update(extra)
        await self.redis.hset(key, mapping=mapping)

    async def get_task_status(self, request_id: str) -> dict[str, str] | None:
        """Retrieve all fields of the task hash.

        Returns ``None`` if the key does not exist.
        """
        data = await self.redis.hgetall(f"task:{request_id}")
        return data if data else None

    # ── Sync Result Cache ───────────────────────────────────────────────

    async def set_result(self, request_id: str, result: str, ttl: int = 300) -> None:
        """Store an inference result string with a TTL (seconds).

        Used by workers to cache completed outputs for gateway retrieval.
        """
        await self.redis.set(f"result:{request_id}", result, ex=ttl)

    async def get_result(self, request_id: str) -> str | None:
        """Fetch and return a cached result, or ``None`` if absent."""
        return await self.redis.get(f"result:{request_id}")

    async def delete_result(self, request_id: str) -> None:
        """Remove a cached result after the gateway has served it."""
        await self.redis.delete(f"result:{request_id}")

    async def result_exists(self, request_id: str) -> bool:
        """Check whether a result key exists (race-condition fallback)."""
        return bool(await self.redis.exists(f"result:{request_id}"))

    # ── Worker Registry ─────────────────────────────────────────────────

    async def register_worker(self, worker_id: str, model_id: str, ttl: int = 15) -> None:
        """Register a worker heartbeat with a time-to-live.
        
        Args:
            worker_id: The unique ID of the worker.
            model_id: The model this worker is serving.
            ttl: Time in seconds before this registration expires.
        """
        key = f"worker:{worker_id}"
        await self.redis.set(key, model_id, ex=ttl)

    async def get_active_models(self) -> set[str]:
        """Scan active workers and return the set of currently served models."""
        models = set()
        # In a real huge cluster, scan is better than keys, but keys is fine for portfolio
        keys = await self.redis.keys("worker:*")
        for key in keys:
            model = await self.redis.get(key)
            if model:
                models.add(model)
        return models

    # ── Pub/Sub ─────────────────────────────────────────────────────────

    async def publish_result_ready(self, request_id: str) -> None:
        """Notify listening gateway instances that a result is ready."""
        channel = f"channel:result:{request_id}"
        await self.redis.publish(channel, "DONE")
        logger.debug("Published DONE to '%s'", channel)

    def subscribe_result_channel(self, request_id: str) -> aioredis.client.PubSub:
        """Return a PubSub object subscribed to the result channel.

        The caller is responsible for listening and unsubscribing.
        """
        pubsub = self.redis.pubsub()
        return pubsub
