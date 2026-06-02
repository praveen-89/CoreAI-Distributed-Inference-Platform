"""
CoreAI Distributed Inference Platform - Result Service

Provides synchronous waiting for asynchronous inference results using
Redis Pub/Sub. When a client calls ``POST /v1/chat/completions``, the
gateway enqueues the task and then blocks (using this module) until
the worker publishes a completion event.
"""

from __future__ import annotations

import asyncio
import json
import logging

from shared.redis_client import RedisClient

logger = logging.getLogger("coreai.gateway.result")


class TimeoutException(Exception):
    """Raised when the inference task does not complete within the timeout."""
    pass


async def wait_for_inference_result(
    redis_client: RedisClient,
    request_id: str,
    timeout: float = 30.0,
) -> dict:
    """Block until an inference task completes and return its result.

    Uses Redis Pub/Sub to listen for completion notifications from the
    worker. Handles race conditions where the worker finishes before
    the gateway can subscribe.

    Args:
        redis_client: The shared async Redis client.
        request_id: The unique task ID to wait for.
        timeout: Maximum seconds to wait before raising TimeoutException.

    Returns:
        The deserialised ChatCompletionResponse JSON dictionary.

    Raises:
        TimeoutException: If the task does not complete in time.
        RuntimeError: If the task failed during processing.
    """
    # ── Race condition check ───────────────────────────────────────────
    # The worker might have finished and published before we even
    # got here. Check the cache first.
    cached = await redis_client.get_result(request_id)
    if cached:
        logger.debug("Result for '%s' found in cache immediately.", request_id)
        await redis_client.delete_result(request_id)
        return json.loads(cached)

    # ── Subscribe to channel ───────────────────────────────────────────
    pubsub = redis_client.subscribe_result_channel(request_id)
    channel = f"channel:result:{request_id}"
    await pubsub.subscribe(channel)

    # Double check cache just in case it finished during subscription
    cached = await redis_client.get_result(request_id)
    if cached:
        await pubsub.unsubscribe(channel)
        await pubsub.close()
        await redis_client.delete_result(request_id)
        return json.loads(cached)

    logger.debug("Subscribed to '%s', waiting up to %.1fs", channel, timeout)

    # ── Wait for notification ──────────────────────────────────────────
    async def _listen_for_message() -> None:
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = message["data"]
                # Assuming data is string 'DONE'
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                if data == "DONE":
                    return

    try:
        await asyncio.wait_for(_listen_for_message(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for result '%s' (%.1fs)", request_id, timeout)
        raise TimeoutException(f"Inference timed out after {timeout} seconds.") from None
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()

    # ── Retrieve and return result ─────────────────────────────────────
    # At this point, the worker published DONE.
    result_str = await redis_client.get_result(request_id)
    if not result_str:
        # This implies a failure or missing cache. Check task status.
        task_status = await redis_client.get_task_status(request_id)
        if task_status and task_status.get("status") == "FAILED":
            error_msg = task_status.get("error", "Unknown error in worker.")
            raise RuntimeError(f"Inference failed: {error_msg}")
        raise RuntimeError("Notification received but result cache was empty.")

    await redis_client.delete_result(request_id)
    return json.loads(result_str)
