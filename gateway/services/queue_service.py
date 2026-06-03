"""
Distributed Inference Platform - Queue Service

Provides the gateway-side logic for enqueuing inference requests into
Redis and managing async task state.  This module acts as the bridge
between the FastAPI route handlers and the shared ``RedisClient``.

Responsibilities:
    1. Build the **task envelope** — a JSON-serialisable dict containing
       the request_id, model, messages, parameters, and metadata.
    2. Push the envelope into the model-specific Redis List queue.
    3. Create / update the ``task:<request_id>`` hash for async status
       tracking.
    4. Enforce **backpressure** by checking queue depth before enqueuing.
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from gateway.schemas import ChatCompletionRequest, TaskStatus
from shared.redis_client import RedisClient

logger = logging.getLogger("coreai.gateway.queue")


def generate_request_id() -> str:
    """Generate a globally unique request identifier.

    Format: ``req-<uuid4>`` to avoid collisions across gateway replicas.
    """
    return f"req-{uuid.uuid4()}"


def build_task_envelope(
    request_id: str,
    payload: ChatCompletionRequest,
) -> dict:
    """Construct the task envelope that gets serialised into Redis.

    The envelope contains everything the worker needs to execute
    inference without calling back to the gateway.

    Returns:
        A plain ``dict`` ready for ``json.dumps()``.
    """
    return {
        "request_id": request_id,
        "model": payload.model,
        "messages": [
            {"role": msg.role.value, "content": msg.content}
            for msg in payload.messages
        ],
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
        "created_at": time.time(),
    }


async def enqueue_inference_task(
    redis_client: RedisClient,
    payload: ChatCompletionRequest,
    *,
    max_queue_depth: int = 5000,
) -> tuple[str, bool]:
    """Validate queue capacity and push an inference task.

    Args:
        redis_client: Connected ``RedisClient`` instance.
        payload: Validated chat completion request from the client.
        max_queue_depth: Maximum pending tasks before rejecting with
            backpressure.

    Returns:
        A tuple of ``(request_id, accepted)`` where ``accepted`` is
        ``False`` if the queue exceeded the backpressure threshold.
    """
    # ── Backpressure check ─────────────────────────────────────────────
    depth = await redis_client.get_queue_depth(payload.model)
    if depth >= max_queue_depth:
        logger.warning(
            "Backpressure triggered for model '%s' (depth=%d, max=%d)",
            payload.model,
            depth,
            max_queue_depth,
        )
        return "", False

    # ── Build & enqueue ────────────────────────────────────────────────
    request_id = generate_request_id()
    envelope = build_task_envelope(request_id, payload)
    await redis_client.enqueue_task(payload.model, envelope)

    logger.info(
        "Task '%s' enqueued for model '%s' (depth=%d)",
        request_id,
        payload.model,
        depth + 1,
    )
    return request_id, True


async def create_async_task(
    redis_client: RedisClient,
    payload: ChatCompletionRequest,
    *,
    max_queue_depth: int = 5000,
) -> tuple[str, bool]:
    """Enqueue an inference task *and* register its initial async state.

    Combines ``enqueue_inference_task`` with a ``task:<request_id>``
    hash write so that ``GET /v1/tasks/{task_id}`` can return status
    immediately.

    Returns:
        ``(request_id, accepted)`` — same semantics as
        ``enqueue_inference_task``.
    """
    request_id, accepted = await enqueue_inference_task(
        redis_client, payload, max_queue_depth=max_queue_depth
    )
    if not accepted:
        return request_id, False

    # Register initial state in the task hash.
    await redis_client.set_task_status(
        request_id,
        TaskStatus.PENDING.value,
        extra={"model": payload.model},
    )
    return request_id, True


async def get_task_result(
    redis_client: RedisClient,
    task_id: str,
) -> dict | None:
    """Fetch the current state of an async task from Redis.

    Returns:
        A dict with ``status`` and optionally ``result`` keys, or
        ``None`` if the task does not exist.
    """
    data = await redis_client.get_task_status(task_id)
    if not data:
        return None

    result_dict: dict = {"task_id": task_id, "status": data.get("status", "UNKNOWN")}
    raw_result = data.get("result")
    if raw_result:
        result_dict["result"] = json.loads(raw_result)
    else:
        result_dict["result"] = None

    return result_dict
