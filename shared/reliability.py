"""
Distributed Inference Platform - Reliability & Crash Recovery

Implements a background "reaper" process that detects crashed workers
by comparing the `processing:*:*` queues against the active worker
heartbeats. If a worker has died with tasks in flight, those tasks
are safely re-enqueued for another worker to process.
"""

from __future__ import annotations

import asyncio
import logging

from shared.redis_client import RedisClient

logger = logging.getLogger("coreai.shared.reliability")


async def reap_stale_tasks(redis_client: RedisClient) -> None:
    """Scan all processing queues and recover tasks from dead workers.

    A worker is considered dead if its `worker:<worker_id>` heartbeat
    key has expired from Redis. Any tasks in its processing queue are
    moved back to the main model queue.
    """
    try:
        # Find all processing queues
        processing_keys = await redis_client.redis.keys("processing:*:*")
        if not processing_keys:
            return

        for p_key in processing_keys:
            # p_key format: processing:<model_id>:<worker_id>
            parts = p_key.split(":")
            if len(parts) != 3:
                continue
            
            _, model_id, worker_id = parts
            
            # Check if worker is still alive
            worker_heartbeat_key = f"worker:{worker_id}"
            is_alive = await redis_client.redis.exists(worker_heartbeat_key)
            
            if not is_alive:
                # Worker is dead. Let's requeue all tasks in its processing queue.
                queue_key = f"queue:{model_id}"
                
                while True:
                    # RPOPLPUSH is atomic. Move from processing to queue.
                    # We use rpoplpush (sync version of brpoplpush)
                    task_raw = await redis_client.redis.rpoplpush(p_key, queue_key)
                    if not task_raw:
                        break  # No more tasks in this processing queue
                        
                    logger.warning(
                        "Reaped abandoned task from dead worker '%s' and re-enqueued to '%s'",
                        worker_id, queue_key
                    )
                
                # Clean up the empty processing queue just in case
                await redis_client.redis.delete(p_key)

    except Exception:
        logger.exception("Error during stale task reaping.")


async def reaper_loop(redis_client: RedisClient, interval: float = 10.0) -> None:
    """Continuously run the reaper process in the background."""
    logger.info("Starting reliability reaper loop (interval: %.1fs)", interval)
    try:
        while True:
            await asyncio.sleep(interval)
            await reap_stale_tasks(redis_client)
    except asyncio.CancelledError:
        logger.info("Reaper loop cancelled.")
        raise
