"""
Distributed Inference Platform - Worker Heartbeat

Provides a background task that periodically registers the worker's
liveliness and supported model in Redis. This allows the gateway to
know which models are currently available to serve requests.
"""

from __future__ import annotations

import asyncio
import logging

from shared.redis_client import RedisClient

logger = logging.getLogger("coreai.worker.heartbeat")


async def heartbeat_loop(
    redis_client: RedisClient,
    worker_id: str,
    model_id: str,
    interval: float = 5.0,
    ttl: int = 15,
) -> None:
    """Continuously register the worker in Redis until cancelled.

    Args:
        redis_client: Active Redis connection.
        worker_id: Unique identifier for this worker.
        model_id: The model being served.
        interval: Seconds between heartbeats.
        ttl: Seconds before the heartbeat expires in Redis.
    """
    logger.info("Heartbeat loop started for worker '%s' (model: %s)", worker_id, model_id)
    
    try:
        while True:
            try:
                await redis_client.register_worker(worker_id, model_id, ttl=ttl)
                logger.debug("Heartbeat registered (TTL: %ds)", ttl)
            except Exception:
                logger.warning("Failed to register heartbeat, will retry.")
                
            await asyncio.sleep(interval)
            
    except asyncio.CancelledError:
        logger.info("Heartbeat loop cancelled.")
        raise
