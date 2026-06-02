"""
CoreAI Distributed Inference Platform - Registry Service

Provides functionality to query the currently active workers from Redis
to determine which models are available for serving.
"""

from __future__ import annotations

import logging

from gateway.schemas import ModelInfo
from shared.redis_client import RedisClient

logger = logging.getLogger("coreai.gateway.registry")


async def get_available_models(redis_client: RedisClient) -> list[ModelInfo]:
    """Retrieve a list of models currently being served by active workers.
    
    If Redis is unavailable, returns an empty list or a default fallback.
    """
    try:
        active_models = await redis_client.get_active_models()
        return [ModelInfo(id=m) for m in sorted(active_models)]
    except Exception:
        logger.exception("Failed to retrieve active models from registry.")
        return []
