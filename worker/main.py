"""
CoreAI Distributed Inference Platform - Worker Daemon

Long-running Python process that starts up, loads a PyTorch model into memory,
and continuously polls the Redis queue for inference tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from typing import Any

from shared.redis_client import RedisClient
from worker.config import get_settings
from worker.heartbeat import heartbeat_loop
from worker.model_runner import ModelRunner

logger = logging.getLogger("coreai.worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


class WorkerDaemon:
    """Manages the lifecycle of a single worker node."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.redis = RedisClient(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            password=self.settings.redis_password,
            db=self.settings.redis_db,
        )
        self.runner = ModelRunner(model_id=self.settings.model_id, device=self.settings.device)
        self._shutdown = False
        self._heartbeat_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the worker daemon."""
        logger.info("Worker '%s' starting up...", self.settings.worker_id)
        
        # Load model synchronously (CPU/GPU bound operation)
        self.runner.load()
        
        await self.redis.connect()
        logger.info("Worker '%s' connected to Redis.", self.settings.worker_id)
        
        self.setup_signal_handlers()
        
        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(
            heartbeat_loop(
                self.redis,
                self.settings.worker_id,
                self.settings.model_id,
            )
        )
        
        await self.loop()

    def setup_signal_handlers(self) -> None:
        """Hook OS signals to allow graceful shutdown."""
        loop = asyncio.get_running_loop()
        
        def _trigger() -> None:
            logger.info("Shutdown signal received. Finishing current task...")
            self._shutdown = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _trigger)
            except NotImplementedError:
                # Windows event loop may not implement add_signal_handler
                signal.signal(sig, lambda *_: _trigger())

    async def loop(self) -> None:
        """Continuously poll the Redis queue for work."""
        logger.info("Entering polling loop for queue:'%s'", self.settings.model_id)
        while not self._shutdown:
            try:
                task, raw_payload = await self.redis.dequeue_task(
                    self.settings.model_id, self.settings.worker_id, timeout=1
                )
                if not task:
                    continue
                
                await self.process_task(task, raw_payload)
            except Exception:
                logger.exception("Error in worker loop")
                await asyncio.sleep(1)
                
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            
        await self.redis.disconnect()
        logger.info("Worker '%s' shut down successfully.", self.settings.worker_id)

    async def process_task(self, task: dict[str, Any], raw_payload: str) -> None:
        """Execute a single inference task and report the result."""
        request_id = task.get("request_id")
        if not request_id:
            logger.error("Encountered task without a request_id. Discarding.")
            return

        logger.info("Processing task '%s'", request_id)
        
        try:
            # Update state to PROCESSING
            await self.redis.set_task_status(request_id, "PROCESSING")
            
            messages = task.get("messages", [])
            max_tokens = task.get("max_tokens", 100)
            temperature = task.get("temperature", 1.0)
            
            # Run inference in a thread pool so we don't block the async event loop
            loop = asyncio.get_running_loop()
            result_dict = await loop.run_in_executor(
                None, 
                lambda: self.runner.generate(messages, max_tokens=max_tokens, temperature=temperature)
            )
            
            # Format output correctly to match ChatCompletionResponse structure
            response = {
                "id": request_id,
                "object": "chat.completion",
                "model": self.settings.model_id,
                "choices": [result_dict["choice"]],
                "usage": result_dict["usage"],
            }
            
            result_json = json.dumps(response)
            
            # Store result and update status
            await self.redis.set_result(request_id, result_json)
            await self.redis.set_task_status(request_id, "COMPLETED", result=result_json)
            
            # Notify any listening gateway instances
            await self.redis.publish_result_ready(request_id)
            
            logger.info("Completed task '%s'", request_id)
            
        except Exception as e:
            logger.exception("Failed to process task '%s'", request_id)
            await self.redis.set_task_status(request_id, "FAILED", extra={"error": str(e)})
        finally:
            # Always acknowledge the task so it isn't retried if we explicitly handled it
            # (Success or explicit Failure). If the worker crashes mid-processing,
            # this won't run, leaving the task in the processing queue for the reaper.
            if raw_payload:
                await self.redis.acknowledge_task(
                    self.settings.model_id, self.settings.worker_id, raw_payload
                )


if __name__ == "__main__":
    daemon = WorkerDaemon()
    asyncio.run(daemon.start())
