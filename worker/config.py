"""
CoreAI Distributed Inference Platform - Worker Configuration

Loads all worker-level environment variables using python-dotenv and exposes
them as a validated Pydantic Settings object.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")


class WorkerSettings(BaseSettings):
    """Configuration for the PyTorch Worker daemon."""

    worker_id: str = Field(
        default_factory=lambda: f"worker-{os.getpid()}",
        description="Unique identifier for this worker instance.",
    )
    
    # Model Configuration
    model_id: str = Field(
        default="gpt2",
        description="The HuggingFace model ID to load and serve.",
    )
    device: str = Field(
        default="cpu",
        description="PyTorch device to use (e.g. 'cpu', 'cuda', 'mps').",
    )

    # Redis Connection
    redis_host: str = Field(default="localhost", description="Redis server hostname.")
    redis_port: int = Field(default=6379, description="Redis server port.")
    redis_password: str = Field(default="", description="Redis AUTH password.")
    redis_db: int = Field(default=0, description="Redis logical database index.")

    # Worker Settings
    poll_interval: float = Field(
        default=0.1,
        description="Seconds to wait between Redis queue polls.",
    )

    model_config = {
        "env_file": _ENV_FILE,
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache(maxsize=1)
def get_settings() -> WorkerSettings:
    """Return a cached, singleton instance of the worker settings."""
    return WorkerSettings()
