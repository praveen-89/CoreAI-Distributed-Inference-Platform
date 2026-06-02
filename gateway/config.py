"""
CoreAI Distributed Inference Platform - Gateway Configuration

Loads all gateway-level environment variables using python-dotenv and exposes
them as a validated Pydantic Settings object. Every service component imports
this module to access centralised configuration values.

Environment variables are read from a `.env` file located at the project root
(if present) and can always be overridden by actual OS-level env vars or
Docker Compose `environment:` entries.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Determine the path to the `.env` file relative to the project root.
# When running inside Docker, the `.env` file is typically not mounted;
# instead, env vars are injected directly via docker-compose.yml.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")


class GatewaySettings(BaseSettings):
    """Typed, validated configuration for the API Gateway service.

    Attributes are populated from environment variables. Pydantic Settings
    performs automatic type coercion (e.g. ``"8000"`` → ``int``).
    """

    # ── Server ──────────────────────────────────────────────────────────
    gateway_host: str = Field(
        default="0.0.0.0",
        description="Bind address for the Uvicorn ASGI server.",
    )
    gateway_port: int = Field(
        default=8000,
        description="Port the API Gateway listens on.",
    )

    # ── Authentication ──────────────────────────────────────────────────
    coreai_api_key: str = Field(
        default="sk-coreai-development-key-2026",
        description="Bearer token that clients must present to access protected endpoints.",
    )

    # ── Redis ───────────────────────────────────────────────────────────
    redis_host: str = Field(default="localhost", description="Redis server hostname.")
    redis_port: int = Field(default=6379, description="Redis server port.")
    redis_password: str = Field(default="", description="Redis AUTH password.")
    redis_db: int = Field(default=0, description="Redis logical database index.")

    # ── Backpressure ────────────────────────────────────────────────────
    max_queue_depth: int = Field(
        default=5000,
        description="Maximum number of pending tasks in the queue before the gateway returns HTTP 429.",
    )

    # ── Timeouts ────────────────────────────────────────────────────────
    sync_request_timeout: int = Field(
        default=30,
        description="Maximum seconds to wait for a synchronous inference result before returning 504.",
    )

    model_config = {
        "env_file": _ENV_FILE,
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache(maxsize=1)
def get_settings() -> GatewaySettings:
    """Return a cached, singleton instance of the gateway settings.

    Using ``@lru_cache`` ensures the ``.env`` file is read exactly once per
    process lifetime, avoiding repeated filesystem I/O on every request.
    """
    return GatewaySettings()
