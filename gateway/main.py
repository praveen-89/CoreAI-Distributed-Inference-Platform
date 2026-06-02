"""
CoreAI Distributed Inference Platform - API Gateway (FastAPI Application)

This module is the primary entrypoint for the API Gateway service.  It
wires together all HTTP routes, middleware, and lifecycle hooks.

Endpoints:
    POST /v1/chat/completions  – Synchronous chat completion
    POST /v1/tasks             – Submit an asynchronous inference task
    GET  /v1/tasks/{task_id}   – Poll for async task status
    GET  /v1/models            – List available models
    GET  /health               – Liveness / readiness probe
    GET  /metrics              – Prometheus metrics (wired in Phase 7)

The gateway connects to Redis on startup and tears down the pool on
shutdown.  Queue enqueuing and async task state are handled by the
``queue_service`` module.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from gateway.config import GatewaySettings, get_settings
from gateway.monitoring import get_metrics, metrics_middleware
from gateway.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    Role,
    TaskStatus,
    TaskStatusResponse,
    TaskSubmitResponse,
)
from gateway.services.queue_service import (
    create_async_task,
    enqueue_inference_task,
    get_task_result,
)
from gateway.services.registry_service import get_available_models
from gateway.services.result_service import TimeoutException, wait_for_inference_result
from shared.redis_client import RedisClient
from shared.reliability import reaper_loop

# ---------------------------------------------------------------------------
#  Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("coreai.gateway")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
#  Module-level Redis client (initialised in lifespan)
# ---------------------------------------------------------------------------
_redis_client: RedisClient | None = None


def get_redis_client() -> RedisClient:
    """FastAPI dependency that returns the shared Redis client.

    Raises ``503 Service Unavailable`` if Redis was not initialised.
    """
    if _redis_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is not available.",
        )
    return _redis_client


# ---------------------------------------------------------------------------
#  Application Lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook.

    On startup: create the Redis connection pool and verify connectivity.
    On shutdown: gracefully close the pool.
    """
    global _redis_client  # noqa: PLW0603

    settings = get_settings()
    logger.info(
        "CoreAI Gateway starting on %s:%s",
        settings.gateway_host,
        settings.gateway_port,
    )

    # ── Redis connection ───────────────────────────────────────────────
    _redis_client = RedisClient(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        db=settings.redis_db,
    )

    # We will store the reaper task so we can cancel it later
    reaper_task: asyncio.Task | None = None

    try:
        await _redis_client.connect()
        logger.info("Redis connection pool initialised.")
        
        # Start the reliability reaper in the background
        reaper_task = asyncio.create_task(reaper_loop(_redis_client))
    except Exception:
        logger.warning(
            "Could not connect to Redis at %s:%s – gateway will run in degraded mode.",
            settings.redis_host,
            settings.redis_port,
        )
        _redis_client = None

    yield

    # ── Teardown ───────────────────────────────────────────────────────
    if reaper_task:
        reaper_task.cancel()
        
    if _redis_client is not None:
        await _redis_client.disconnect()
    logger.info("CoreAI Gateway shutting down.")


# ---------------------------------------------------------------------------
#  FastAPI Application Instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CoreAI Distributed Inference Platform",
    description=(
        "A simplified, high-performance Azure OpenAI-style inference serving "
        "platform using FastAPI, PyTorch, Redis queues, and Docker."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Register prometheus metrics middleware
app.middleware("http")(metrics_middleware)


# ---------------------------------------------------------------------------
#  Authentication Dependency
# ---------------------------------------------------------------------------


async def verify_api_key(
    request: Request,
    settings: GatewaySettings = Depends(get_settings),
) -> None:
    """Validate the ``Authorization: Bearer <token>`` header.

    Raises ``401 Unauthorized`` if the token is missing or does not match
    the configured ``COREAI_API_KEY``.
    """
    auth_header: str | None = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected 'Bearer <api_key>'.",
        )
    token = auth_header.removeprefix("Bearer ").strip()
    if token != settings.coreai_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )


# ---------------------------------------------------------------------------
#  Routes — Chat Completions
# ---------------------------------------------------------------------------


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
    tags=["Chat"],
    summary="Create a chat completion",
)
async def create_chat_completion(
    payload: ChatCompletionRequest,
    settings: GatewaySettings = Depends(get_settings),
) -> ChatCompletionResponse:
    """Accept a chat completion request and enqueue it for processing.

    If Redis is available the request is pushed to the model queue.
    A stub response is returned until result retrieval is wired in
    Phase 5.

    Returns HTTP 429 if the queue exceeds the backpressure threshold.
    """
    logger.info("Received chat completion request for model '%s'", payload.model)

    # ── Enqueue to Redis if connected ──────────────────────────────────
    if _redis_client is not None:
        request_id, accepted = await enqueue_inference_task(
            _redis_client,
            payload,
            max_queue_depth=settings.max_queue_depth,
        )
        if not accepted:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Server is at capacity. Please retry later.",
            )
        logger.info("Request '%s' enqueued. Waiting for result...", request_id)

        try:
            result_dict = await wait_for_inference_result(
                _redis_client,
                request_id,
                timeout=settings.sync_request_timeout,
            )
            return ChatCompletionResponse(**result_dict)
        except TimeoutException as e:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=str(e),
            )
        except RuntimeError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    # ── Fallback if Redis is unavailable ───────────────────────────────
    logger.warning("Redis not available. Returning stub response.")
    return ChatCompletionResponse(
        model=payload.model,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role=Role.ASSISTANT,
                    content=f"[STUB] Echo from model '{payload.model}': "
                    f"{payload.messages[-1].content}",
                ),
                finish_reason="stop",
            )
        ],
    )


# ---------------------------------------------------------------------------
#  Routes — Async Task Submission
# ---------------------------------------------------------------------------


@app.post(
    "/v1/tasks",
    response_model=TaskSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
    tags=["Tasks"],
    summary="Submit an async inference task",
)
async def submit_task(
    payload: ChatCompletionRequest,
    settings: GatewaySettings = Depends(get_settings),
) -> TaskSubmitResponse:
    """Submit a task for asynchronous processing.

    Enqueues the task into Redis and registers its initial PENDING state.
    Returns HTTP 429 if backpressure threshold is exceeded.
    """
    logger.info("Async task submitted for model '%s'", payload.model)

    if _redis_client is not None:
        request_id, accepted = await create_async_task(
            _redis_client,
            payload,
            max_queue_depth=settings.max_queue_depth,
        )
        if not accepted:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Server is at capacity. Please retry later.",
            )
        return TaskSubmitResponse(
            task_id=request_id,
            status=TaskStatus.PENDING,
            model=payload.model,
        )

    # ── Fallback if Redis is unavailable ───────────────────────────────
    return TaskSubmitResponse(model=payload.model)


@app.get(
    "/v1/tasks/{task_id}",
    response_model=TaskStatusResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
    tags=["Tasks"],
    summary="Poll task status",
)
async def get_task_status_route(task_id: str) -> TaskStatusResponse:
    """Retrieve the current status of an async task from Redis."""
    logger.info("Status poll for task '%s'", task_id)

    if _redis_client is not None:
        result = await get_task_result(_redis_client, task_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task '{task_id}' not found.",
            )
        return TaskStatusResponse(
            task_id=result["task_id"],
            status=TaskStatus(result["status"]),
            result=result.get("result"),
        )

    # ── Fallback: stub ─────────────────────────────────────────────────
    return TaskStatusResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        result=None,
    )


# ---------------------------------------------------------------------------
#  Routes — Model Catalog
# ---------------------------------------------------------------------------


@app.get(
    "/v1/models",
    response_model=ModelListResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Models"],
    summary="List available models",
)
async def list_models() -> ModelListResponse:
    """Return the catalog of models currently served by active workers.
    
    If Redis is unavailable, returns a static fallback list.
    """
    if _redis_client is not None:
        models = await get_available_models(_redis_client)
        if models:
            return ModelListResponse(data=models)

    # ── Fallback ───────────────────────────────────────────────────────
    logger.warning("Falling back to static model list.")
    return ModelListResponse(
        data=[
            ModelInfo(id="gpt-2"),
        ]
    )


# ---------------------------------------------------------------------------
#  Routes — Health & Diagnostics
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Diagnostics"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Liveness probe for container orchestrators and load balancers.

    No authentication is required so that Kubernetes probes, Docker
    HEALTHCHECK, and external monitors can reach it without credentials.
    Reports actual Redis connectivity status.
    """
    redis_ok = False
    if _redis_client is not None:
        redis_ok = await _redis_client.ping()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        redis_connected=redis_ok,
    )


@app.get(
    "/metrics",
    tags=["Diagnostics"],
    summary="Prometheus metrics",
)
async def metrics_route() -> Response:
    """Expose Prometheus metrics for scraping."""
    return get_metrics()


# ---------------------------------------------------------------------------
#  Global Exception Handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler that returns a consistent JSON error envelope.

    Prevents raw stack traces from leaking to API consumers in production.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            detail=str(exc),
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
#  Development Server Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "gateway.main:app",
        host=settings.gateway_host,
        port=settings.gateway_port,
        reload=True,
    )
