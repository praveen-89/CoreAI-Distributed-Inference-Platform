"""
Distributed Inference Platform - Monitoring & Metrics

Defines Prometheus metrics for observability and a middleware to automatically
record HTTP request latencies and counts.
"""

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest

# ── Metrics Definitions ───────────────────────────────────────────────

# Track total HTTP requests by method, endpoint, and status code
HTTP_REQUESTS_TOTAL = Counter(
    "coreai_http_requests_total",
    "Total number of HTTP requests processed by the API gateway.",
    ["method", "endpoint", "status_code"],
)

# Track request latency by method and endpoint
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "coreai_http_request_duration_seconds",
    "Histogram of HTTP request processing durations in seconds.",
    ["method", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# Track the number of tasks currently in the processing pipeline (enqueued + processing)
ACTIVE_INFERENCE_TASKS = Gauge(
    "coreai_active_inference_tasks",
    "Number of inference tasks currently active or queued in the system.",
    ["model"],
)


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """FastAPI middleware to automatically record Prometheus metrics for all requests.
    
    Excludes the ``/metrics`` and ``/health`` endpoints to avoid noise.
    """
    path = request.url.path
    
    # Skip noise endpoints
    if path in ("/metrics", "/health"):
        return await call_next(request)

    start_time = time.time()
    method = request.method
    
    # Handle the request
    try:
        response = await call_next(request)
        status_code = str(response.status_code)
    except Exception:
        # If an unhandled exception bubbles up, it's a 500
        status_code = "500"
        raise
    finally:
        # Record metrics
        duration = time.time() - start_time
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=path, status_code=status_code).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, endpoint=path).observe(duration)

    return response


def get_metrics() -> Response:
    """Return the current metrics in the Prometheus exposition format."""
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
