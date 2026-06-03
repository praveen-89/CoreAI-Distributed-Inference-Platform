"""
Distributed Inference Platform - Phase 7 Monitoring Tests
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from gateway.main import app


def test_metrics_endpoint_returns_prometheus_format():
    """Verify that the /metrics endpoint returns prometheus formatted text."""
    client = TestClient(app)
    
    # First make a request to trigger the middleware
    client.get("/health")
    
    response = client.get("/metrics")
    
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    
    # Check that standard Python/Prometheus metrics exist
    text = response.text
    assert "coreai_http_requests_total" in text
    assert "coreai_http_request_duration_seconds" in text


def test_metrics_middleware_records_requests():
    """Verify the middleware records the status codes and methods correctly."""
    client = TestClient(app)
    
    # Trigger a 401 Unauthorized via the main endpoint
    client.get("/v1/models")
    
    response = client.get("/metrics")
    text = response.text
    
    # The middleware should have recorded a GET request to /v1/models with 401
    assert 'coreai_http_requests_total{endpoint="/v1/models",method="GET",status_code="401"}' in text
