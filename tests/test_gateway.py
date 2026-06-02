"""
CoreAI Distributed Inference Platform – Phase 2 Gateway Tests

Tests cover:
  • Health endpoint (no auth required)
  • API key authentication (valid, invalid, missing)
  • Chat completions endpoint (stub response validation)
  • Async task submission and polling (stub)
  • Model catalog listing
  • Pydantic input validation (bad payloads)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gateway.main import app

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

# The default API key set in gateway/config.py when no .env file is present.
_VALID_API_KEY = "sk-coreai-development-key-2026"
_AUTH_HEADER = {"Authorization": f"Bearer {_VALID_API_KEY}"}


@pytest.fixture()
def client() -> TestClient:
    """Create a synchronous test client bound to the FastAPI app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
#  Health Endpoint (unauthenticated)
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """``GET /health`` should always be accessible without auth."""

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_body(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["redis_connected"] is False


# ---------------------------------------------------------------------------
#  Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Endpoints protected by ``verify_api_key`` must enforce auth."""

    def test_missing_auth_header(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "model": "gpt-2",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert response.status_code == 401

    def test_invalid_api_key(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong-key"},
            json={
                "model": "gpt-2",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 401

    def test_valid_api_key_passes(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            headers=_AUTH_HEADER,
            json={
                "model": "gpt-2",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
#  Chat Completions (Stub)
# ---------------------------------------------------------------------------


class TestChatCompletions:
    """``POST /v1/chat/completions`` stub behaviour."""

    def test_stub_response_structure(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            headers=_AUTH_HEADER,
            json={
                "model": "gpt-2",
                "messages": [{"role": "user", "content": "What is 1+1?"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "gpt-2"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "[STUB]" in data["choices"][0]["message"]["content"]

    def test_validation_error_empty_messages(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            headers=_AUTH_HEADER,
            json={"model": "gpt-2", "messages": []},
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_validation_error_missing_model(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            headers=_AUTH_HEADER,
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
#  Async Tasks (Stub)
# ---------------------------------------------------------------------------


class TestAsyncTasks:
    """``POST /v1/tasks`` and ``GET /v1/tasks/{task_id}`` stub behaviour."""

    def test_submit_task_returns_202(self, client: TestClient) -> None:
        response = client.post(
            "/v1/tasks",
            headers=_AUTH_HEADER,
            json={
                "model": "gpt-2",
                "messages": [{"role": "user", "content": "Summarise this"}],
            },
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "PENDING"
        assert data["task_id"].startswith("task-")

    def test_poll_task_returns_pending(self, client: TestClient) -> None:
        response = client.get(
            "/v1/tasks/task-abc-123",
            headers=_AUTH_HEADER,
        )
        assert response.status_code == 200
        assert response.json()["status"] == "PENDING"


# ---------------------------------------------------------------------------
#  Model Catalog (Stub)
# ---------------------------------------------------------------------------


class TestModelCatalog:
    """``GET /v1/models`` stub behaviour."""

    def test_list_models(self, client: TestClient) -> None:
        response = client.get("/v1/models", headers=_AUTH_HEADER)
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) >= 1
        assert data["data"][0]["id"] == "gpt-2"
