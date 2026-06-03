"""
Distributed Inference Platform - Pydantic Request / Response Schemas

Defines the data contracts for all API Gateway endpoints. These schemas
enforce strict input validation via Pydantic v2 and generate automatic
OpenAPI documentation.

The naming and structure intentionally mirror the OpenAI API specification
(https://platform.openai.com/docs/api-reference/chat) to enable drop-in
compatibility with existing SDKs such as LangChain, Semantic Kernel, and
the official ``openai`` Python package.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
#  Enumerations
# ═══════════════════════════════════════════════════════════════════════════


class Role(str, Enum):
    """Roles in a chat conversation following the OpenAI convention."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class TaskStatus(str, Enum):
    """Lifecycle states for an asynchronous inference task."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ═══════════════════════════════════════════════════════════════════════════
#  Chat Completions – Request Models
# ═══════════════════════════════════════════════════════════════════════════


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Role = Field(..., description="The role of the message author.")
    content: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="The text content of the message.",
    )


class ChatCompletionRequest(BaseModel):
    """Request payload for ``POST /v1/chat/completions``.

    Mirrors the OpenAI Chat Completions API schema so existing SDKs can
    integrate without modification.
    """

    model: str = Field(
        ...,
        min_length=1,
        description="ID of the model to use (e.g. 'gpt-2').",
    )
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="A list of messages comprising the conversation so far.",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature between 0 and 2.",
    )
    max_tokens: int = Field(
        default=100,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate in the completion.",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Chat Completions – Response Models
# ═══════════════════════════════════════════════════════════════════════════


class ChoiceMessage(BaseModel):
    """The generated message inside a completion choice."""

    role: Role = Role.ASSISTANT
    content: str = ""


class Choice(BaseModel):
    """A single completion choice returned by the model."""

    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Token usage statistics for the completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Response payload for ``POST /v1/chat/completions``.

    Structure matches the OpenAI API response format for maximum SDK
    compatibility.
    """

    id: str = Field(
        default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}",
    )
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[Choice] = []
    usage: Usage = Usage()


# ═══════════════════════════════════════════════════════════════════════════
#  Async Task Submission – Request & Response Models
# ═══════════════════════════════════════════════════════════════════════════


class TaskSubmitResponse(BaseModel):
    """Response payload for ``POST /v1/tasks``."""

    task_id: str = Field(
        default_factory=lambda: f"task-{uuid.uuid4()}",
    )
    status: TaskStatus = TaskStatus.PENDING
    model: str = ""


class TaskStatusResponse(BaseModel):
    """Response payload for ``GET /v1/tasks/{task_id}``."""

    task_id: str
    status: TaskStatus
    result: ChatCompletionResponse | None = None


# ═══════════════════════════════════════════════════════════════════════════
#  Model Catalog – Response Models
# ═══════════════════════════════════════════════════════════════════════════


class ModelInfo(BaseModel):
    """Metadata for a single model available on the platform."""

    id: str
    object: str = "model"
    owned_by: str = "coreai-platform"
    status: str = "active"


class ModelListResponse(BaseModel):
    """Response payload for ``GET /v1/models``."""

    data: list[ModelInfo] = []


# ═══════════════════════════════════════════════════════════════════════════
#  Health Check – Response Model
# ═══════════════════════════════════════════════════════════════════════════


class HealthResponse(BaseModel):
    """Response payload for ``GET /health``."""

    status: str = "healthy"
    version: str = "1.0.0"
    redis_connected: bool = False


# ═══════════════════════════════════════════════════════════════════════════
#  Error – Response Model
# ═══════════════════════════════════════════════════════════════════════════


class ErrorResponse(BaseModel):
    """Standard error envelope returned on 4xx / 5xx responses."""

    error: str
    detail: str = ""
