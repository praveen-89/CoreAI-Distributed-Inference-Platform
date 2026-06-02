# CoreAI Distributed Inference Platform

The **CoreAI Distributed Inference Platform** is a simplified, high-performance, Azure OpenAI-style inference serving platform built using Python, FastAPI, PyTorch, and Redis. It demonstrates the decoupling of user-facing ingress gateways from heavy compute model execution workers using an asynchronous queuing architecture.

---

## Key Technical Features

* **Asynchronous Decoupled Ingestion**: FastAPI gateways accept incoming requests, validate payloads, and offload them to Redis list-based task queues.
* **Competing Consumers pattern**: Standard worker daemons written in Python/PyTorch pull tasks from queues, execute inference, and store results back to Redis.
* **Real-time Results Delivery**: The gateway uses Redis Pub/Sub to await and stream execution results back to the waiting client, maintaining standard synchronous HTTP request-response compatibility.
* **Fault Tolerance & Reliability**: Task state checking and queue re-queuing patterns are used to recover from worker crash failure modes without dropping messages.
* **Prometheus Metrics**: Exposes endpoints scraping system-level performance data including queue depth, API latency, and worker counts.

---

## Directory Structure

```text
├── .env.example            # Environment configurations blueprint
├── pyproject.toml          # Project metadata, Python 3.12 dependencies, Ruff/Pytest settings
├── PRD.md                  # Product Requirements Document
├── ARCHITECTURE.md         # High-level architecture and sequence flow specifications
├── DOCUMENTATION.md        # Technical API schema specifications and monitoring details
├── IMPLEMENTATION_PLAN.md  # Step-by-step roadmap for execution phases
│
├── gateway/                # FastAPI Ingress Gateway service
│   └── __init__.py
├── worker/                 # PyTorch inference worker daemon service
│   └── __init__.py
├── shared/                 # Common modules (Redis client, schemas, utilities)
│   └── __init__.py
├── docker/                 # Container files (Dockerfiles, compose setup)
│   └── README.md
├── docs/                   # Additional operational guides and documentation assets
│   └── README.md
└── tests/                  # Test suite directories (unit and integration tests)
    └── __init__.py
```

---

## Local Development & Setup

### Prerequisites
* Python 3.12+
* Docker & Docker Compose
* Redis Server (for local running outside docker)

### Installing Dependencies
1. Set up a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```

### Linting & Formatting (Ruff)
Ruff is configured in `pyproject.toml`. To check linting or formatting:
```bash
ruff check .
ruff format . --check
```

### Running Tests
To run unit and integration tests:
```bash
pytest
```
