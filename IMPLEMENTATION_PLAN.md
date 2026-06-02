# Implementation Plan & Roadmap
## CoreAI Distributed Inference Platform

This document outlines the step-by-step roadmap for building and executing the CoreAI Distributed Inference Platform.

---

### Phase 1: Repository Setup
Setup the base codebase structure, Python virtual environments, and initial dependency configuration.
* **Deliverables**: Python virtual environment configuration, root-level dependency files, and basic project folder structure.
* **Files to create**:
  * `[NEW] requirements.txt` (or `pyproject.toml`)
  * `[NEW] .gitignore`
  * Directory structure: `/gateway`, `/worker`, `/shared`, `/tests`
* **Dependencies**: Python 3.10+, pip
* **Testing strategy**: Validate local directory structure and ensure dependencies can be installed cleanly.
* **Estimated complexity**: Low (1/5)

---

### Phase 2: API Gateway
Develop the user-facing HTTP ingress microservice using FastAPI.
* **Deliverables**: Functional REST API running on FastAPI with `/v1/chat/completions`, `/v1/models`, and `/health` endpoints. Basic API key authentication middleware.
* **Files to create**:
  * `[NEW] gateway/main.py` (FastAPI entrypoint, routes, middleware)
  * `[NEW] gateway/config.py` (Environment variable configuration)
  * `[NEW] gateway/schemas.py` (Pydantic schemas for request/response validation)
* **Dependencies**: `fastapi`, `uvicorn`, `pydantic`, `python-dotenv`
* **Testing strategy**:
  * Unit tests mock request payloads using FastAPI `TestClient`.
  * Manual API testing using `curl` or Postman to verify route handling and authentication status codes.
* **Estimated complexity**: Medium (2.5/5)

---

### Phase 3: Redis Queue Layer
Configure connection logic to Redis and integrate request enqueuing from the Gateway.
* **Deliverables**: Connection pool to Redis, client helper functions for pushing requests into a Redis list (queue), and registering basic task states in Redis hashes.
* **Files to create**:
  * `[NEW] shared/redis_client.py` (Shared helper class for Redis operations)
  * `[NEW] gateway/services/queue_service.py` (Functions to push tasks to Redis)
* **Dependencies**: `redis` (aioredis for async operations)
* **Testing strategy**:
  * Mock Redis connections in tests.
  * Integration tests with a local Redis instance: push a task and assert it appears in the list with expected serialization.
* **Estimated complexity**: Medium (2.5/5)

---

### Phase 4: Worker Service
Build the PyTorch inference daemon that processes tasks.
* **Deliverables**: Long-running Python process that starts up, loads a small PyTorch model (e.g. HuggingFace GPT-2 or a basic text generator) to memory, blocks on the Redis list queue, and evaluates inputs.
* **Files to create**:
  * `[NEW] worker/main.py` (Worker daemon main execution loop)
  * `[NEW] worker/model_runner.py` (PyTorch model loading and inference handler)
  * `[NEW] worker/config.py` (Worker environment configuration)
* **Dependencies**: `torch`, `transformers` (for loading GPT-2), `redis`
* **Testing strategy**:
  * Run model runner tests locally to ensure successful model loading and inference.
  * Push fake tasks to Redis and check if worker pulls, processes, and writes results to stdout.
* **Estimated complexity**: High (4/5)

---

### Phase 5: Result Retrieval
Connect worker output back to the API Gateway to support blocking synchronous endpoints.
* **Deliverables**: Redis Pub/Sub communication channel and result cache. Gateway subscribes to request-specific channels to fetch completed worker output.
* **Files to create**:
  * `[NEW] gateway/services/result_service.py` (Wait-on-result wrapper using Redis Pub/Sub)
* **Dependencies**: `redis`
* **Testing strategy**:
  * Integration test: Submit request to `/v1/chat/completions`, simulate worker writing response to Redis + publishing, check that HTTP client receives response.
* **Estimated complexity**: High (3.5/5)

---

### Phase 6: Worker Registry
Establish worker tracking and status management for the cluster.
* **Deliverables**: Heartbeat system where workers regularly check in with Redis, and gateway reads active workers to dynamically serve active models.
* **Files to create**:
  * `[NEW] worker/heartbeat.py` (Periodic thread in worker sending heatbeats)
  * `[NEW] gateway/services/registry_service.py` (Gateway check for healthy workers)
* **Dependencies**: `redis`
* **Testing strategy**:
  * Spin up worker, verify heartbeat key exists in Redis with TTL.
  * Verify that stopping the worker causes the registry to mark it inactive after TTL.
* **Estimated complexity**: Medium (3/5)

---

### Phase 7: Monitoring
Implement system performance metrics and observability exporters.
* **Deliverables**: Prometheus metrics client integration in the gateway, reporting request durations, queue sizes, and error rates. Basic Prometheus configuration.
* **Files to create**:
  * `[NEW] gateway/monitoring.py` (Prometheus metrics registration)
  * `[NEW] monitoring/prometheus.yml` (Prometheus configuration file)
* **Dependencies**: `prometheus-client`
* **Testing strategy**:
  * Query `/metrics` endpoint on Gateway and verify Prometheus format output.
  * Spin up Prometheus container and verify scraping succeeds.
* **Estimated complexity**: Medium (2.5/5)

---

### Phase 8: Docker Deployment
Orchestrate all services into a unified multi-container local stack.
* **Deliverables**: Dockerfiles for gateway and worker, and a `docker-compose.yml` to launch gateway, redis, workers, and monitoring services.
* **Files to create**:
  * `[NEW] gateway/Dockerfile`
  * `[NEW] worker/Dockerfile`
  * `[NEW] docker-compose.yml`
* **Dependencies**: Docker, Docker Compose
* **Testing strategy**:
  * Run `docker-compose up` and execute a complete end-to-end completions request flow.
* **Estimated complexity**: Medium (3/5)

---

### Phase 9: Advanced Reliability Features
Build fault tolerance features to protect against worker crashes and poison pills.
* **Deliverables**: Reliable queue pop pattern using a backup processing queue (`RPOPLPUSH`), worker crash detection loop, and task retry/DLQ processing.
* **Files to create**:
  * `[NEW] shared/reliability.py` (Orchestration for dead letter queueing and retries)
* **Dependencies**: `redis`
* **Testing strategy**:
  * Start worker, force kill the process mid-inference, verify that the task is detected by the gateway/reaper and re-enqueued.
* **Estimated complexity**: High (4/5)

---

## Future Enhancements
The following features are moved out of the core MVP implementation to reduce initial complexity:
1. **Dynamic Batching**: Collecting individual requests in worker memory before invoking the PyTorch forward pass. (Deferred to post-v1.0.0).
2. **Grafana Dashboard Integration**: Full dashboard creation for system visualization (Basic Prometheus `/metrics` endpoint is kept in Phase 7).
3. **Advanced Rate Limiting**: Token-bucket rate limiting based on client IP or API keys (Basic backpressure protection based on queue depth is kept).
4. **OAuth2/JWT Auth**: Implementing directory service authentication (Simple API token key validation is kept in Phase 2).
