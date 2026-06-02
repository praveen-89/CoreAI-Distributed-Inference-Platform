# Technical Documentation & Specifications
## CoreAI Distributed Inference Platform

This document contains the API specifications, monitoring setup, deployment configs, and interview discussion points for the CoreAI Distributed Inference Platform.

---

### 1. API Specifications

All endpoints require the HTTP header `Authorization: Bearer <COREAI_API_KEY>`.

#### 1.1 Chat Completions (Synchronous)
* **Endpoint**: `POST /v1/chat/completions`
* **Content-Type**: `application/json`
* **Request Payload**:
```json
{
  "model": "gpt-2",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```
* **Success Response (200 OK)**:
```json
{
  "id": "chatcmpl-923f5b72e1ab",
  "object": "chat.completion",
  "created": 1780492800,
  "model": "gpt-2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 7,
    "total_tokens": 14
  }
}
```
* **Error Responses**:
  * `400 Bad Request`: Validation errors (e.g. unsupported model, negative temperature).
  * `401 Unauthorized`: Missing or invalid API Key.
  * `429 Too Many Requests`: Queue backpressure threshold hit or rate limit exceeded.
  * `504 Gateway Timeout`: The inference worker took longer than the configured timeout (e.g. 30s) to return the result.

#### 1.2 Submit Inference Task (Asynchronous)
* **Endpoint**: `POST /v1/tasks`
* **Request Payload**: Same as Chat Completions payload.
* **Success Response (202 Accepted)**:
```json
{
  "task_id": "task-bf98031d-b8d5-45d6-b072-a083b4822f36",
  "status": "PENDING",
  "model": "gpt-2"
}
```

#### 1.3 Query Task Status (Asynchronous Poll)
* **Endpoint**: `GET /v1/tasks/{task_id}`
* **Success Response (200 OK - Processing)**:
```json
{
  "task_id": "task-bf98031d-b8d5-45d6-b072-a083b4822f36",
  "status": "PROCESSING",
  "result": null
}
```
* **Success Response (200 OK - Completed)**:
```json
{
  "task_id": "task-bf98031d-b8d5-45d6-b072-a083b4822f36",
  "status": "COMPLETED",
  "result": {
    "id": "chatcmpl-923f5b72e1ab",
    "object": "chat.completion",
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "The capital of France is Paris."
        },
        "finish_reason": "stop"
      }
    ]
  }
}
```

#### 1.4 List Models
* **Endpoint**: `GET /v1/models`
* **Success Response (200 OK)**:
```json
{
  "data": [
    {
      "id": "gpt-2",
      "object": "model",
      "owned_by": "coreai-platform",
      "status": "active"
    },
    {
      "id": "bert-base-uncased",
      "object": "model",
      "owned_by": "coreai-platform",
      "status": "active"
    }
  ]
}
```

---

### 2. Monitoring & Metrics Strategy

#### 2.1 Prometheus Metrics Schema
The API Gateway exposes metrics on `/metrics` endpoint. The platform scrapes the following custom metrics:

| Metric Name | Type | Description |
| :--- | :--- | :--- |
| `coreai_request_total` | Counter | Total count of requests received by the gateway (labeled by `endpoint`, `model`, `status_code`). |
| `coreai_request_latency_seconds` | Histogram | Request latency seen by the client (excluding network round trip). |
| `coreai_queue_depth` | Gauge | Number of tasks currently waiting in the Redis queue per model (labeled by `model`). |
| `coreai_inference_time_seconds` | Histogram | Execution time of the PyTorch model inside the worker. |
| `coreai_active_workers` | Gauge | Count of active inference workers currently connected and heartbeat-active. |

#### 2.2 Grafana Dashboard Setup (Future Enhancement)
This feature is moved to Future Enhancements. A pre-configured Grafana dashboard will be added in subsequent iterations to visualize metrics scraped by Prometheus, showing:
* **Core KPIs**: Total requests, average latency, worker failure rate, current queue depth.
* **Queue Health**: Time series showing `coreai_queue_depth` to alert operators if tasks are stacking up.
* **Worker Saturation**: Gauge of active workers vs idle workers.
* **Model Latency breakdown**: Percentiles (p50, p90, p99) of `coreai_inference_time_seconds`.

---

### 3. Deployment Architecture

The platform runs containerized components orchestrated via Docker Compose.

```yaml
version: '3.8'

services:
  redis:
    image: redis:7.2-alpine
    container_name: coreai-redis
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes --requirepass coreai_secure_pass
    volumes:
      - redis_data:/data

  api-gateway:
    build:
      context: ./gateway
      dockerfile: Dockerfile
    container_name: coreai-api-gateway
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=coreai_secure_pass
      - COREAI_API_KEY=sk-coreai-development-key-2026
    depends_on:
      - redis

  worker-gpt2:
    build:
      context: ./worker
      dockerfile: Dockerfile
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=coreai_secure_pass
      - MODEL_NAME=gpt-2
      - DEVICE=cpu # Can be switched to cuda if GPU is available
    depends_on:
      - redis

  prometheus:
    image: prom/prometheus:latest
    container_name: coreai-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

volumes:
  redis_data:
```

---

### 4. Technical Interview Discussion Points & Deep Dives

During system design and technical portfolio presentations, candidates can leverage the following talking points based on this architecture:

#### 4.1 Redis vs. Kafka vs. RabbitMQ
* **Redis**: Chosen for its **in-memory latency characteristics** (< 1ms pushes and pops). Since LLM inference takes hundreds of milliseconds or seconds, message broker overhead must be minimized. Redis also combines lists (for queuing), hashes (for async task status caching), and pub/sub in a single fast engine, eliminating infrastructure sprawl.
* **Kafka**: Designed for high-throughput stream processing and long retention. Using Kafka for real-time request-response loops introduces unnecessary replication and partition tuning overhead.
* **RabbitMQ**: Excellent AMQP broker but heavier to maintain than Redis. Lacks built-in high-performance cache storage for queryable task statuses.

#### 4.2 Handling PyTorch Out-of-Memory (OOM) Errors
* **The Problem**: Running variable-sized input sequences can cause unexpected GPU memory spikes.
* **Prevention**:
  1. Strict input sequence token length validation at the gateway level.
  2. Running worker model execution inside a context manager that disables gradients: `with torch.no_grad():`.
  3. Pre-allocating CUDA memory caches or cleaning up via `torch.cuda.empty_cache()` if a worker encounters a batch memory failure.
* **Self-Healing**: If a worker runs out of memory, Python will crash with a runtime CUDA OOM error. The docker daemon is configured to automatically restart the container (`restart: on-failure`), and the Redis reliable queue pattern ensures the aborted task is re-queued and retried (possibly with a smaller batch limit or on a larger worker instance).

#### 4.3 Dynamic Batching Trade-offs (Future Enhancement)
* **The Concept**: Instead of processing 1 request at a time, a worker collects tasks over a tiny delay window (e.g. 5ms) and executes them together in a single tensor operation.
* **Trade-off Matrix**:
  * **No Batching**: Lowest latency for single requests, but low GPU utilization. (Current MVP state).
  * **Static Batching**: High throughput, but requests must wait until the batch is full, hurting latency.
  * **Dynamic Batching**: Balances both. If traffic is low, it runs with batch size 1 immediately. Under high concurrency, it naturally batches requests up to a configured threshold, maximizing throughput. (Future enhancement).
