# Product Requirements Document (PRD)
## CoreAI Distributed Inference Platform

| Status | Draft |
| :--- | :--- |
| **Author** | Praveen Gupta |
| **Target Version** | v1.0.0 |
| **Date** | June 2, 2026 |

---

### 1. Business Problem & Opportunity

#### 1.1 Context
In the modern enterprise, Large Language Models (LLMs) and PyTorch-based deep learning models are critical drivers of product intelligence. However, serving deep learning inference at scale presents substantial challenges:
1. **Inefficient Compute Utilization**: Deep learning models require high-performance hardware (GPUs, specialized TPUs/CPUs). Provisioning dedicated compute resources for every service or microservice results in low utilization, resource fragmentation, and exorbitant cloud bills.
2. **High Latency and Congestion**: Synchronous model execution patterns block API gateway worker threads, causing cascading queueing delays, timeouts, and poor user experiences under bursty traffic.
3. **Complex Deployment Lifecycle**: AI engineers often struggle with standardizing model packaging, leading to inconsistencies between local training/validation environments and production cluster configurations.
4. **Lack of Operational Visibility**: Debugging distributed inference execution, tracing request bottlenecks, and identifying stale workers without centralized logging and metrics is highly error-prone.

#### 1.2 The CoreAI Solution
The **CoreAI Distributed Inference Platform** is a simplified, Azure OpenAI-style, API-compatible serving platform. It decouples the user-facing API gateways from the core model execution nodes using a highly efficient broker-based distributed system. By centralizing request ingestion, queuing tasks in an in-memory database (Redis), and dispatching jobs to specialized workers running PyTorch containerized services, this platform maximizes compute density, ensures high availability, and isolates workloads.

---

### 2. System Goals

* **Maximize Hardware Utilization**: Decouple the request intake from model execution to allow inference workers to run at 100% saturation during load peaks.
* **OpenAI API Compatibility**: Provide a developer-facing REST API mimicking the standard `/v1/chat/completions` interface to enable seamless integration with existing tools, SDKs (such as LangChain or Semantic Kernel), and codebases.
* **Asynchronous Execution Model**: Support both blocking synchronous requests (simulating real-time completions) and asynchronous submission workflows (submitting long-running batch jobs and polling for results).
* **Self-Healing Topology**: Gracefully recover from worker crashes, broker disconnects, and API gateway restarts without dropping client requests or leaving jobs in a permanently hung state.
* **Low Gateway Overhead**: Ensure that the platform's routing, queueing, and metadata retrieval layers add minimal latency (< 20ms at p99, excluding model evaluation time).

---

### 3. User Personas & Use Cases

#### 3.1 Personas
* **Application Developer**: Wants a simple, reliable, and predictable endpoint to send prompts and get responses without worrying about model loading, GPU memory management, or concurrency.
* **CoreAI Platform Operator**: Responsible for maintaining system uptime, scaling workers during traffic bursts, monitoring queue depths, and analyzing platform error rates.
* **Data Scientist/ML Engineer**: Wants to build and package custom PyTorch inference pipelines that run consistently across development, staging, and production environments.

#### 3.2 Key Use Cases
1. **Real-time Chat Completion**: An application requests a response for a user prompt and expects a synchronous response within a low latency window (e.g., chat apps, copilot integrations).
2. **Batch Inference Execution**: An offline processing pipeline submits 1,000 document prompts, receives a batch job ID, and polls for status updates, retrieving results incrementally as they finish.
3. **Dynamic Scaling**: The system detects an increase in queue length and allows operators to scale worker containers manually via Docker Compose to process the backlog faster.

---

### 4. Functional Requirements

| Req ID | Component | Requirement Description | Priority |
| :--- | :--- | :--- | :--- |
| **FR-01** | Gateway API | The system MUST expose a REST API matching the OpenAI schema for Chat Completions (`/v1/chat/completions`). | P0 |
| **FR-02** | Gateway API | The API gateway MUST support synchronous, blocking requests that wait for the worker to complete inference. | P0 |
| **FR-03** | Gateway API | The API gateway MUST support asynchronous job submission (`/v1/tasks`) returning a unique tracking ID. | P0 |
| **FR-04** | Gateway API | The system MUST expose a `/v1/models` endpoint returning a catalog of active and supported models. | P1 |
| **FR-05** | Auth | The API gateway MUST validate incoming requests using a header-based API key validation mechanism. | P1 |
| **FR-06** | Queuing | The platform MUST utilize a Redis broker to queue incoming inference requests. | P0 |
| **FR-07** | Queuing | The queue system MUST isolate tasks by model type (e.g., separate queues for `gpt-2` and `bert`). | P1 |
| **FR-08** | Worker | Workers MUST pre-load their respective PyTorch weights into memory (GPU or CPU) on startup to prevent cold-start latency per request. | P0 |
| **FR-09** | Worker | Workers MUST pull tasks from their assigned model queue, perform PyTorch inference, and write results back to Redis. | P0 |
| **FR-10** | Diagnostics | The platform MUST expose a `/health` endpoint on all services (Gateways, Workers) indicating status and connectivity. | P1 |
| **FR-11** | Diagnostics | The API Gateway MUST expose a Prometheus-compatible `/metrics` endpoint reporting request counts, latency histograms, queue depth, and active worker counts. | P1 |

---

### 5. Non-Functional Requirements

#### 5.1 Performance & Latency
* **Gateway Overhead**: The overhead added by the API Gateway routing, JSON serialization/deserialization, and Redis queue push/pop operations must be less than 20ms at p99.
* **Queuing Delay**: Under normal system loads (< 70% capacity), request queuing duration should not exceed 50ms.
* **Concurrency**: The API gateway must support at least 1,000 concurrent HTTP client connections using asynchronous I/O (FastAPI + ASGI).

#### 5.2 Scalability
* **Horizontal Scalability**: Adding more worker containers must scale processing throughput linearly (e.g., doubling workers doubles processing rate for queue backlogs).
* **Dynamic Batching**: (Moved to Future Enhancements).

#### 5.3 Reliability & Fault Tolerance
* **At-Least-Once Delivery**: The system must guarantee that once a task is successfully accepted by the gateway, it is executed at least once, even in the event of worker node failures.
* **Worker Crash Recovery**: If a worker pulls a task and crashes before finishing, the task must be automatically re-enqueued after a visibility timeout.
* **Backpressure Handling**: The gateway must return HTTP 429 (Too Many Requests) when the Redis queue length exceeds a configurable threshold (e.g., 5,000 pending tasks).

#### 5.4 Security
* **Access Control**: All public endpoints (except `/health` and `/metrics`) must require a valid API key (`Authorization: Bearer <key>`).
* **Input Validation**: All client prompts must be sanitized, and input token limits must be enforced via Pydantic model schemas to prevent Out of Memory (OOM) exploits.
* **Container Isolation**: Services must run as non-root users inside their Docker containers to restrict privileges.

---

### 6. Scope Boundaries & Constraints

#### 6.1 In Scope
* Standardized FastAPI server serving as the API Gateway.
* Redis server serving as the queue broker, metadata catalog database, and result pub/sub system.
* Python workers executing PyTorch models (configured by default for small models like HuggingFace GPT-2 or BERT).
* Reliable Queue pattern (`RPOPLPUSH` / `RPOP` with backup storage) to implement task persistence and worker failure recovery.
* Multi-container Docker Compose setup defining all networking, dependencies, and monitoring infrastructure (Prometheus).

#### 6.2 Out of Scope
* Multi-node GPU tensor parallelism (e.g., splitting a single model across multiple separate physical nodes via Megatron-LM/vLLM distributed layers).
* Advanced user account sign-ups, billing, and credit card processing.
* Support for dynamic runtime model downloading (workers only run models mounted or baked into their container images).
* Production-grade SSL/TLS certificates handling at the FastAPI layer (assumed to be terminated by an external ingress controller or load balancer like Nginx/Envoy).

#### 6.3 Future Enhancements
* **Dynamic Batching**: Stacking multiple concurrent request prompts into a single PyTorch tensor operation.
* **Grafana Dashboard Integration**: Full dashboard visualization (beyond basic Prometheus raw metrics endpoint).
* **Automatic Scaling**: K8s or Docker auto-scaling workers based on queue depth metrics.
* **OAuth2 / JWT Authentication**: Enhanced directory services authentication.
