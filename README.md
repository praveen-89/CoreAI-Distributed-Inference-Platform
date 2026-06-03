# Distributed Inference Platform

This is a distributed inference platform I built to serve PyTorch models with high performance. It uses FastAPI for the user-facing gateway and Redis for asynchronous task queuing, which completely decouples the heavy inference workloads from the API ingestion layer.

## How to Run This Project

Follow these steps to run the platform locally using Docker.

### 1. Prerequisites
* Docker and Docker Compose installed on your system.
* Python 3.12 (optional, if you want to run tests locally outside Docker).

### 2. Start the Platform
To bring up the entire stack (API Gateway, Redis, Inference Worker, and Prometheus for monitoring), run:
```bash
docker compose up --build -d
```
This will start:
* **Redis** on port 6379 (Task Queue and Pub/Sub).
* **API Gateway** on port 8000.
* **Worker Daemon** (Runs in the background, executing tasks).
* **Prometheus** on port 9090 (Metrics scraping).

### 3. Architecture & Workflow

Here is the high-level architecture of how requests are processed:

```mermaid
graph TD
    Client[Client] -->|HTTP POST| Gateway[FastAPI Gateway]
    Gateway -->|Enqueue Task| Redis[Redis Queue]
    Redis -->|Dequeue Task| Worker[PyTorch Worker]
    Worker -->|Execute Model| Worker
    Worker -->|Save Result| Redis
    Gateway -->|Poll/Subscribe| Redis
    Gateway -->|Return HTTP Response| Client
    Prometheus[Prometheus] -->|Scrape /metrics| Gateway
```

And the sequence of operations for an inference request:

```mermaid
sequenceDiagram
    participant C as Client
    participant G as FastAPI Gateway
    participant R as Redis
    participant W as PyTorch Worker
    
    C->>G: POST /v1/chat/completions
    G->>G: Validate API Key
    G->>R: Push task to queue
    G->>R: Subscribe to result
    R-->>W: Worker polls queue
    W->>W: Run ML Inference
    W->>R: Publish result
    R-->>G: Receive result
    G-->>C: HTTP 200 OK (Response)
```

### 4. Test the API

The API is secured with a default API key. In PowerShell, `curl` is an alias for `Invoke-WebRequest`. You can query the available models like this:
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/v1/models" -Headers @{Authorization="Bearer sk-coreai-development-key-2026"}
```

You can submit a chat completion request to the gateway using this command:
```powershell
Invoke-WebRequest -Method POST -Uri "http://localhost:8000/v1/chat/completions" `
  -Headers @{
      "Content-Type" = "application/json"
      "Authorization" = "Bearer sk-coreai-development-key-2026"
  } `
  -Body '{"model": "gpt-2", "messages": [{"role": "user", "content": "Hello, how are you?"}]}'
```

### 5. Viewing Metrics
I've wired up Prometheus to scrape metrics from the gateway. You can view them by navigating to:
[http://localhost:9090](http://localhost:9090)

### 5. Shutting Down
When you're done, you can stop all services and remove the containers:
```bash
docker compose down
```

## Running Tests Locally
If you want to run the test suite:
1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Run Pytest:
   ```bash
   pytest
   ```

---
*Created by Praveen Gupta*
