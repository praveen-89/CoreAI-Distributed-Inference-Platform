# Docker Folder Configuration

This directory contains container configurations and orchestration specifications for local development and staging deployments.

## Directory Contents
* **`api-gateway.Dockerfile`**: Multi-stage Docker build for the FastAPI ingress node.
* **`worker.Dockerfile`**: PyTorch optimized build for inference nodes.
* **`docker-compose.yml`**: Multi-service local coordinator orchestrating Redis, Gateway, Worker, and Prometheus instances.
