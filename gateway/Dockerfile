# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies if any are needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only the files needed for the gateway to avoid invalidating cache
# when worker code changes.
COPY pyproject.toml README.md ./
COPY gateway/ ./gateway/
COPY shared/ ./shared/

# Install the package in editable mode with default dependencies
RUN pip install --no-cache-dir -e .

# Expose the API Gateway port
EXPOSE 8000

# Run Uvicorn server
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
