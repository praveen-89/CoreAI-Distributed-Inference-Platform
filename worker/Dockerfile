# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# First install torch separately to leverage Docker layer caching 
# since it's massive and rarely changes
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy project files
COPY pyproject.toml README.md ./
COPY worker/ ./worker/
COPY shared/ ./shared/

# Install the package with worker dependencies (which includes transformers)
RUN pip install --no-cache-dir -e .[worker]

# Run the worker daemon
CMD ["python", "-m", "worker.main"]
