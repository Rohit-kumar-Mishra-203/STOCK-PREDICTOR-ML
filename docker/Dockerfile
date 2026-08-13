# Why python:3.10-slim: matches your local Python version exactly
# (avoids "works locally, breaks in Docker" version mismatches), and
# "slim" strips unnecessary OS packages to keep the image smaller than
# the full python:3.10 image, without needing Alpine's more fragile
# compatibility tradeoffs for scientific packages like torch/numpy.
FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffers stdout - both
# matter in containers: .pyc files are pointless in an ephemeral
# container, and unbuffered stdout means logs appear immediately in
# `docker logs`, not stuck in a buffer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed to build some Python packages
# (e.g. some torch/numpy wheels need basic build tools) - kept minimal
# deliberately, only what's actually required.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements.txt first - this is the key layer-caching trick.
# Docker caches each layer; as long as requirements.txt doesn't change,
# this expensive pip install layer is reused across builds even when
# your actual source code changes constantly during development.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the actual source code - this layer invalidates on every
# code change, but that's fine since it's fast (no reinstall needed).
COPY src/ ./src/
COPY config/ ./config/

# Create a non-root user to run the app - running containers as root
# is a real, avoidable security risk in production. This is a small
# addition that signals security awareness.
RUN useradd --create-home appuser && \
    mkdir -p /app/data /app/models_store /app/logs && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck lets Docker (and orchestrators like Kubernetes/ECS) know
# if the container is actually serving traffic, not just running -
# ties directly into the /health endpoint we built in main.py.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

# Default command runs the API. docker-compose overrides this for the
# dashboard service (see docker-compose.yml).
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]