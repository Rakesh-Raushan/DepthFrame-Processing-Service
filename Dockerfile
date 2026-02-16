# ── Stage 1: Build ──
# Install dependencies in a full image with build tools
FROM python:3.13-slim AS builder

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install production dependencies only (no dev extras)
# --frozen ensures uv.lock is respected exactly
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ src/

# Install the project itself
RUN uv sync --frozen --no-dev


# ── Stage 2: Runtime ──
# Slim image with only what's needed to run
FROM python:3.13-slim AS runtime

# Security: non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/src /app/src

# Copy entrypoint
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create data directory
RUN mkdir -p /app/data && chown -R appuser:appuser /app

# Bake the dataset CSV into the image for self-contained evaluation.
# This adds ~4.4MB to the image but means `docker compose up --build`
# works without any manual file placement. In a production deployment,
# the CSV would be mounted at runtime instead.
COPY data/Challenge2.csv /app/data/Challenge2.csv

# Activate the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# Configuration defaults (override via docker-compose or -e)
ENV DPS_DATA_DIR="/app/data"
ENV DPS_DB_PATH="/app/data/image_store.db"
ENV DPS_HOST="0.0.0.0"
ENV DPS_PORT="8000"
ENV DPS_LOG_LEVEL="info"

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check for orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
