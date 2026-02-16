# =============================================================================
# DepthFrame Processing Service - Development Commands
# =============================================================================
.PHONY: help install dev test lint format docker-build docker-up docker-down clean

# Default target
help:
	@echo "DepthFrame Processing Service - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install production dependencies"
	@echo "  make dev          Install all dependencies (including dev tools)"
	@echo ""
	@echo "Development:"
	@echo "  make run          Run the API locally (uvicorn)"
	@echo "  make test         Run tests with pytest"
	@echo "  make test-cov     Run tests with coverage report"
	@echo "  make lint         Run linter (ruff)"
	@echo "  make format       Format code (ruff)"
	@echo "  make typecheck    Run type checker (mypy)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-up    Start containers (docker-compose)"
	@echo "  make docker-down  Stop containers"
	@echo "  make docker-logs  View container logs"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Remove cache files and build artifacts"

# =============================================================================
# Setup
# =============================================================================
install:
	uv sync --no-dev

dev:
	uv sync --all-extras
	pre-commit install

# =============================================================================
# Development
# =============================================================================
run:
	uv run uvicorn depthframe_processing_service.api.app:app --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check src/ tests/ --fix

typecheck:
	uv run mypy src/depthframe_processing_service


# =============================================================================
# Ingestion
# =============================================================================
ingest:
	uv run python -m depthframe_processing_service.ingest


# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t depthframe-processing-service .

docker-run:
	docker compose up --build

docker-stop:
	docker compose down

docker-clean:
	docker compose down -v --rmi local
	docker image prune -f

# =============================================================================
# Cleanup
# =============================================================================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
