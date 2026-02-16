# DepthFrame Processing Service

> Production-grade REST API for ingesting, storing, and serving image log data with depth-range querying and customizable colormap rendering.

[![CI — Build & Smoke Test](https://github.com/Rakesh-Raushan/DepthFrame-Processing-Service/actions/workflows/ci.yml/badge.svg)](https://github.com/Rakesh-Raushan/DepthFrame-Processing-Service/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97_Live_Demo-Swagger_UI-yellow.svg)](https://perpetualquest-depthframe-processing-service.hf.space/docs)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![uv](https://img.shields.io/badge/uv-managed-blueviolet.svg)](https://docs.astral.sh/uv/)

---

## What This Does

Ingest a CSV of image log data → Resize, validate, and store in SQLite → Query depth-range frames via REST API with domain-specific colormap rendering.

**Functional Features**:
- Ingestion pipeline: CSV → validation → anti-aliased resize (200→150px) → SQLite storage
- Depth-range frame retrieval with colormapped PNG/JPEG output
- 6 domain-appropriate colormaps (resistivity, conductivity, geological, high-contrast, gray, viridis)
- Raw grayscale data endpoint(Bonus) for ML pipeline consumers or any raw data consumers.
- Self-describing metadata endpoint (depth bounds, dimensions, processing provenance)

**Engineering Quality**:
- Repository pattern for swappable storage backend
- Pydantic schema validation with Swagger UI dropdowns
- Structured logging, health checks, dependency injection
- Unit + integration test suite
- Multi-stage Docker build with idempotent startup

---

## Quick Start

### Live Demo (No Setup Required)

> **[Try the API →](https://perpetualquest-depthframe-processing-service.hf.space/docs)**
>
> Interactive Swagger UI hosted on Hugging Face Spaces. No installation needed — test all endpoints directly in the browser.

### Docker (Recommended)

```bash
git clone <repo-url>
cd DepthFrame-Processing-Service

docker compose up --build
# First run ingests CSV → DB automatically
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Local Development

```bash
# Install dependencies (requires uv)
make dev

# Ingest the dataset
make ingest

# Start the API
make run
# → http://localhost:8000/docs
```

---

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/image` | Colormapped image frame by depth range (PNG/JPEG) |
| `GET` | `/api/v1/image/raw` | Raw grayscale pixel data for programmatic use |
| `GET` | `/api/v1/metadata` | Dataset info, processing provenance, available colormaps |
| `GET` | `/api/v1/colormaps` | Colormap list with descriptions |
| `GET` | `/health` | Service health check |

**Query parameters** for `/api/v1/image`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth_min` | float | required | Minimum depth (inclusive) |
| `depth_max` | float | required | Maximum depth (inclusive) |
| `colormap` | enum | `resistivity` | Colormap to apply (dropdown in Swagger) |
| `format` | enum | `png` | Output format: `png` or `jpeg` |

---


## API Usage Commands

```bash
curl "http://localhost:8000/api/v1/image?depth_min=9100&depth_max=9200&colormap=resistivity" \
  --output frame.png
```

```bash
curl "http://localhost:8000/api/v1/image/raw?depth_min=9100&depth_max=9200" \
  --output frame.bin
# Reshape using X-Frame-Rows and X-Frame-Width response headers
```

```bash
curl "http://localhost:8000/api/v1/metadata"
```

```bash
curl "http://localhost:8000/api/v1/colormaps"
```

```bash
curl "http://localhost:8000/health"
```

**Full API Reference**: Interactive OpenAPI docs at `/docs`


## Development Commands

```bash
make dev              # Install all dependencies (prod + dev)
make ingest           # Run ingestion pipeline (CSV → SQLite)
make run              # Start API server (uvicorn, hot reload)
make test             # Run all tests
make test-unit        # Run unit tests only
make test-integration # Run integration tests only
make lint             # Ruff + mypy
make docker-build     # Build Docker image
make docker-run       # Start with docker compose
make docker-stop      # Stop containers
make docker-clean     # Remove containers, volumes, images
make clean            # Remove caches, DB, pycache
```

---

## Project Structure

```
DepthFrame-Processing-Service/
├── src/depthframe_processing_service/
│   ├── api/                    # FastAPI application layer
│   │   ├── app.py              #   App factory + create_app()
│   │   ├── dependencies.py     #   Lifespan, DI providers
│   │   ├── routes.py           #   Endpoint handlers
│   │   └── schemas.py          #   Pydantic request/response models
│   ├── colormaps/              # Colormap engine
│   │   └── registry.py         #   Pluggable registry + apply()
│   ├── db/                     # Persistence layer
│   │   └── repository.py       #   SQLite repository pattern
│   ├── ingestion/              # ETL pipeline
│   │   └── pipeline.py         #   load → validate → resize → store
│   ├── config.py               # Pydantic BaseSettings (BHIS_* env vars)
│   └── ingest.py               # CLI entry point for ingestion
├── tests/
│   ├── unit/                   # Fast, isolated tests
│   ├── integration/            # Full HTTP + DB tests
│   └── conftest.py             # Shared fixtures
├── notebooks/
│   ├── 01_exploration.ipynb    # Data quality & visualization EDA
│   └── 02_resizing_analysis.ipynb  # Interpolation method evaluation
├── docs/                       # Engineering documentation
│   ├── 01_problem_understanding.md
│   └── 02_architecture_overview.md
├── data/                       # Runtime data (gitignored)
│   └── Challenge2.csv          # Input dataset
├── scripts/
│   └── entrypoint.sh           # Docker entrypoint (idempotent)
├── Dockerfile                  # Multi-stage build
├── docker-compose.yml          # Production deployment
├── pyproject.toml              # uv-managed dependencies
└── Makefile                    # Development ergonomics
```

## Technical Documentation

> For design rationale and engineering decisions, see [`/docs`](docs/).

1. **[Problem Understanding](docs/01_problem_understanding.md)**: Domain context (borehole image logs, line-scan acquisition), dataset forensics, interpolation method analysis (with frequency-domain validation), storage and colormap design decisions
2. **[Architecture Overview](docs/02_architecture_overview.md)**: System design, data flow, separation of concerns, dependency injection, technology choices

**Exploratory Analysis**:
- [`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb): Data quality, pixel distribution, colormap design
- [`notebooks/02_resizing_analysis.ipynb`](notebooks/02_resizing_analysis.ipynb): Multi-frame interpolation evaluation for different interpolation techniques

---

## Configuration

All settings configurable via environment variables (prefix `BHIS_`):

```bash
DPS_DATA_DIR=data                    # Data directory path
DPS_DB_PATH=data/image_store.db      # SQLite database path
DPS_CSV_FILENAME=Challenge2.csv      # Input CSV filename
DPS_TARGET_WIDTH=150                 # Resize target width
DPS_INTERPOLATION_METHOD=AREA        # cv2 interpolation method
DPS_DEFAULT_COLORMAP=resistivity     # Default colormap for API
DPS_HOST=0.0.0.0                     # API host
DPS_PORT=8000                        # API port
DPS_LOG_LEVEL=info                   # Logging level (uvicorn: info, debug, warning, error, critical)
```

---

## License

MIT
