# 02 — Architecture Overview

> System design, data flow, and engineering decisions for the DepthFrame Processing Service.

---

## 1. System Architecture

### High-Level Components

The system follows a layered architecture with explicit boundaries between transport (API), processing (ingestion pipeline, colormap engine), and persistence (repository).

```
┌──────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                      │
│  ┌─────────────┐ ┌─────────────┐ ┌────────────┐ ┌────────┐ │
│  │  /image     │ │ /image/raw  │ │ /metadata  │ │ /health│ │
│  │  (frame)    │ │ (bytes)     │ │ /colormaps │ │        │ │
│  └──────┬──────┘ └──────┬──────┘ └─────┬──────┘ └───┬────┘ │
└─────────┼───────────────┼──────────────┼─────────────┼──────┘
          │               │              │             │
          ▼               ▼              ▼             ▼
┌──────────────────────────────────────────────────────────────┐
│                       Service Layer                          │
│  ┌─────────────────────┐      ┌─────────────────────┐       │
│  │ ColormapRegistry    │      │ ImageRepository     │       │
│  │ (apply colormap     │      │ (depth-range query  │       │
│  │  at response time)  │      │  over BLOB rows)    │       │
│  └─────────────────────┘      └─────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
          │                              │
          │     ┌────────────────────┐   │
          │     │ Ingestion Pipeline │   │
          │     │ (offline, one-time)│   │
          │     │ CSV → validate →   │   │
          │     │ resize → store     │   │
          │     └────────────────────┘   │
          ▼                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    Persistence Layer                          │
│  ┌─────────────────────┐      ┌─────────────────────┐       │
│  │  SQLite             │      │  Metadata Table     │       │
│  │  (image_scans,      │      │  (processing        │       │
│  │   depth → BLOB)     │      │   provenance)       │       │
│  └─────────────────────┘      └─────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
src/depthframe_processing_service/
├── api/                        # Transport layer
│   ├── app.py                  #   Application factory
│   ├── dependencies.py         #   Lifespan + DI providers
│   ├── routes.py               #   Endpoint handlers
│   └── schemas.py              #   Pydantic models
├── colormaps/                  # Colormap engine
│   └── registry.py             #   Pluggable registry
├── db/                         # Persistence
│   └── repository.py           #   SQLite repository pattern
├── ingestion/                  # ETL pipeline
│   └── pipeline.py             #   load → validate → resize → store
├── config.py                   # Centralized settings
└── ingest.py                   # CLI entry point
```

---

## 2. Design Principles

### Separation of Concerns

**API layer** (`api/routes.py`): HTTP request/response handling, parameter validation, error translation. Does NOT contain data processing logic or direct SQL.

**Service layer** (`colormaps/registry.py`, `db/repository.py`): Business logic — colormap application, depth-range querying, data reconstruction. Does NOT know about HTTP or FastAPI.

**Ingestion pipeline** (`ingestion/pipeline.py`): Offline ETL — CSV loading, validation, resize, storage. Each step is a pure function. Does NOT depend on the API layer.

**Why this matters**:
- Pipeline testable without spinning up the API server
- Colormap functions testable without a database
- Database can be swapped (SQLite → Postgres) by implementing the same repository interface
- API framework could be replaced (FastAPI → Flask) without touching data logic

### Dependency Injection

```python
# Repository and registry injected via FastAPI's Depends()
async def get_image_frame(
    repo: ImageRepository = Depends(get_repository),
    registry: ColormapRegistry = Depends(get_colormap_registry),
) -> Response:
```

**Benefits**: Tests override dependencies with test doubles (in-memory DB, fresh registry) — no monkeypatching needed.

### Lifespan-Managed Resources

The database connection opens once at startup and closes at shutdown via FastAPI's lifespan context manager — not per-request. SQLite in WAL mode supports concurrent reads from this single connection.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    repository = ImageRepository(settings.db_path)
    repository.connect()
    yield
    repository.close()
```

---

## 3. Data Flow

### Ingestion Flow (Offline, One-Time)

```
Challenge2.csv
    │
    ▼
load_csv()              Read CSV, 5461 rows × 201 cols
    │
    ▼
validate_and_clean()    Drop null rows (1 trailing), sort by depth,
    │                   drop duplicates, validate pixel range, fill NaNs
    ▼
resize_image()          float32 → cv2.INTER_AREA → 200→150px → uint8
    │
    ▼
store_to_database()     bulk_insert_scans() + metadata
    │
    ▼
SQLite DB               5460 rows × (depth REAL PK, pixel_data BLOB)
                        + metadata table (processing provenance)
```

Each step is a pure function returning data to the next — no shared state, independently testable.

### Request Flow (GET /api/v1/image)

```
Client request
  ?depth_min=9100&depth_max=9200&colormap=resistivity&format=png
    │
    ▼
Parameter validation    Pydantic: depth_min < depth_max, colormap exists,
    │                   format ∈ {png, jpeg}
    ▼
repo.query_depth_range()   SQL: WHERE depth >= 9100 AND depth <= 9200
    │                      Returns: List[ScanRow] ordered by depth
    ▼
np.array([...])         Assemble 2D frame from scan rows
    │
    ▼
registry.apply()        Normalize → colormap → RGB → PIL → PNG bytes
    │
    ▼
Response                image/png with X-Frame-* headers
                        (depth bounds, row count, width, colormap name)
```

**Key decision**: Colormap applied at response time, not at storage time. Stored data is always grayscale uint8. This decouples interpretation from persistence — different consumers can request different colormaps for the same data without re-ingestion.

---

## 4. Technology Choices

### FastAPI

Chosen over Flask/Django for: automatic OpenAPI docs (`/docs`), Pydantic validation, `str, Enum` rendering as Swagger dropdowns, async support (future-proof), and dependency injection system that simplifies testing.

### SQLite

Correct for ~800KB of data with range-query access pattern. WAL mode for concurrent reads. Repository pattern abstracts the dialect — switching to Postgres requires changing one connection string, not the API or pipeline code.

### uv

Dependency management via `uv` with lockfile (`uv.lock`) for reproducible builds. `pyproject.toml` as the single source of truth for dependencies.

### OpenCV (headless)

`opencv-python-headless` for interpolation — no GUI dependencies, smaller Docker image. `cv2.INTER_AREA` chosen via rigorous multi-frame evaluation (see `notebooks/02_resizing_analysis.ipynb`).

---

## 5. Containerization

### Docker Strategy

**Multi-stage build**:
- Stage 1 (builder): `python:3.13-slim` + uv → install dependencies from lockfile
- Stage 2 (runtime): `python:3.13-slim` → copy `.venv` and source only

**Security**: Non-root user (`appuser:1000`), minimal installed packages.

**Idempotent entrypoint** (`scripts/entrypoint.sh`):
- Checks if DB file exists
- If not: validates CSV is mounted, runs ingestion
- If yes: skips straight to API startup
- `exec uvicorn` as PID 1 for proper signal handling

**Persistence**: Named Docker volume for SQLite DB survives container recreation. CSV bind-mounted read-only (`:ro`) since it's only needed for the one-time ingestion.

---

## 6. Testing Strategy

### Test Organization

```
tests/
├── conftest.py           # Shared fixtures (synthetic CSV, tmp DB)
├── unit/                 # Fast, no I/O
│   ├── test_pipeline.py  #   validate_and_clean, resize_image
│   ├── test_repository.py#   CRUD, range queries, roundtrip
│   ├── test_colormaps.py #   Registry, apply, PNG/JPEG output
│   └── test_schemas.py   #   Pydantic validation logic
└── integration/
    ├── test_ingestion.py #   Full pipeline: CSV → DB
    └── test_api.py       #   HTTP cycle via httpx + test DB
```

### Key Testing Patterns

**Pure functions**: Pipeline steps (validate, resize) tested with synthetic data — no DB, no API, no filesystem.

**Repository**: Tested with `tmp_path` fixture — real SQLite, temporary directory, cleaned up automatically.

**API integration**: `httpx.AsyncClient` with `ASGITransport` against the real FastAPI app. Dependencies overridden to point to a test DB populated with synthetic data — tests the full HTTP cycle without touching production data.

---

## Summary

| Concern | Decision | Rationale |
|---------|----------|-----------|
| Framework | FastAPI | OpenAPI docs, Pydantic, DI, async-ready |
| Database | SQLite + repository pattern | Right-sized for ~800KB; swappable |
| Storage format | uint8 BLOB per row | 8× efficient vs float columns |
| Colormap | Apply at read time | Decouples interpretation from storage |
| Resize | cv2.INTER_AREA in float32 | Validated via FFT + multi-frame analysis |
| Config | Pydantic BaseSettings | Typed, validated, env-overridable |
| Container | Multi-stage, non-root, idempotent | Production-ready, secure, repeatable |
| Testing | Unit + integration, DI overrides | Fast feedback + full-cycle confidence |
