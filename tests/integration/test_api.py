"""Integration tests for the API endpoints.

Uses httpx.AsyncClient against the real FastAPI app with a
temporary SQLite database populated with synthetic data.
Tests the full HTTP request/response cycle including:
- Parameter validation
- Database queries
- Colormap application
- Content-type negotiation
- Error responses
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from depthframe_processing_service.api.app import create_app
from depthframe_processing_service.api.dependencies import (
    get_colormap_registry,
    get_repository,
)
from depthframe_processing_service.colormaps.registry import ColormapRegistry
from depthframe_processing_service.db.repository import ImageRepository


@pytest.fixture()
def populated_repo(tmp_path: Path) -> ImageRepository:
    """Create a temporary DB with synthetic scan data."""
    db_path = tmp_path / "test.db"
    repo = ImageRepository(db_path)
    repo.connect()

    rng = np.random.default_rng(42)
    n_rows = 200
    depths = np.arange(9000.0, 9000.0 + n_rows * 0.1, 0.1)[:n_rows]
    pixels = rng.integers(30, 230, size=(n_rows, 150), dtype=np.uint8)

    repo.bulk_insert_scans(depths, pixels)
    repo.set_metadata("depth_min", str(depths.min()))
    repo.set_metadata("depth_max", str(depths.max()))
    repo.set_metadata("depth_step", "0.1")
    repo.set_metadata("row_count", str(n_rows))
    repo.set_metadata("original_width", "200")
    repo.set_metadata("resized_width", "150")
    repo.set_metadata("interpolation_method", "AREA")
    repo.set_metadata("source_file", "test_data.csv")

    yield repo
    repo.close()


@pytest.fixture()
def app_with_test_db(populated_repo: ImageRepository) -> create_app:
    """Create a FastAPI app with dependency overrides pointing to test DB."""
    app = create_app()

    # Override dependencies to use test DB instead of lifespan-managed one
    app.dependency_overrides[get_repository] = lambda: populated_repo
    app.dependency_overrides[get_colormap_registry] = lambda: ColormapRegistry()

    return app


@pytest_asyncio.fixture()
async def client(app_with_test_db) -> AsyncClient:
    """Async HTTP client for the test app."""
    transport = ASGITransport(app=app_with_test_db)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── Health ──


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_ok(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["db_connected"] is True
        assert data["row_count"] == 200


# ── Metadata ──


class TestMetadataEndpoint:
    @pytest.mark.asyncio
    async def test_metadata_returns_all_fields(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert data["depth_min"] == 9000.0
        assert data["resized_width"] == 150
        assert data["interpolation_method"] == "AREA"
        assert "resistivity" in data["available_colormaps"]

    @pytest.mark.asyncio
    async def test_metadata_includes_colormaps(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/metadata")
        data = resp.json()
        assert len(data["available_colormaps"]) >= 5  # we register 6 defaults


# ── Colormaps ──


class TestColormapsEndpoint:
    @pytest.mark.asyncio
    async def test_list_colormaps(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/colormaps")
        assert resp.status_code == 200
        data = resp.json()
        assert "colormaps" in data
        assert "default" in data
        names = [c["name"] for c in data["colormaps"]]
        assert "resistivity" in names
        assert "gray" in names

    @pytest.mark.asyncio
    async def test_colormaps_have_descriptions(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/colormaps")
        data = resp.json()
        for cmap in data["colormaps"]:
            assert "description" in cmap
            assert len(cmap["description"]) > 0


# ── Image frame ──


class TestImageEndpoint:
    @pytest.mark.asyncio
    async def test_get_png_image(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/api/v1/image",
            params={"depth_min": 9000.0, "depth_max": 9005.0},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        # Verify PNG magic bytes
        assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"
        # Check custom headers
        assert "x-frame-rows" in resp.headers
        assert "x-colormap" in resp.headers

    @pytest.mark.asyncio
    async def test_get_jpeg_image(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/api/v1/image",
            params={
                "depth_min": 9000.0,
                "depth_max": 9005.0,
                "format": "jpeg",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content[:2] == b"\xff\xd8"  # JPEG magic

    @pytest.mark.asyncio
    async def test_custom_colormap(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/api/v1/image",
            params={
                "depth_min": 9000.0,
                "depth_max": 9005.0,
                "colormap": "geological",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["x-colormap"] == "geological"

    @pytest.mark.asyncio
    async def test_invalid_depth_range(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/api/v1/image",
            params={"depth_min": 9100.0, "depth_max": 9000.0},
        )
        assert resp.status_code == 400
        assert "depth_min" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_unknown_colormap(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/api/v1/image",
            params={
                "depth_min": 9000.0,
                "depth_max": 9005.0,
                "colormap": "nonexistent",
            },
        )
        assert resp.status_code == 422  # FastAPI returns 422 for invalid enum values

    @pytest.mark.asyncio
    async def test_out_of_range_depth(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/api/v1/image",
            params={"depth_min": 1000.0, "depth_max": 2000.0},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_frame_dimensions_in_headers(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/api/v1/image",
            params={"depth_min": 9000.0, "depth_max": 9001.0},
        )
        assert resp.status_code == 200
        assert resp.headers["x-frame-width"] == "150"
        assert int(resp.headers["x-frame-rows"]) > 0


# ── Raw image ──


class TestRawImageEndpoint:
    @pytest.mark.asyncio
    async def test_get_raw_data(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/api/v1/image/raw",
            params={"depth_min": 9000.0, "depth_max": 9001.0},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        assert resp.headers["x-dtype"] == "uint8"

        # Reconstruct and verify shape
        rows = int(resp.headers["x-frame-rows"])
        width = int(resp.headers["x-frame-width"])
        arr = np.frombuffer(resp.content, dtype=np.uint8).reshape(rows, width)
        assert arr.shape[1] == 150

    @pytest.mark.asyncio
    async def test_raw_invalid_range(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/api/v1/image/raw",
            params={"depth_min": 9100.0, "depth_max": 9000.0},
        )
        assert resp.status_code == 400
