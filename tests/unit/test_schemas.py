"""Unit tests for API request/response schemas."""

import pytest
from pydantic import ValidationError

from depthframe_processing_service.api.schemas import (
    HealthResponse,
    ImageQueryParams,
    MetadataResponse,
)


class TestImageQueryParams:
    def test_valid_params(self) -> None:
        params = ImageQueryParams(depth_min=9000.0, depth_max=9100.0)
        assert params.depth_min == 9000.0
        assert params.depth_max == 9100.0
        assert params.colormap == "resistivity"  # default
        assert params.format == "png"  # default

    def test_custom_colormap_and_format(self) -> None:
        params = ImageQueryParams(
            depth_min=9000.0,
            depth_max=9100.0,
            colormap="geological",
            format="jpeg",
        )
        assert params.colormap == "geological"
        assert params.format == "jpeg"

    def test_min_equals_max_fails(self) -> None:
        with pytest.raises(ValidationError, match="depth_min"):
            ImageQueryParams(depth_min=9000.0, depth_max=9000.0)

    def test_min_greater_than_max_fails(self) -> None:
        with pytest.raises(ValidationError, match="depth_min"):
            ImageQueryParams(depth_min=9200.0, depth_max=9100.0)

    def test_invalid_format_fails(self) -> None:
        with pytest.raises(ValidationError):
            ImageQueryParams(depth_min=9000.0, depth_max=9100.0, format="bmp")


class TestMetadataResponse:
    def test_serialization(self) -> None:
        resp = MetadataResponse(
            depth_min=9000.1,
            depth_max=9546.0,
            depth_step=0.1,
            row_count=5460,
            original_width=200,
            resized_width=150,
            interpolation_method="AREA",
            source_file="Challenge2.csv",
            available_colormaps=["gray", "resistivity"],
        )
        data = resp.model_dump()
        assert data["row_count"] == 5460
        assert len(data["available_colormaps"]) == 2


class TestHealthResponse:
    def test_healthy(self) -> None:
        resp = HealthResponse(status="healthy", db_connected=True, row_count=5460)
        assert resp.status == "healthy"

    def test_unhealthy(self) -> None:
        resp = HealthResponse(status="unhealthy", db_connected=False, row_count=0)
        assert resp.db_connected is False
