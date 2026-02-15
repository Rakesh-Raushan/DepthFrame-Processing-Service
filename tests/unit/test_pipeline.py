"""Unit tests for the ingestion pipeline functions.

Tests individual pipeline steps (load, validate, resize) in isolation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from depthframe_processing_service.ingestion.pipeline import (
    INTERPOLATION_METHODS,
    ValidationReport,
    load_csv,
    resize_image,
    validate_and_clean,
)


class TestLoadCsv:
    """Tests for CSV loading functionality."""

    def test_load_csv_success(
        self, temp_csv_file: Path, sample_dataframe: pd.DataFrame
    ) -> None:
        """load_csv should successfully load a valid CSV."""
        df = load_csv(temp_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_dataframe)
        assert "depth" in df.columns

    def test_load_csv_file_not_found(self, tmp_path: Path) -> None:
        """load_csv should raise FileNotFoundError for missing files."""
        missing_path = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_csv(missing_path)

    def test_load_csv_preserves_columns(
        self, temp_csv_file: Path, sample_dataframe: pd.DataFrame
    ) -> None:
        """load_csv should preserve all columns from the CSV."""
        df = load_csv(temp_csv_file)

        assert set(df.columns) == set(sample_dataframe.columns)


class TestValidateAndClean:
    """Tests for data validation and cleaning."""

    def test_validate_and_clean_returns_tuple(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """validate_and_clean should return (depths, pixels, report)."""
        depths, pixels, report = validate_and_clean(sample_dataframe)

        assert isinstance(depths, np.ndarray)
        assert isinstance(pixels, np.ndarray)
        assert isinstance(report, ValidationReport)

    def test_validate_and_clean_depths_1d(self, sample_dataframe: pd.DataFrame) -> None:
        """validate_and_clean should return 1D depth array."""
        depths, _, _ = validate_and_clean(sample_dataframe)

        assert depths.ndim == 1

    def test_validate_and_clean_pixels_2d(self, sample_dataframe: pd.DataFrame) -> None:
        """validate_and_clean should return 2D pixel array."""
        _, pixels, _ = validate_and_clean(sample_dataframe)

        assert pixels.ndim == 2

    def test_validate_drops_null_rows(
        self, sample_dataframe_with_nulls: pd.DataFrame
    ) -> None:
        """validate_and_clean should drop fully-null rows."""
        _, _, report = validate_and_clean(sample_dataframe_with_nulls)

        assert report.null_rows_dropped >= 1

    def test_validate_drops_duplicate_depths(
        self, sample_dataframe_with_duplicates: pd.DataFrame
    ) -> None:
        """validate_and_clean should drop duplicate depth entries."""
        _, _, report = validate_and_clean(sample_dataframe_with_duplicates)

        assert report.duplicate_depths_dropped >= 1

    def test_validate_sorts_by_depth(self, sample_dataframe: pd.DataFrame) -> None:
        """validate_and_clean should sort data by depth ascending."""
        # Shuffle the dataframe
        shuffled = sample_dataframe.sample(frac=1, random_state=42)

        depths, _, _ = validate_and_clean(shuffled)

        assert np.all(np.diff(depths) > 0)  # Strictly increasing

    def test_validate_clips_pixel_values(self) -> None:
        """validate_and_clean should clip pixel values outside [0, 255]."""
        df = pd.DataFrame(
            {
                "depth": [1.0, 2.0],
                "pixel_0": [-10, 300],  # Out of range
                "pixel_1": [100, 150],
            }
        )

        _, pixels, _ = validate_and_clean(df)

        assert pixels.min() >= 0
        assert pixels.max() <= 255

    def test_validation_report_fields(self, sample_dataframe: pd.DataFrame) -> None:
        """ValidationReport should have all expected fields."""
        _, _, report = validate_and_clean(sample_dataframe)

        assert hasattr(report, "raw_rows")
        assert hasattr(report, "clean_rows")
        assert hasattr(report, "null_rows_dropped")
        assert hasattr(report, "duplicate_depths_dropped")
        assert hasattr(report, "depth_min")
        assert hasattr(report, "depth_max")
        assert hasattr(report, "depth_step")
        assert hasattr(report, "is_monotonic")
        assert hasattr(report, "pixel_min")
        assert hasattr(report, "pixel_max")
        assert hasattr(report, "original_width")

    def test_validation_report_values(self, sample_dataframe: pd.DataFrame) -> None:
        """ValidationReport should have correct values."""
        _, _, report = validate_and_clean(sample_dataframe)

        assert report.raw_rows == len(sample_dataframe)
        assert report.clean_rows <= report.raw_rows
        assert report.depth_min <= report.depth_max
        assert report.is_monotonic is True  # After sorting

    def test_validate_fills_nan_pixels(self) -> None:
        """validate_and_clean should fill NaN pixel values with row mean."""
        df = pd.DataFrame(
            {
                "depth": [1.0],
                "pixel_0": [100.0],
                "pixel_1": [np.nan],  # Will be filled with row mean
                "pixel_2": [200.0],
            }
        )

        _, pixels, _ = validate_and_clean(df)

        # NaN should be replaced with mean of [100, 200] = 150
        assert not np.isnan(pixels).any()
        assert pixels[0, 1] == 150.0  # Filled with mean


class TestResizeImage:
    """Tests for image resizing functionality."""

    @pytest.fixture
    def sample_pixel_array(self) -> np.ndarray:
        """Sample pixel array for resize testing."""
        np.random.seed(42)
        return np.random.rand(100, 200).astype(np.float64) * 255

    def test_resize_image_output_shape(self, sample_pixel_array: np.ndarray) -> None:
        """resize_image should produce correct output dimensions."""
        target_width = 150

        result = resize_image(sample_pixel_array, target_width)

        assert result.shape[0] == sample_pixel_array.shape[0]  # Same rows
        assert result.shape[1] == target_width

    def test_resize_image_output_dtype(self, sample_pixel_array: np.ndarray) -> None:
        """resize_image should return uint8 array."""
        result = resize_image(sample_pixel_array, 150)

        assert result.dtype == np.uint8

    def test_resize_image_output_range(self, sample_pixel_array: np.ndarray) -> None:
        """resize_image output should be in [0, 255] range."""
        result = resize_image(sample_pixel_array, 150)

        assert result.min() >= 0
        assert result.max() <= 255

    @pytest.mark.parametrize("method", list(INTERPOLATION_METHODS.keys()))
    def test_resize_all_interpolation_methods(
        self, sample_pixel_array: np.ndarray, method: str
    ) -> None:
        """All interpolation methods should work."""
        result = resize_image(sample_pixel_array, 150, method=method)

        assert result.shape == (100, 150)
        assert result.dtype == np.uint8

    def test_resize_unknown_method_raises(self, sample_pixel_array: np.ndarray) -> None:
        """resize_image should raise for unknown interpolation methods."""
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            resize_image(sample_pixel_array, 150, method="INVALID")

    def test_resize_error_lists_available_methods(
        self, sample_pixel_array: np.ndarray
    ) -> None:
        """Error message should list available methods."""
        with pytest.raises(ValueError) as exc_info:
            resize_image(sample_pixel_array, 150, method="INVALID")

        for method in INTERPOLATION_METHODS.keys():
            assert method in str(exc_info.value)

    def test_resize_upscale(self, sample_pixel_array: np.ndarray) -> None:
        """resize_image should handle upscaling."""
        target_width = 400  # Larger than original 200

        result = resize_image(sample_pixel_array, target_width)

        assert result.shape[1] == target_width

    def test_resize_same_size(self, sample_pixel_array: np.ndarray) -> None:
        """resize_image should handle same-size 'resize'."""
        target_width = 200  # Same as original

        result = resize_image(sample_pixel_array, target_width)

        assert result.shape[1] == target_width

    def test_resize_preserves_row_count(self, sample_pixel_array: np.ndarray) -> None:
        """resize_image should not change the number of rows."""
        result = resize_image(sample_pixel_array, 50)

        assert result.shape[0] == sample_pixel_array.shape[0]


class TestInterpolationMethods:
    """Tests for the interpolation method mapping."""

    def test_interpolation_methods_not_empty(self) -> None:
        """INTERPOLATION_METHODS should have entries."""
        assert len(INTERPOLATION_METHODS) > 0

    def test_interpolation_methods_expected_keys(self) -> None:
        """INTERPOLATION_METHODS should have expected methods."""
        expected = {"NEAREST", "BILINEAR", "BICUBIC", "AREA", "LANCZOS4"}

        assert expected == set(INTERPOLATION_METHODS.keys())

    def test_interpolation_methods_values_are_int(self) -> None:
        """All interpolation values should be integers (OpenCV flags)."""
        for value in INTERPOLATION_METHODS.values():
            assert isinstance(value, int)


class TestValidationReport:
    """Tests for the ValidationReport dataclass."""

    def test_validation_report_creation(self) -> None:
        """ValidationReport should be creatable with all fields."""
        report = ValidationReport(
            raw_rows=100,
            clean_rows=95,
            null_rows_dropped=3,
            duplicate_depths_dropped=2,
            depth_min=100.0,
            depth_max=200.0,
            depth_step=0.5,
            is_monotonic=True,
            pixel_min=0.0,
            pixel_max=255.0,
            original_width=200,
        )

        assert report.raw_rows == 100
        assert report.clean_rows == 95
        assert report.null_rows_dropped == 3
        assert report.duplicate_depths_dropped == 2
        assert report.depth_min == 100.0
        assert report.depth_max == 200.0
        assert report.depth_step == 0.5
        assert report.is_monotonic is True
        assert report.pixel_min == 0.0
        assert report.pixel_max == 255.0
        assert report.original_width == 200
