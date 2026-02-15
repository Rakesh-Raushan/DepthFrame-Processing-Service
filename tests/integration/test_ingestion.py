"""Integration tests for the complete ingestion pipeline.

Tests the full workflow: CSV loading → validation → resizing → database storage.
These tests verify that components work correctly together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from depthframe_processing_service.colormaps.registry import ColormapRegistry
from depthframe_processing_service.db.repository import ImageRepository
from depthframe_processing_service.ingestion.pipeline import (
    load_csv,
    resize_image,
    validate_and_clean,
)


class TestCsvToValidation:
    """Integration tests for CSV loading through validation."""

    def test_load_and_validate_pipeline(self, temp_csv_file: Path) -> None:
        """Should load CSV and validate successfully."""
        # Load
        df = load_csv(temp_csv_file)

        # Validate
        depths, pixels, report = validate_and_clean(df)

        # Verify integration
        assert len(depths) == report.clean_rows
        assert pixels.shape[0] == report.clean_rows
        assert report.original_width == pixels.shape[1]

    def test_load_validate_with_data_issues(self, temp_csv_with_issues: Path) -> None:
        """Should handle CSV with data quality issues end-to-end."""
        df = load_csv(temp_csv_with_issues)
        depths, pixels, report = validate_and_clean(df)

        # Should have cleaned up the data
        assert report.clean_rows < report.raw_rows
        assert len(depths) == report.clean_rows

        # Depths should be monotonic after cleaning
        assert np.all(np.diff(depths) > 0)


class TestValidationToResize:
    """Integration tests for validation through resizing."""

    def test_validate_and_resize_pipeline(self, sample_dataframe: pd.DataFrame) -> None:
        """Should validate data and resize successfully."""
        depths, pixels, report = validate_and_clean(sample_dataframe)

        target_width = 150
        resized = resize_image(pixels, target_width)

        # Same number of rows
        assert resized.shape[0] == len(depths)
        # New width
        assert resized.shape[1] == target_width
        # Proper dtype for storage
        assert resized.dtype == np.uint8

    def test_resize_maintains_row_correspondence(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Each resized row should correspond to its original depth."""
        depths, pixels, _ = validate_and_clean(sample_dataframe)
        resized = resize_image(pixels, 150)

        # Verify 1:1 correspondence
        assert len(depths) == resized.shape[0]


class TestResizeToStorage:
    """Integration tests for resizing through database storage."""

    @pytest.fixture
    def processed_data(
        self, sample_dataframe: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, object]:
        """Provide validated and resized data ready for storage."""
        depths, pixels, report = validate_and_clean(sample_dataframe)
        resized = resize_image(pixels, 150)
        return depths, resized, report

    def test_store_resized_data(
        self,
        repository: ImageRepository,
        processed_data: tuple[np.ndarray, np.ndarray, object],
    ) -> None:
        """Should store resized data in repository successfully."""
        depths, resized, _ = processed_data

        row_count = repository.bulk_insert_scans(depths, resized)

        assert row_count == len(depths)
        assert repository.get_row_count() == len(depths)

    def test_retrieve_stored_data(
        self,
        repository: ImageRepository,
        processed_data: tuple[np.ndarray, np.ndarray, object],
    ) -> None:
        """Should retrieve stored data with correct values."""
        depths, resized, _ = processed_data
        repository.bulk_insert_scans(depths, resized)

        # Query all stored data
        min_depth, max_depth = repository.get_depth_bounds()
        rows = repository.query_depth_range(min_depth, max_depth)

        assert len(rows) == len(depths)

        # Verify pixel data matches
        for i, row in enumerate(rows):
            assert row.depth == depths[i]
            assert np.array_equal(row.pixel_data, resized[i])


class TestFullPipeline:
    """End-to-end integration tests for the complete pipeline."""

    def test_csv_to_database(
        self,
        temp_csv_file: Path,
        repository: ImageRepository,
    ) -> None:
        """Complete pipeline: CSV → validate → resize → database."""
        # Step 1: Load
        df = load_csv(temp_csv_file)

        # Step 2: Validate
        depths, pixels, report = validate_and_clean(df)

        # Step 3: Resize
        target_width = 150
        resized = resize_image(pixels, target_width)

        # Step 4: Store
        repository.bulk_insert_scans(depths, resized)

        # Verify end-to-end
        assert repository.is_populated()
        assert repository.get_row_count() == report.clean_rows

        # Query back and verify
        min_d, max_d = repository.get_depth_bounds()
        rows = repository.query_depth_range(min_d, max_d)

        assert all(row.pixel_data.shape[0] == target_width for row in rows)

    def test_pipeline_with_metadata(
        self,
        temp_csv_file: Path,
        repository: ImageRepository,
    ) -> None:
        """Pipeline should properly store metadata alongside scan data."""
        df = load_csv(temp_csv_file)
        depths, pixels, report = validate_and_clean(df)
        resized = resize_image(pixels, 150)
        repository.bulk_insert_scans(depths, resized)

        # Store metadata manually (as would be done by store_to_database)
        repository.set_metadata("original_width", str(report.original_width))
        repository.set_metadata("resized_width", "150")
        repository.set_metadata("row_count", str(report.clean_rows))

        # Verify metadata retrieval
        assert repository.get_metadata("original_width") == str(report.original_width)
        assert repository.get_metadata("resized_width") == "150"
        assert repository.get_metadata("row_count") == str(report.clean_rows)


class TestRepositoryWithColormaps:
    """Integration tests for repository data with colormap application."""

    def test_retrieve_and_colorize(
        self,
        populated_repository: ImageRepository,
        colormap_registry: ColormapRegistry,
    ) -> None:
        """Should retrieve scan data and apply colormaps."""
        # Get depth range
        min_depth, max_depth = populated_repository.get_depth_bounds()

        # Retrieve rows
        rows = populated_repository.query_depth_range(min_depth, max_depth)

        # Stack into 2D image
        pixel_data = np.stack([row.pixel_data for row in rows])

        # Apply colormap
        colored = colormap_registry.apply("resistivity", pixel_data)

        assert isinstance(colored, bytes)
        assert len(colored) > 0
        # Should be PNG
        assert colored[:8] == b"\x89PNG\r\n\x1a\n"

    def test_colormap_consistency_across_queries(
        self,
        populated_repository: ImageRepository,
        colormap_registry: ColormapRegistry,
    ) -> None:
        """Same data should produce same colormap output."""
        min_depth, max_depth = populated_repository.get_depth_bounds()

        # Query twice
        rows1 = populated_repository.query_depth_range(min_depth, max_depth)
        rows2 = populated_repository.query_depth_range(min_depth, max_depth)

        # Stack and colorize
        img1 = np.stack([row.pixel_data for row in rows1])
        img2 = np.stack([row.pixel_data for row in rows2])

        colored1 = colormap_registry.apply("resistivity", img1)
        colored2 = colormap_registry.apply("resistivity", img2)

        assert colored1 == colored2


class TestDepthRangeQueries:
    """Integration tests for depth-based querying."""

    @pytest.fixture
    def repository_with_range(self, repository: ImageRepository) -> ImageRepository:
        """Repository with known depth range for testing queries."""
        depths = np.arange(100.0, 200.0, 0.5)  # 100.0, 100.5, ... 199.5
        pixels = np.random.randint(0, 256, size=(len(depths), 150), dtype=np.uint8)
        repository.bulk_insert_scans(depths, pixels)
        return repository

    def test_query_partial_range(self, repository_with_range: ImageRepository) -> None:
        """Should query a subset of depth range."""
        rows = repository_with_range.query_depth_range(120.0, 130.0)

        # Should have depths 120.0, 120.5, ..., 130.0 = 21 rows
        assert len(rows) == 21
        assert all(120.0 <= row.depth <= 130.0 for row in rows)

    def test_query_single_depth(self, repository_with_range: ImageRepository) -> None:
        """Should query a single exact depth value."""
        rows = repository_with_range.query_depth_range(150.0, 150.0)

        assert len(rows) == 1
        assert rows[0].depth == 150.0

    def test_query_non_existent_range(
        self, repository_with_range: ImageRepository
    ) -> None:
        """Should return empty list for non-matching range."""
        rows = repository_with_range.query_depth_range(500.0, 600.0)

        assert rows == []

    def test_query_maintains_order(
        self, repository_with_range: ImageRepository
    ) -> None:
        """Query results should be in ascending depth order."""
        rows = repository_with_range.query_depth_range(100.0, 199.5)

        depths = [row.depth for row in rows]
        assert depths == sorted(depths)
