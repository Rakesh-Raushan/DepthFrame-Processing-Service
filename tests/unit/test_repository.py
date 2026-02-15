"""Unit tests for the ImageRepository class.

Tests database operations in isolation using temporary/in-memory databases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from depthframe_processing_service.db.repository import ImageRepository, ScanRow


class TestImageRepositoryConnection:
    """Tests for repository connection management."""

    def test_connect_creates_database_file(self, temp_db_path: Path) -> None:
        """Repository should create database file on connect."""
        repo = ImageRepository(temp_db_path)
        assert not temp_db_path.exists()

        repo.connect()
        assert temp_db_path.exists()
        repo.close()

    def test_connect_creates_parent_directories(self, tmp_path: Path) -> None:
        """Repository should create parent directories if they don't exist."""
        nested_path = tmp_path / "nested" / "dir" / "test.db"
        repo = ImageRepository(nested_path)

        repo.connect()
        assert nested_path.exists()
        repo.close()

    def test_context_manager(self, temp_db_path: Path) -> None:
        """Repository should work as a context manager."""
        with ImageRepository(temp_db_path) as repo:
            assert repo.conn is not None

        # Connection should be closed after exiting context
        assert repo._conn is None

    def test_conn_property_raises_when_not_connected(self, temp_db_path: Path) -> None:
        """Accessing conn property should raise when not connected."""
        repo = ImageRepository(temp_db_path)

        with pytest.raises(RuntimeError, match="not connected"):
            _ = repo.conn


class TestImageRepositoryWrite:
    """Tests for write operations."""

    def test_bulk_insert_scans_success(
        self,
        repository: ImageRepository,
        sample_depths: np.ndarray,
        sample_pixel_data: np.ndarray,
    ) -> None:
        """bulk_insert_scans should insert all rows successfully."""
        count = repository.bulk_insert_scans(sample_depths, sample_pixel_data)

        assert count == len(sample_depths)
        assert repository.get_row_count() == len(sample_depths)

    def test_bulk_insert_scans_shape_mismatch(
        self,
        repository: ImageRepository,
        sample_pixel_data: np.ndarray,
    ) -> None:
        """bulk_insert_scans should raise on shape mismatch."""
        wrong_depths = np.array([1.0, 2.0])  # Wrong number of depths

        with pytest.raises(ValueError, match="Shape mismatch"):
            repository.bulk_insert_scans(wrong_depths, sample_pixel_data)

    def test_bulk_insert_scans_wrong_dtype(
        self,
        repository: ImageRepository,
        sample_depths: np.ndarray,
    ) -> None:
        """bulk_insert_scans should raise on non-uint8 pixel data."""
        float_pixels = np.random.rand(5, 150).astype(np.float32)

        with pytest.raises(TypeError, match="uint8"):
            repository.bulk_insert_scans(sample_depths, float_pixels)

    def test_bulk_insert_replaces_on_duplicate_depth(
        self,
        repository: ImageRepository,
    ) -> None:
        """bulk_insert_scans should replace existing rows on duplicate depth."""
        depths = np.array([100.0, 101.0])
        pixels1 = np.zeros((2, 150), dtype=np.uint8)
        pixels2 = np.full((2, 150), 255, dtype=np.uint8)

        repository.bulk_insert_scans(depths, pixels1)
        repository.bulk_insert_scans(depths, pixels2)

        # Should still have 2 rows, not 4
        assert repository.get_row_count() == 2

        # Data should be updated
        rows = repository.query_depth_range(100.0, 101.0)
        assert np.all(rows[0].pixel_data == 255)

    def test_set_metadata(self, repository: ImageRepository) -> None:
        """set_metadata should store key-value pairs."""
        repository.set_metadata("source_file", "test.csv")
        repository.set_metadata("version", "1.0")

        assert repository.get_metadata("source_file") == "test.csv"
        assert repository.get_metadata("version") == "1.0"

    def test_set_metadata_replaces_existing(self, repository: ImageRepository) -> None:
        """set_metadata should replace existing values."""
        repository.set_metadata("key", "value1")
        repository.set_metadata("key", "value2")

        assert repository.get_metadata("key") == "value2"


class TestImageRepositoryRead:
    """Tests for read operations."""

    def test_query_depth_range(self, populated_repository: ImageRepository) -> None:
        """query_depth_range should return rows within the specified range."""
        rows = populated_repository.query_depth_range(100.0, 101.0)

        # Should include 100.0, 100.5, 101.0
        assert len(rows) == 3
        assert all(isinstance(row, ScanRow) for row in rows)
        assert all(100.0 <= row.depth <= 101.0 for row in rows)

    def test_query_depth_range_ordered(
        self, populated_repository: ImageRepository
    ) -> None:
        """query_depth_range should return rows ordered by depth ascending."""
        rows = populated_repository.query_depth_range(100.0, 102.0)

        depths = [row.depth for row in rows]
        assert depths == sorted(depths)

    def test_query_depth_range_empty(
        self, populated_repository: ImageRepository
    ) -> None:
        """query_depth_range should return empty list for non-matching range."""
        rows = populated_repository.query_depth_range(500.0, 600.0)

        assert rows == []

    def test_query_depth_range_pixel_data_correct(
        self,
        repository: ImageRepository,
    ) -> None:
        """query_depth_range should return correct pixel data."""
        depths = np.array([100.0])
        pixels = np.arange(150, dtype=np.uint8).reshape(1, 150)

        repository.bulk_insert_scans(depths, pixels)
        rows = repository.query_depth_range(99.0, 101.0)

        assert len(rows) == 1
        assert np.array_equal(rows[0].pixel_data, pixels[0])

    def test_get_metadata_returns_none_for_missing(
        self, repository: ImageRepository
    ) -> None:
        """get_metadata should return None for non-existent keys."""
        assert repository.get_metadata("nonexistent") is None

    def test_get_all_metadata(self, repository: ImageRepository) -> None:
        """get_all_metadata should return all stored metadata."""
        repository.set_metadata("key1", "value1")
        repository.set_metadata("key2", "value2")

        metadata = repository.get_all_metadata()

        assert metadata == {"key1": "value1", "key2": "value2"}

    def test_get_depth_bounds(self, populated_repository: ImageRepository) -> None:
        """get_depth_bounds should return min and max depths."""
        min_depth, max_depth = populated_repository.get_depth_bounds()

        assert min_depth == 100.0
        assert max_depth == 102.0

    def test_get_depth_bounds_empty_raises(self, repository: ImageRepository) -> None:
        """get_depth_bounds should raise when database is empty."""
        with pytest.raises(ValueError, match="No scan data"):
            repository.get_depth_bounds()

    def test_get_row_count(self, populated_repository: ImageRepository) -> None:
        """get_row_count should return correct count."""
        assert populated_repository.get_row_count() == 5

    def test_get_row_count_empty(self, repository: ImageRepository) -> None:
        """get_row_count should return 0 for empty database."""
        assert repository.get_row_count() == 0

    def test_is_populated_true(self, populated_repository: ImageRepository) -> None:
        """is_populated should return True when data exists."""
        assert populated_repository.is_populated() is True

    def test_is_populated_false(self, repository: ImageRepository) -> None:
        """is_populated should return False when empty."""
        assert repository.is_populated() is False


class TestScanRow:
    """Tests for the ScanRow dataclass."""

    def test_scan_row_frozen(self) -> None:
        """ScanRow should be immutable (frozen)."""
        row = ScanRow(depth=100.0, pixel_data=np.array([1, 2, 3], dtype=np.uint8))

        with pytest.raises(Exception):  # FrozenInstanceError
            row.depth = 200.0  # type: ignore

    def test_scan_row_properties(self) -> None:
        """ScanRow should store depth and pixel_data correctly."""
        pixel_data = np.array([10, 20, 30], dtype=np.uint8)
        row = ScanRow(depth=150.5, pixel_data=pixel_data)

        assert row.depth == 150.5
        assert np.array_equal(row.pixel_data, pixel_data)
