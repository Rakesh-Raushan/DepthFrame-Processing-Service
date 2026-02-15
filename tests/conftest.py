"""Shared pytest fixtures for all test modules."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from depthframe_processing_service.db.repository import ImageRepository
from depthframe_processing_service.colormaps.registry import ColormapRegistry


# ── Repository Fixtures ──


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path for tests."""
    return tmp_path / "test_image_store.db"


@pytest.fixture
def repository(temp_db_path: Path) -> Generator[ImageRepository, None, None]:
    """Provide a connected ImageRepository with an in-memory/temp database."""
    repo = ImageRepository(temp_db_path)
    repo.connect()
    yield repo
    repo.close()


@pytest.fixture
def in_memory_repository() -> Generator[ImageRepository, None, None]:
    """Provide an ImageRepository using SQLite in-memory database."""
    # Use :memory: via a temp file path that doesn't persist
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "memory_test.db"
        repo = ImageRepository(db_path)
        repo.connect()
        yield repo
        repo.close()


# ── Sample Data Fixtures ──


@pytest.fixture
def sample_depths() -> np.ndarray:
    """Sample depth values for testing."""
    return np.array([100.0, 100.5, 101.0, 101.5, 102.0], dtype=np.float64)


@pytest.fixture
def sample_pixel_data() -> np.ndarray:
    """Sample pixel data (5 rows x 150 columns) for testing."""
    np.random.seed(42)
    return np.random.randint(0, 256, size=(5, 150), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image() -> np.ndarray:
    """Sample 2D grayscale image for colormap testing."""
    np.random.seed(42)
    return np.random.randint(0, 256, size=(100, 150), dtype=np.uint8)


@pytest.fixture
def populated_repository(
    repository: ImageRepository,
    sample_depths: np.ndarray,
    sample_pixel_data: np.ndarray,
) -> ImageRepository:
    """Provide a repository with sample data already inserted."""
    repository.bulk_insert_scans(sample_depths, sample_pixel_data)
    return repository


# ── Colormap Fixtures ──


@pytest.fixture
def colormap_registry() -> ColormapRegistry:
    """Provide a fresh ColormapRegistry instance."""
    return ColormapRegistry()


# ── DataFrame Fixtures ──


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample DataFrame mimicking CSV input with depth and pixel columns."""
    np.random.seed(42)
    n_rows = 10
    n_pixels = 200  # Original width before resizing

    depths = np.linspace(100.0, 105.0, n_rows)
    pixel_data = np.random.randint(0, 256, size=(n_rows, n_pixels))

    # Create DataFrame with depth column and pixel_0, pixel_1, ... columns
    df = pd.DataFrame(pixel_data, columns=[f"pixel_{i}" for i in range(n_pixels)])
    df.insert(0, "depth", depths)

    return df


@pytest.fixture
def sample_dataframe_with_nulls(sample_dataframe: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with some null values for validation testing."""
    df = sample_dataframe.copy()
    # Add a fully null row
    null_row = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
    df = pd.concat([df, null_row], ignore_index=True)
    # Add a row with null depth
    null_depth_row = df.iloc[0:1].copy()
    null_depth_row["depth"] = np.nan
    df = pd.concat([df, null_depth_row], ignore_index=True)
    return df


@pytest.fixture
def sample_dataframe_with_duplicates(sample_dataframe: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with duplicate depth values for deduplication testing."""
    df = sample_dataframe.copy()
    # Duplicate the first row
    dup_row = df.iloc[0:1].copy()
    df = pd.concat([df, dup_row], ignore_index=True)
    return df


# ── Temporary CSV Fixtures ──


@pytest.fixture
def temp_csv_file(tmp_path: Path, sample_dataframe: pd.DataFrame) -> Path:
    """Create a temporary CSV file from sample data."""
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_csv_with_issues(
    tmp_path: Path, sample_dataframe_with_nulls: pd.DataFrame
) -> Path:
    """Create a temporary CSV file with data quality issues."""
    csv_path = tmp_path / "test_data_issues.csv"
    sample_dataframe_with_nulls.to_csv(csv_path, index=False)
    return csv_path
