"""Ingestion pipeline: CSV => validation +> resize => store in SQLite storage.
Each step is modularized for testability and maintainability.
The pipeline orchestrator (run_ingestion) composes these steps

Assumes:
- Small-to-moderate CSV size that can be loaded into memory for processing.
For larger datasets, we’d implement chunked/streaming processing and revisit global sort/deduplication logic.
- Validation rules are as per the provided data but we can have a separate validation module for more complex rules or schema evolution.
- Resizing is done in-memory with OpenCV for speed and quality; for very large datasets, we could consider out-of-core processing or GPU acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import cv2
import pandas as pd
import numpy as np

from depthframe_processing_service.config import Settings
from depthframe_processing_service.db.repository import ImageRepository

logger = logging.getLogger(__name__)

# Map config strigs to OpenCV interpolation flags
INTERPOLATION_METHODS = {
    "NEAREST": cv2.INTER_NEAREST,
    "BILINEAR": cv2.INTER_LINEAR,
    "BICUBIC": cv2.INTER_CUBIC,
    "AREA": cv2.INTER_AREA,
    "LANCZOS4": cv2.INTER_LANCZOS4,
}


@dataclass
class ValidationReport:
    """Summary of data quality checks performed during ingestion."""

    raw_rows: int
    clean_rows: int
    null_rows_dropped: int
    duplicate_depths_dropped: int
    depth_min: float
    depth_max: float
    depth_step: float
    is_monotonic: bool
    pixel_min: float
    pixel_max: float
    original_width: int


# Load


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV data into a DataFrame.
    Raises FileNotFoundError with a clear message if the data file
    is missing — a common first-run issue.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            f"Place the CSV in the data/ directory. See data/README.md."
        )
    logger.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows x %d columns", df.shape[0], df.shape[1])
    return df


# Validate and clean


def validate_and_clean(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, ValidationReport]:
    """Validate data quality and return cleaned arrays.

    Checks performed:
    - Drop fully-null rows (CSV EOF artifacts)
    - Drop duplicate depths (keep first)
    - Verify depth monotonicity
    - Verify pixel value range [0, 255]
    - Extract depth and pixel arrays

    Returns:
        (depths, pixels, report) where depths is 1D float64,
        pixels is 2D float64 (not yet uint8 — resize first).
    """

    raw_rows = len(df)

    # Drop rows where ALL values are null (trailing CSV artifacts seen in given data)
    null_mask = df.isnull().all(axis=1)
    null_count = null_mask.sum()
    if null_count > 0:
        logger.info("Dropping %d fully-null rows", null_count)
        df = df[~null_mask].copy()

    # Drop rows with null depth (unusable without index)
    depth_null = df["depth"].isnull()
    if depth_null.any():
        logger.warning("Dropping %d rows with null depth", depth_null.sum())
        df = df[~depth_null].copy()

    # Sort by depth to ensure monotonicity
    df = df.sort_values("depth").reset_index(drop=True)

    # Drop duplicate depths (keep first occurrence)
    dup_mask = df["depth"].duplicated(keep="first")
    dup_count = dup_mask.sum()
    if dup_count > 0:
        logger.warning("Dropping %d duplicate depth entries", dup_count)
        df = df[~dup_mask].reset_index(drop=True)

    # Extract arrays
    depths: np.ndarray = np.array(df["depth"].values, dtype=np.float64)
    pixel_cols = [c for c in df.columns if c != "depth"]
    pixels: np.ndarray = np.array(df[pixel_cols].values, dtype=np.float64)

    # Fill any remaining NaN pixels with row mean (rare but defensive)
    row_has_nan = np.isnan(pixels).any(axis=1)
    if row_has_nan.any():
        nan_count = row_has_nan.sum()
        logger.warning("Filling NaN pixels in %d rows with row mean", nan_count)
        for i in np.where(row_has_nan)[0]:
            row = pixels[i]
            row_mean = np.nanmean(row)
            pixels[i] = np.where(np.isnan(row), row_mean, row)

    # Validate pixel range
    px_min, px_max = pixels.min(), pixels.max()
    if px_min < 0 or px_max > 255:
        logger.warning(
            "Pixel values outside [0, 255]: [%.1f, %.1f]. Clipping.",
            px_min,
            px_max,
        )
        pixels = np.clip(pixels, 0, 255)

    # Depth step analysis
    depth_diffs = np.diff(depths)
    is_monotonic = bool(np.all(depth_diffs > 0))
    depth_step = float(np.median(depth_diffs))

    report = ValidationReport(
        raw_rows=raw_rows,
        clean_rows=len(depths),
        null_rows_dropped=null_count,
        duplicate_depths_dropped=dup_count,
        depth_min=float(depths.min()),
        depth_max=float(depths.max()),
        depth_step=depth_step,
        is_monotonic=is_monotonic,
        pixel_min=float(px_min),
        pixel_max=float(px_max),
        original_width=pixels.shape[1],
    )

    logger.info(
        "Validation complete: %d clean rows, depth %.1f–%.1f, pixels [%.0f, %.0f]",
        report.clean_rows,
        report.depth_min,
        report.depth_max,
        report.pixel_min,
        report.pixel_max,
    )

    return depths, pixels, report


# Resize


def resize_image(
    pixels: np.ndarray, target_width: int, method: str = "AREA"
) -> np.ndarray:
    """Resize the full image array from original width to target width.

    Interpolation is performed in float32 precision. Quantization to
    uint8 is done as the final step — see notebooks/02_resizing_analysis.ipynb
    for justification.

    Args:
        pixels: 2D float64 array, shape (n_rows, original_width)
        target_width: desired output width (e.g., 150)
        method: interpolation method name (key in INTERPOLATION_METHODS)

    Returns:
        2D uint8 array, shape (n_rows, target_width)
    """
    if method not in INTERPOLATION_METHODS:
        raise ValueError(
            f"Unknown interpolation method: {method!r}. "
            f"Available: {list(INTERPOLATION_METHODS.keys())}"
        )

    interp_flag = INTERPOLATION_METHODS[method]

    # Work in float32 for interpolation precision
    img_f32 = pixels.astype(np.float32)
    resized = cv2.resize(
        img_f32,
        (target_width, img_f32.shape[0]),
        interpolation=interp_flag,
    )

    # Clip and quantize as final step
    resized = np.clip(resized, 0, 255).astype(np.uint8)

    logger.info(
        "Resized: (%d, %d) → (%d, %d) using %s",
        pixels.shape[0],
        pixels.shape[1],
        resized.shape[0],
        resized.shape[1],
        method,
    )
    return resized


# Store


def store_to_database(
    repo: ImageRepository,
    depths: np.ndarray,
    pixels: np.ndarray,
    report: ValidationReport,
    settings: Settings,
) -> None:
    """Store resized image data and metadata into the repository.

    Metadata includes processing provenance so downstream consumers
    know how the data was produced.
    """
    # Store pixel data
    row_count = repo.bulk_insert_scans(depths, pixels)

    # Store metadata for provenance and API self-description
    metadata = {
        "original_width": str(report.original_width),
        "resized_width": str(pixels.shape[1]),
        "depth_min": str(report.depth_min),
        "depth_max": str(report.depth_max),
        "depth_step": str(report.depth_step),
        "row_count": str(row_count),
        "interpolation_method": settings.interpolation_method,
        "source_file": settings.csv_filename,
        "pixel_range_original": f"[{report.pixel_min:.0f}, {report.pixel_max:.0f}]",
        "validation_summary": json.dumps(
            {
                "raw_rows": report.raw_rows,
                "clean_rows": report.clean_rows,
                "null_rows_dropped": report.null_rows_dropped,
                "duplicate_depths_dropped": report.duplicate_depths_dropped,
                "is_monotonic": report.is_monotonic,
            }
        ),
    }
    for key, value in metadata.items():
        repo.set_metadata(key, value)

    logger.info("Stored %d rows + %d metadata entries", row_count, len(metadata))


# Orchestrator


def run_ingestion(settings: Settings | None = None) -> ValidationReport:
    """Run the full ingestion pipeline: load → validate → resize → store.

    This is the main entry point. Idempotent: if the database is already
    populated, it will be overwritten (INSERT OR REPLACE).

    Args:
        settings: optional settings override (uses global defaults if None)

    Returns:
        Validation report from the pipeline run.
    """
    if settings is None:
        from depthframe_processing_service.config import settings as default_settings

        settings = default_settings

    logger.info("=" * 60)
    logger.info("Starting ingestion pipeline")
    logger.info("=" * 60)

    # Step 1: Load
    df = load_csv(settings.csv_path)

    # Step 2: Validate
    depths, pixels, report = validate_and_clean(df)

    # Step 3: Resize
    resized = resize_image(
        pixels,
        target_width=settings.target_width,
        method=settings.interpolation_method,
    )

    # Step 4: Store
    with ImageRepository(settings.db_path) as repo:
        store_to_database(repo, depths, resized, report, settings)

    logger.info("=" * 60)
    logger.info("Ingestion complete: %d rows stored", report.clean_rows)
    logger.info("=" * 60)

    return report
