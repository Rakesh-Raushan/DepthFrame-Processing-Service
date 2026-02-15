"""API route handlers for image frame retrieval.

All routes are mounted under /api/v1/ for versioning.
Each handler is light: validate input, call domain services,
return response. Business logic managed at repository
and colormap layers.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from depthframe_processing_service.api.dependencies import (
    get_colormap_registry,
    get_repository,
)
from depthframe_processing_service.api.schemas import (
    ColormapInfo,
    ColormapListResponse,
    HealthResponse,
    MetadataResponse,
)
from depthframe_processing_service.colormaps.registry import ColormapRegistry
from depthframe_processing_service.config import settings
from depthframe_processing_service.db.repository import ImageRepository

from enum import Enum


class ImageFormat(str, Enum):
    """Supported output image formats."""

    png = "png"
    jpeg = "jpeg"


class ColormapName(str, Enum):
    """Available colormap names. Renders as dropdown in Swagger UI."""

    resistivity = "resistivity"
    conductivity = "conductivity"
    geological = "geological"
    high_contrast = "high_contrast"
    gray = "gray"
    viridis = "viridis"


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["image"])


# Image endpoints


@router.get(
    "/image",
    summary="Get colormapped image frame for a depth range",
    responses={
        200: {"content": {"image/png": {}, "image/jpeg": {}}},
        400: {"description": "Invalid parameters"},
        404: {"description": "No data in requested depth range"},
    },
)
async def get_image_frame(
    depth_min: float = Query(..., description="Minimum depth (inclusive)"),
    depth_max: float = Query(..., description="Maximum depth (inclusive)"),
    colormap: ColormapName = Query(
        default=ColormapName.resistivity,
        description="Colormap name to apply",
    ),
    format: ImageFormat = Query(
        default=ImageFormat.png,
        description="Output image format",
    ),
    repo: ImageRepository = Depends(get_repository),
    registry: ColormapRegistry = Depends(get_colormap_registry),
) -> Response:
    """Retrieve an image frame between depth_min and depth_max.

    The grayscale pixel data is fetched from the database, assembled
    into a 2D frame, colormapped, and returned as a PNG or JPEG image.

    The colormap is applied at response time — stored data is always
    grayscale. This allows different colormaps to be applied to the
    same data without re-ingestion.
    """
    # Validate depth ordering
    if depth_min >= depth_max:
        raise HTTPException(
            status_code=400,
            detail=f"depth_min ({depth_min}) must be less than depth_max ({depth_max})",
        )

    # Validate colormap exists
    if not registry.has(colormap):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown colormap: {colormap!r}. Available: {registry.list_names()}"
            ),
        )

    # Validate depth bounds against actual data
    try:
        data_min, data_max = repo.get_depth_bounds()
    except ValueError:
        raise HTTPException(
            status_code=503, detail="Database is empty. Run ingestion first."
        )

    if depth_min > data_max or depth_max < data_min:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No data in range [{depth_min}, {depth_max}]. "
                f"Available data range: [{data_min}, {data_max}]"
            ),
        )

    # Query
    rows = repo.query_depth_range(depth_min, depth_max)

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No scan rows found between {depth_min} and {depth_max}",
        )

    # Assemble frame: list of ScanRow → 2D numpy array
    frame = np.array([row.pixel_data for row in rows])

    logger.info(
        "Serving frame: depth [%.1f, %.1f], shape %s, colormap=%s",
        depth_min,
        depth_max,
        frame.shape,
        colormap,
    )

    # Apply colormap and encode
    image_bytes = registry.apply(
        colormap,
        frame,
        output_format=format,
    )

    media_type = "image/png" if format == "png" else "image/jpeg"
    return Response(
        content=image_bytes,
        media_type=media_type,
        headers={
            "X-Frame-Depth-Min": str(rows[0].depth),
            "X-Frame-Depth-Max": str(rows[-1].depth),
            "X-Frame-Rows": str(len(rows)),
            "X-Frame-Width": str(frame.shape[1]),
            "X-Colormap": colormap,
        },
    )


# Metadata endpoints


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Get dataset metadata and processing provenance",
)
async def get_metadata(
    repo: ImageRepository = Depends(get_repository),
    registry: ColormapRegistry = Depends(get_colormap_registry),
) -> MetadataResponse:
    """Return metadata about the stored dataset.

    Includes processing parameters (interpolation method, dimensions),
    depth bounds, and available colormaps. Makes the API self-describing
    for clients that need to know valid query ranges.
    """
    meta = repo.get_all_metadata()

    if not meta:
        raise HTTPException(
            status_code=503,
            detail="No metadata available. Run ingestion first.",
        )

    return MetadataResponse(
        depth_min=float(meta.get("depth_min", 0)),
        depth_max=float(meta.get("depth_max", 0)),
        depth_step=float(meta.get("depth_step", 0)),
        row_count=int(meta.get("row_count", 0)),
        original_width=int(meta.get("original_width", 0)),
        resized_width=int(meta.get("resized_width", 0)),
        interpolation_method=meta.get("interpolation_method", "unknown"),
        source_file=meta.get("source_file", "unknown"),
        available_colormaps=registry.list_names(),
    )


@router.get(
    "/colormaps",
    response_model=ColormapListResponse,
    summary="List available colormaps",
)
async def list_colormaps(
    registry: ColormapRegistry = Depends(get_colormap_registry),
) -> ColormapListResponse:
    """Return all registered colormaps with descriptions.

    Clients can use this to populate a colormap picker UI
    or validate colormap names before requesting frames.
    """
    return ColormapListResponse(
        colormaps=[ColormapInfo(**entry) for entry in registry.list_colormaps()],
        default=settings.default_colormap,
    )


# Health endpoints

health_router = APIRouter(tags=["health"])


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health_check(
    repo: ImageRepository = Depends(get_repository),
) -> HealthResponse:
    """Basic health check for orchestration (Useful for Docker HEALTHCHECK, k8s probes)."""
    try:
        row_count = repo.get_row_count()
        return HealthResponse(
            status="healthy",
            db_connected=True,
            row_count=row_count,
        )
    except Exception:
        return HealthResponse(
            status="unhealthy",
            db_connected=False,
            row_count=0,
        )


# raw data endpoint


@router.get(
    "/image/raw",
    summary="Get raw grayscale pixel data for a depth range",
    responses={
        200: {"content": {"application/octet-stream": {}}},
        404: {"description": "No data in requested depth range"},
    },
)
async def get_raw_frame(
    depth_min: float = Query(..., description="Minimum depth (inclusive)"),
    depth_max: float = Query(..., description="Maximum depth (inclusive)"),
    repo: ImageRepository = Depends(get_repository),
) -> Response:
    """Retrieve raw grayscale pixel data as a binary numpy array.

    Useful for ML pipelines consumers that want
    to apply their own processing. The response is a flat uint8
    buffer; reshape using the dimensions in the response headers.
    """
    if depth_min >= depth_max:
        raise HTTPException(
            status_code=400,
            detail=f"depth_min ({depth_min}) must be less than depth_max ({depth_max})",
        )

    rows = repo.query_depth_range(depth_min, depth_max)

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No scan rows found between {depth_min} and {depth_max}",
        )

    frame = np.array([row.pixel_data for row in rows])

    return Response(
        content=frame.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Frame-Depth-Min": str(rows[0].depth),
            "X-Frame-Depth-Max": str(rows[-1].depth),
            "X-Frame-Rows": str(frame.shape[0]),
            "X-Frame-Width": str(frame.shape[1]),
            "X-Dtype": "uint8",
        },
    )
