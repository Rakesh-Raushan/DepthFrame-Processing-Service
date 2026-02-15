"""Request/response schemas for the image frame API.

Pydantic models handle validation, serialization, and auto docs via FastAPI integration.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ImageQueryParams(BaseModel):
    """Query parameters for the image frame endpoint.

    Validates that depth bounds are within a reasonable range
    and that min < max. Additional validation against actual
    data bounds happens at the route level via the repository.
    """

    depth_min: float = Field(..., description="Minimum depth (inclusive)")
    depth_max: float = Field(..., description="Maximum depth (inclusive)")
    colormap: str = Field(
        default="resistivity",
        description="Colormap name to apply",
    )
    format: str = Field(
        default="png",
        description="Output image format",
        pattern="^(png|jpeg)$",
    )

    @model_validator(mode="after")
    def check_depth_bounds(self) -> ImageQueryParams:
        """Ensure that depth_min < depth_max."""
        if self.depth_min >= self.depth_max:
            raise ValueError(
                f"depth_min ({self.depth_min}) must be less than depth_max ({self.depth_max})"
            )
        return self


class MetadataResponse(BaseModel):
    """Response schema for the metadata endpoint."""

    depth_min: float
    depth_max: float
    depth_step: float
    row_count: int
    original_width: int
    resized_width: int
    interpolation_method: str
    source_file: str
    available_colormaps: list[str]


class ColormapInfo(BaseModel):
    """Individual colormap entry."""

    name: str
    description: str


class ColormapListResponse(BaseModel):
    """Response schema for the colormaps list endpoint."""

    colormaps: list[ColormapInfo]
    default: str


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str
    db_connected: bool
    row_count: int
