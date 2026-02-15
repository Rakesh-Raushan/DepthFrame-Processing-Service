"""Colormap registry for depth frame visualization.

Provides a pluggable registry of colormaps that can be applied to depth frames for visualization.
This allows plugging domain-appropriate colormaps (e.g. resistivity-based) without hardcoding them into the processing pipeline.

Usage:
    from borehole_image_service.colormaps.registry import ColormapRegistry

    registry = ColormapRegistry()
    cmap = registry.get("resistivity")
    colored_image = registry.apply("resistivity", grayscale_array)
"""

from __future__ import annotations

import io
import logging

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# some domian specific colormaps for depth visualization


_RESISTIVITY_COLORS = [
    "#1a0a00",
    "#4d2600",
    "#8B4513",
    "#CD853F",
    "#DAA520",
    "#FFD700",
    "#FFEC8B",
    "#FFFACD",
    "#FFFFFF",
]

_GEOLOGICAL_COLORS = [
    "#000033",
    "#003366",
    "#336633",
    "#669933",
    "#CC9933",
    "#CC6633",
    "#993333",
    "#FFFFFF",
]

_HIGH_CONTRAST_COLORS = [
    "#000000",
    "#1a1a2e",
    "#16213e",
    "#0f3460",
    "#e94560",
    "#ff6b6b",
    "#ffd93d",
    "#FFFFFF",
]


class ColormapRegistry:
    """Registry of named colormaps for image log visualization.

    Thread-safe for reads after initialization. Colormaps are registered
    at construction time; additional colormaps can be added via register().
    """

    def __init__(self) -> None:
        self._colormaps: dict[str, mcolors.Colormap] = {}
        self._descriptions: dict[str, str] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register domain-standard colormaps."""
        self.register(
            "resistivity",
            mcolors.LinearSegmentedColormap.from_list(
                "resistivity", _RESISTIVITY_COLORS, N=256
            ),
            description="Brown→Gold→White: standard for resistivity image logs",
        )
        self.register(
            "conductivity",
            mcolors.LinearSegmentedColormap.from_list(
                "resistivity", _RESISTIVITY_COLORS, N=256
            ).reversed(),
            description="White→Gold→Brown: inverted resistivity (conductivity convention)",
        )
        self.register(
            "geological",
            mcolors.LinearSegmentedColormap.from_list(
                "geological", _GEOLOGICAL_COLORS, N=256
            ),
            description="Blue→Green→Brown→White: formation boundary interpretation",
        )
        self.register(
            "high_contrast",
            mcolors.LinearSegmentedColormap.from_list(
                "high_contrast", _HIGH_CONTRAST_COLORS, N=256
            ),
            description="Dark→Red→Yellow→White: fracture detection and thin-bed analysis",
        )
        self.register(
            "gray",
            plt.get_cmap("gray"),
            description="Standard grayscale: raw data visualization",
        )
        self.register(
            "viridis",
            plt.get_cmap("viridis"),
            description="Perceptually uniform: general-purpose scientific visualization",
        )

    def register(
        self,
        name: str,
        cmap: mcolors.Colormap,
        description: str = "",
    ) -> None:
        """Register a colormap under a given name."""
        self._colormaps[name] = cmap
        self._descriptions[name] = description
        logger.debug("Registered colormap: %s", name)

    def get(self, name: str) -> mcolors.Colormap:
        """Retrieve a colormap by name.

        Raises KeyError if the name is not registered.
        """
        if name not in self._colormaps:
            raise KeyError(
                f"Unknown colormap: {name!r}. Available: {self.list_names()}"
            )
        return self._colormaps[name]

    def list_names(self) -> list[str]:
        """Return sorted list of registered colormap names."""
        return sorted(self._colormaps.keys())

    def list_colormaps(self) -> list[dict[str, str]]:
        """Return list of colormaps with their descriptions."""
        return [
            {"name": name, "description": self._descriptions.get(name, "")}
            for name in self.list_names()
        ]

    def apply(
        self,
        name: str,
        pixel_array: np.ndarray,
        output_format: str = "png",
        jpeg_quality: int = 90,
    ) -> bytes:
        """Apply a named colormap to a grayscale array, return image bytes.

        Args:
            name: registered colormap name
            pixel_array: 2D uint8 array (grayscale image)
            output_format: "png" or "jpeg"
            jpeg_quality: JPEG quality (1-100), ignored for PNG

        Returns:
            Encoded image bytes (PNG or JPEG)
        """
        cmap = self.get(name)

        # Normalize uint8 → [0, 1] for matplotlib colormap
        normalized = pixel_array.astype(np.float32) / 255.0

        # Apply colormap → RGBA float → RGB uint8
        colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)

        # Encode
        img = Image.fromarray(colored)
        buf = io.BytesIO()

        if output_format.lower() == "jpeg":
            img.save(buf, format="JPEG", quality=jpeg_quality)
        else:
            img.save(buf, format="PNG")

        return buf.getvalue()

    def has(self, name: str) -> bool:
        """Check if a colormap name is registered."""
        return name in self._colormaps


# Module-level singleton for convenience
colormap_registry = ColormapRegistry()
