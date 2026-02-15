"""Unit tests for the ColormapRegistry class.

Tests colormap registration, retrieval, and application in isolation.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from depthframe_processing_service.colormaps.registry import (
    ColormapRegistry,
    colormap_registry,
)


class TestColormapRegistryDefaults:
    """Tests for default colormap registration."""

    def test_default_colormaps_registered(
        self, colormap_registry: ColormapRegistry
    ) -> None:
        """Registry should have default colormaps pre-registered."""
        expected = [
            "conductivity",
            "geological",
            "gray",
            "high_contrast",
            "resistivity",
            "viridis",
        ]

        registered = colormap_registry.list_names()

        for name in expected:
            assert name in registered

    def test_list_names_sorted(self, colormap_registry: ColormapRegistry) -> None:
        """list_names should return names in sorted order."""
        names = colormap_registry.list_names()

        assert names == sorted(names)

    def test_list_colormaps_returns_descriptions(
        self, colormap_registry: ColormapRegistry
    ) -> None:
        """list_colormaps should return names with descriptions."""
        colormaps = colormap_registry.list_colormaps()

        assert all("name" in cmap for cmap in colormaps)
        assert all("description" in cmap for cmap in colormaps)

        # Check a specific description
        resistivity = next(c for c in colormaps if c["name"] == "resistivity")
        assert "Brown" in resistivity["description"]


class TestColormapRegistryOperations:
    """Tests for colormap registration and retrieval."""

    def test_register_new_colormap(self, colormap_registry: ColormapRegistry) -> None:
        """Should be able to register a new colormap."""
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom", ["#000000", "#FFFFFF"], N=256
        )

        colormap_registry.register("custom", custom_cmap, "Custom test colormap")

        assert colormap_registry.has("custom")
        assert "custom" in colormap_registry.list_names()

    def test_get_registered_colormap(self, colormap_registry: ColormapRegistry) -> None:
        """get should return the registered colormap."""
        cmap = colormap_registry.get("resistivity")

        assert isinstance(cmap, mcolors.Colormap)

    def test_get_unknown_colormap_raises(
        self, colormap_registry: ColormapRegistry
    ) -> None:
        """get should raise KeyError for unknown colormaps."""
        with pytest.raises(KeyError, match="Unknown colormap"):
            colormap_registry.get("nonexistent")

    def test_get_error_message_includes_available(
        self, colormap_registry: ColormapRegistry
    ) -> None:
        """get error message should list available colormaps."""
        with pytest.raises(KeyError) as exc_info:
            colormap_registry.get("invalid")

        assert "Available:" in str(exc_info.value)

    def test_has_registered(self, colormap_registry: ColormapRegistry) -> None:
        """has should return True for registered colormaps."""
        assert colormap_registry.has("resistivity") is True

    def test_has_unregistered(self, colormap_registry: ColormapRegistry) -> None:
        """has should return False for unregistered colormaps."""
        assert colormap_registry.has("nonexistent") is False

    def test_register_overwrites_existing(
        self, colormap_registry: ColormapRegistry
    ) -> None:
        """Registering with same name should overwrite."""
        cmap1 = plt.cm.gray
        cmap2 = plt.cm.viridis

        colormap_registry.register("test_overwrite", cmap1)
        colormap_registry.register("test_overwrite", cmap2)

        # Should be the second one
        retrieved = colormap_registry.get("test_overwrite")
        assert retrieved == cmap2


class TestColormapApply:
    """Tests for applying colormaps to images."""

    def test_apply_returns_bytes(
        self,
        colormap_registry: ColormapRegistry,
        sample_grayscale_image: np.ndarray,
    ) -> None:
        """apply should return image bytes."""
        result = colormap_registry.apply("resistivity", sample_grayscale_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_apply_png_format(
        self,
        colormap_registry: ColormapRegistry,
        sample_grayscale_image: np.ndarray,
    ) -> None:
        """apply should produce valid PNG by default."""
        result = colormap_registry.apply("resistivity", sample_grayscale_image)

        # PNG signature
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_apply_jpeg_format(
        self,
        colormap_registry: ColormapRegistry,
        sample_grayscale_image: np.ndarray,
    ) -> None:
        """apply should produce valid JPEG when requested."""
        result = colormap_registry.apply(
            "resistivity", sample_grayscale_image, output_format="jpeg"
        )

        # JPEG signature
        assert result[:2] == b"\xff\xd8"

    def test_apply_different_colormaps_produce_different_output(
        self,
        colormap_registry: ColormapRegistry,
        sample_grayscale_image: np.ndarray,
    ) -> None:
        """Different colormaps should produce different images."""
        result1 = colormap_registry.apply("resistivity", sample_grayscale_image)
        result2 = colormap_registry.apply("viridis", sample_grayscale_image)

        assert result1 != result2

    def test_apply_grayscale(
        self,
        colormap_registry: ColormapRegistry,
        sample_grayscale_image: np.ndarray,
    ) -> None:
        """apply with gray colormap should work."""
        result = colormap_registry.apply("gray", sample_grayscale_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_apply_unknown_colormap_raises(
        self,
        colormap_registry: ColormapRegistry,
        sample_grayscale_image: np.ndarray,
    ) -> None:
        """apply should raise KeyError for unknown colormaps."""
        with pytest.raises(KeyError):
            colormap_registry.apply("invalid", sample_grayscale_image)

    def test_apply_small_image(self, colormap_registry: ColormapRegistry) -> None:
        """apply should work with very small images."""
        tiny_image = np.array([[0, 128], [255, 64]], dtype=np.uint8)

        result = colormap_registry.apply("resistivity", tiny_image)

        assert isinstance(result, bytes)


class TestModuleSingleton:
    """Tests for the module-level singleton."""

    def test_module_singleton_exists(self) -> None:
        """Module should export a singleton registry."""
        assert colormap_registry is not None
        assert isinstance(colormap_registry, ColormapRegistry)

    def test_module_singleton_has_defaults(self) -> None:
        """Module singleton should have default colormaps."""
        assert colormap_registry.has("resistivity")
        assert colormap_registry.has("viridis")
