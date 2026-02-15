"""Application configuration.

All settings are configurable via environment variables (prefixed DPS_ for DepthFrame-Processing-Service)
or a .env file. Uses a simple dataclass for configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Service configuration with sensible defaults."""

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    db_path: Path = field(default_factory=lambda: Path("data/image_store.db"))
    csv_filename: str = "Challenge2.csv"

    # Image processing settings
    original_width: int = 200
    target_width: int = 150
    interpolation_method: str = "AREA"

    # API settings
    host: str = "0.0.0.0"
    port: int = 8000
    default_colormap: str = "resistivity"

    # logging settings
    log_level: str = "INFO"

    @property
    def csv_path(self) -> Path:
        """Full path to the CSV file."""
        return self.data_dir / self.csv_filename


settings = Settings()
