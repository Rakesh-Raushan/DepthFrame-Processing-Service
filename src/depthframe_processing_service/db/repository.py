"""SQLite repository for resized image storage and retrieval.

- Per-row BLOB storage (150 bytes /row as uint8) for efficient read/write of resized frames not 150 columns.
- Depth as PRIMARY KEY with index for O(log n) range queries.
- Metadata table for provenance and processing parameters
- Repository pattern: all SQL is encapsulated here; swap to PostgreSQL or another DB by changing this module only.
-  Thread safety assumption: SQLite connections should not be shared across threads. For production use with multiple workers, instantiate one repository per thread/request or use connection pooling.
- For larger datasets, consider partitioning or sharding strategies to maintain performance or move to a more scalable database solution.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanRow:
    """A single depth scanline with its pixel data."""

    depth: float
    pixel_data: (
        np.ndarray
    )  # uint8 array of shape (width,) representing the resized scanline


class ImageRepository:
    """Repository for storing and retrieving resized depth frames in SQLite."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # Connection management
    def connect(self):
        """Open (or create) the database and ensure schema exists."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        # WAL mode for better concurrent read performance
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info("Connected to database: %s", self._db_path)

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Database connection closed.")

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Repository not connected. Call connect() first.")
        return self._conn

    def __enter__(self) -> ImageRepository:
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ── Schema ──

    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS image_scans (
                depth       REAL PRIMARY KEY,
                pixel_data  BLOB NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_depth
                ON image_scans(depth);

            CREATE TABLE IF NOT EXISTS metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    # Write operations

    def bulk_insert_scans(self, depths: np.ndarray, pixel_array: np.ndarray) -> int:
        """Insert resized scan rows in a single transaction.

        Args:
            depths: 1D array of depth values, shape (n_rows,)
            pixel_array: 2D uint8 array, shape (n_rows, width)

        Returns:
            Number of rows inserted.
        """
        if depths.shape[0] != pixel_array.shape[0]:
            raise ValueError(
                f"Shape mismatch: {depths.shape[0]} depths vs "
                f"{pixel_array.shape[0]} pixel rows"
            )
        if pixel_array.dtype != np.uint8:
            raise TypeError(f"Expected uint8 pixel data, got {pixel_array.dtype}")

        rows = [
            (float(depths[i]), pixel_array[i].tobytes()) for i in range(len(depths))
        ]

        self.conn.executemany(
            "INSERT OR REPLACE INTO image_scans (depth, pixel_data) VALUES (?, ?)",
            rows,
        )
        self.conn.commit()
        logger.info("Inserted %d scan rows", len(rows))
        return len(rows)

    def set_metadata(self, key: str, value: str) -> None:
        """Store a metadata key-value pair."""
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    # Read operations

    def query_depth_range(self, depth_min: float, depth_max: float) -> list[ScanRow]:
        """Retrieve scan rows within a depth range (inclusive).

        Returns rows ordered by depth ascending.
        """
        cursor = self.conn.execute(
            "SELECT depth, pixel_data FROM image_scans "
            "WHERE depth >= ? AND depth <= ? ORDER BY depth",
            (depth_min, depth_max),
        )
        return [
            ScanRow(
                depth=row[0],
                pixel_data=np.frombuffer(row[1], dtype=np.uint8),
            )
            for row in cursor.fetchall()
        ]

    def get_metadata(self, key: str) -> Optional[str]:
        """Retrieve a metadata value by key."""
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def get_all_metadata(self) -> dict[str, str]:
        """Retrieve all metadata as a dictionary."""
        rows = self.conn.execute("SELECT key, value FROM metadata").fetchall()
        return dict(rows)

    def get_depth_bounds(self) -> tuple[float, float]:
        """Return (min_depth, max_depth) from stored data."""
        row = self.conn.execute(
            "SELECT MIN(depth), MAX(depth) FROM image_scans"
        ).fetchone()
        if row[0] is None:
            raise ValueError("No scan data in database")
        return (row[0], row[1])

    def get_row_count(self) -> int:
        """Return total number of stored scan rows."""
        row = self.conn.execute("SELECT COUNT(*) FROM image_scans").fetchone()
        return int(row[0])

    def is_populated(self) -> bool:
        """Check if the database has any scan data."""
        return self.get_row_count() > 0
