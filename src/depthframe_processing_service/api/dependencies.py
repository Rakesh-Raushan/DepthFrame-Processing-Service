"""FastAPI dependency injection for shared resources.

Uses FastAPI's dependency system to manage lifecycle of:
- ImageRepository (database connection)
- ColormapRegistry (singleton, stateless after init)

The repository is connected at app startup and closed at shutdown
via the lifespan context manager. Individual request handlers receive
it via Depends().
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from depthframe_processing_service.colormaps.registry import (
    ColormapRegistry,
    colormap_registry,
)
from depthframe_processing_service.config import settings
from depthframe_processing_service.db.repository import ImageRepository

logger = logging.getLogger(__name__)

# Module-level state managed by lifespan
_repository: ImageRepository | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage app-level resources: open DB on startup, close on shutdown."""
    global _repository

    logger.info("Starting up: connecting to database: %s", settings.db_path)
    _repository = ImageRepository(settings.db_path)
    _repository.connect()

    if not _repository.is_populated():
        logger.warning("Database is empty. Run ingestion first: make ingest")

    yield

    logger.info("Shutting down: closing database connection")
    if _repository is not None:
        _repository.close()
        _repository = None


def get_repository() -> ImageRepository:
    """Dependency: provides the active ImageRepository instance.

    The repository is opened once at startup (via lifespan) and shared
    across requests. SQLite in WAL mode supports concurrent reads.
    """
    if _repository is None:
        raise RuntimeError(
            "Repository not initialized. App lifespan may not have started."
        )
    return _repository


def get_colormap_registry() -> ColormapRegistry:
    """Dependency: provides the ColormapRegistry singleton.

    Stateless after init: safe to share across requests without
    any concurrency concerns.
    """
    return colormap_registry
