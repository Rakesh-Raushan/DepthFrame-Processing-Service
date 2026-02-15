"""FastAPI application factory.

Creates the app with lifespan management, route registration,
CORS middleware, and OpenAPI customization.

"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from depthframe_processing_service.api.dependencies import lifespan
from depthframe_processing_service.api.routes import health_router, router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""

    app = FastAPI(
        title="DepthFrame Processing Service",
        description=(
            "REST API to query depth-range frames with custom colormap rendering.\n\n"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — permissive for development, tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # Register route groups
    app.include_router(router)
    app.include_router(health_router)

    return app


# Module-level app instance for uvicorn
app = create_app()
